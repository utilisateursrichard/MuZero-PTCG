"""
ptcg_muzero/export/hub.py
==========================
Export des paramètres vers HuggingFace Hub en format safetensors.

Flux
----
1. Aplatir le pytree de paramètres JAX en dict[str, np.ndarray]
2. Sauvegarder localement via safetensors
3. Pousser sur le Hub avec huggingface_hub si le flag est activé
4. Inclure config.json + model_card.md dans le dépôt

Variables d'environnement
--------------------------
HF_TOKEN (ou le nom configuré dans HFConfig.token_env_var)
    Token HuggingFace avec droit d'écriture sur le repo.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

import jax
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Aplatissement du pytree → dict plat
# ─────────────────────────────────────────────────────────────────────────────
def _flatten_params(params: dict, prefix: str = "") -> Dict[str, np.ndarray]:
    """
    Convertit récursivement un pytree de paramètres Flax en dict plat.

    Exemple :
        {"muzero": {"h": {"Dense_0": {"kernel": array}}}}
        → {"muzero/h/Dense_0/kernel": array}
    """
    flat = {}
    for k, v in params.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_params(v, prefix=key))
        else:
            flat[key] = np.array(v)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde locale
# ─────────────────────────────────────────────────────────────────────────────
def save_local(
    muzero_params: dict,
    deck_params:   dict,
    cfg:           Config,
    step:          int,
) -> Path:
    """
    Sauvegarde les paramètres au format safetensors dans local_dir.
    Retourne le chemin du répertoire créé.
    """
    try:
        from safetensors.numpy import save_file
    except ImportError:
        raise ImportError(
            "safetensors non installé. Lancez : pip install safetensors"
        )

    out_dir = Path(cfg.hf.local_dir) / f"step_{step:07d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Paramètres MuZero ─────────────────────────────────────────────────
    mz_flat = _flatten_params(muzero_params)
    save_file(mz_flat, str(out_dir / "muzero.safetensors"))

    # ── Paramètres Deck Builder ───────────────────────────────────────────
    dk_flat = _flatten_params(deck_params, prefix="deck")
    save_file(dk_flat, str(out_dir / "deck_builder.safetensors"))

    # ── config.json ───────────────────────────────────────────────────────
    (out_dir / "config.json").write_text(cfg.to_json())

    # ── meta.json ─────────────────────────────────────────────────────────
    (out_dir / "meta.json").write_text(json.dumps({"step": step}))

    # ── model_card.md ─────────────────────────────────────────────────────
    (out_dir / "README.md").write_text(_generate_model_card(cfg, step))

    logger.info("Saved checkpoint locally → %s", out_dir)
    return out_dir


def get_hf_token(cfg: Config | None = None, token_env_var: str = "HF_TOKEN") -> str | None:
    """Récupère le token HuggingFace depuis les variables d'environnement ou Kaggle Secrets."""
    env_var = cfg.hf.token_env_var if (cfg and hasattr(cfg, "hf") and hasattr(cfg.hf, "token_env_var")) else token_env_var
    token = os.environ.get(env_var, "") or os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret(env_var)
            if not token and env_var != "HF_TOKEN":
                token = user_secrets.get_secret("HF_TOKEN")
            if token:
                logger.info("Token HF récupéré depuis Kaggle Secrets.")
        except Exception:
            pass
    return token if token else None


# ─────────────────────────────────────────────────────────────────────────────
# Push vers HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────
def push_to_hub(
    muzero_params: dict,
    deck_params:   dict,
    cfg:           Config,
    step:          int,
) -> bool:
    """
    Sauvegarde localement puis pousse vers le Hub.
    Publie d'abord un snapshot immuable puis son pointeur ``latest.json``.
    Le pointeur n'est écrit qu'après succès du snapshot, afin qu'un lecteur ne
    puisse jamais charger un mélange de fichiers provenant de deux étapes.
    """
    if not cfg.hf.enabled:
        return False

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.warning(
            "huggingface_hub non installé — push désactivé. "
            "Lancez : pip install huggingface_hub"
        )
        return False

    token = get_hf_token(cfg)
    if not token:
        logger.warning(
            "HF_TOKEN absent (variable '%s' ou Secret Kaggle absent) — push ignoré.",
            cfg.hf.token_env_var,
        )
        return False

    try:
        # Sauvegarde locale
        out_dir = save_local(muzero_params, deck_params, cfg, step)

        # Création du repo si nécessaire
        api = HfApi(token=token)
        try:
            create_repo(
                cfg.hf.repo_id,
                private=cfg.hf.private,
                exist_ok=True,
                token=token,
            )
        except Exception as e:
            logger.debug("create_repo (peut être déjà existant) : %s", e)

        # Upload de tous les fichiers du répertoire
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=cfg.hf.repo_id,
            path_in_repo=f"step_{step:07d}",
            commit_message=f"Training checkpoint — step {step}",
            token=token,
        )

        # Publish the pointer only after the complete, immutable snapshot is
        # available on the Hub.  One file/one commit makes this atomic.
        latest_path = out_dir / "latest.json"
        latest_path.write_text(json.dumps({"step": step, "path": f"step_{step:07d}"}))
        api.upload_file(
            path_or_fileobj=str(latest_path),
            path_in_repo="latest.json",
            repo_id=cfg.hf.repo_id,
            commit_message=f"Latest checkpoint pointer — step {step}",
            token=token,
        )

        logger.info("✓ Pushed to HuggingFace Hub  repo=%s  step=%d",
                    cfg.hf.repo_id, step)
        return True

    except Exception as exc:
        logger.error("HF push failed (step=%d): %s", step, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Chargement depuis le Hub (inférence / reprise)
# ─────────────────────────────────────────────────────────────────────────────
def load_from_hub(
    repo_id: str,
    step:    int | None = None,
    token:   str | None = None,
    cfg:     Config | None = None,
) -> tuple[dict, dict, Config, int]:
    """
    Télécharge les poids depuis le Hub et retourne
    (muzero_params, deck_params, cfg, step).

    Si step=None, charge le snapshot désigné par ``latest.json``. Les anciens
    dépôts sans pointeur restent compatibles via les fichiers à la racine.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file
    except ImportError:
        raise ImportError("Installez : pip install huggingface_hub safetensors")

    if not token:
        token = get_hf_token(cfg)

    if step is None:
        try:
            latest = hf_hub_download(repo_id=repo_id, filename="latest.json", token=token)
            step = int(json.loads(Path(latest).read_text())["step"])
        except Exception:
            logger.warning("latest.json absent : fallback vers le format HF historique à la racine.")

    subfolder = f"step_{step:07d}" if step is not None else ""

    def _dl(filename):
        return hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/{filename}" if subfolder else filename,
            token=token,
        )

    mz_path  = _dl("muzero.safetensors")
    dk_path  = _dl("deck_builder.safetensors")
    cfg_path = _dl("config.json")

    step_val = step if step is not None else 0
    try:
        meta_path = _dl("meta.json")
        meta = json.loads(Path(meta_path).read_text())
        step_val = meta.get("step", step_val)
    except Exception:
        pass

    mz_flat  = load_file(mz_path)
    dk_flat  = load_file(dk_path)
    loaded_cfg = Config.from_json(Path(cfg_path).read_text())

    muzero_params = _unflatten_params(mz_flat)
    deck_params   = _unflatten_params(dk_flat)

    return (
        jax.tree_util.tree_map(jax.device_put, muzero_params),
        jax.tree_util.tree_map(jax.device_put, deck_params),
        loaded_cfg,
        step_val,
    )


def _unflatten_params(flat: Dict[str, np.ndarray]) -> dict:
    """Inverse de _flatten_params : dict plat → pytree."""
    nested: dict = {}
    for key, val in flat.items():
        parts = key.split("/")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return nested


# ─────────────────────────────────────────────────────────────────────────────
# Model card generator
# ─────────────────────────────────────────────────────────────────────────────
def _generate_model_card(cfg: Config, step: int) -> str:
    return f"""---
language: en
tags:
  - reinforcement-learning
  - muzero
  - pokemon-tcg
  - jax
license: mit
---

# PTCG MuZero Agent

MuZero agent for the Pokémon Trading Card Game (PTCG), trained via hybrid
**ISMCTS** (belief sampling) + **collapsed chance nodes** for stochastic
transitions.

## Architecture

| Component | Details |
|-----------|---------|
| Representation | Transformer encoder ({cfg.model.num_enc_layers} layers, d={cfg.model.latent_dim}) |
| Prediction | Two-head MLP (policy + value) |
| Dynamics | Residual MLP with stochastic branch |
| Card embedding | Learned ({cfg.model.card_embed_dim}d) + static CSV features (48d) |
| Search | Gumbel MuZero + ISMCTS ({cfg.search.num_belief_samples} determinisations) |
| Interpretability | 5 linear probing classifiers |

## Training

- **Framework**: JAX / Flax  
- **Devices**: {cfg.infra.num_devices}× GPU (data-parallel via jax.pmap)  
- **Step**: {step:,}  
- **Batch size**: {cfg.train.batch_size}  
- **LR**: {cfg.train.learning_rate}  

## Files

| File | Content |
|------|---------|
| `muzero.safetensors` | MuZero network weights (h + f + g + probes) |
| `deck_builder.safetensors` | Deck builder policy weights |
| `config.json` | Full training configuration |
"""
