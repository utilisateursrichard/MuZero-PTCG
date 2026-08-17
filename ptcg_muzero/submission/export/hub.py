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
import sys
import threading
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


def verify_hf_token(token: str | None) -> bool:
    """Vérifie si le token HuggingFace est valide auprès de l'API Hugging Face."""
    if not token or not token.strip():
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.whoami(token=token)
        return True
    except Exception as e:
        err_str = str(e).lower()
        if "401" in err_str or "unauthorized" in err_str or "invalid" in err_str or "token" in err_str:
            logger.error("Token HF invalide ou rejeté par l'API Hugging Face : %s", e)
            return False
        logger.warning("Validation du token HF impossible auprès du serveur (erreur réseau ou serveur) : %s", e)
        return True


def get_hf_token(
    cfg: Config | None = None,
    token_env_var: str = "HF_TOKEN",
    required: bool = False,
) -> str | None:
    """Récupère et valide le token HuggingFace depuis l'environnement, huggingface_hub ou Kaggle Secrets.

    Si HF est activé (cfg.hf.enabled) ou si required=True, le script s'arrête (sys.exit(1))
    immédiatement si le token est inconnu ou invalide.
    """
    should_require = required or (cfg is not None and hasattr(cfg, "hf") and getattr(cfg.hf, "enabled", False))

    env_var = cfg.hf.token_env_var if (cfg and hasattr(cfg, "hf") and hasattr(cfg.hf, "token_env_var")) else token_env_var
    token = os.environ.get(env_var, "") or os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass
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

    if not token or not token.strip():
        if should_require:
            logger.error(
                "ERREUR FATALE : Token Hugging Face inconnu (variable '%s', HF_TOKEN ou Secret Kaggle absent). Interruption du script.",
                env_var,
            )
            sys.exit(1)
        return None

    token = token.strip()
    if not verify_hf_token(token):
        if should_require:
            logger.error(
                "ERREUR FATALE : Token Hugging Face invalide/inconnu (variable '%s' ou HF_TOKEN rejeté). Interruption du script.",
                env_var,
            )
            sys.exit(1)
        return None

    return token


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

        # Upload de tous les fichiers du répertoire dans le snapshot immuable
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=cfg.hf.repo_id,
            path_in_repo=f"step_{step:07d}",
            commit_message=f"Training checkpoint — step {step}",
            token=token,
        )

        # Upload également des fichiers à la racine du repo pour accès direct
        for fname in ["muzero.safetensors", "deck_builder.safetensors", "config.json", "meta.json", "README.md"]:
            fpath = out_dir / fname
            if fpath.exists():
                try:
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=fname,
                        repo_id=cfg.hf.repo_id,
                        commit_message=f"Update root {fname} — step {step}",
                        token=token,
                    )
                except Exception as e_root:
                    logger.debug("Upload racine %s : %s", fname, e_root)

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
) -> tuple[dict | None, dict | None, Config, int]:
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
        logger.warning("Installez huggingface_hub et safetensors pour le téléchargement Hub.")
        return None, None, cfg or Config(), 0

    if not token:
        token = get_hf_token(cfg, required=cfg.hf.enabled if cfg else True)
    else:
        if not verify_hf_token(token):
            logger.error("ERREUR FATALE : Le token Hugging Face fourni est invalide/inconnu. Interruption du script.")
            sys.exit(1)

    try:
        if step is None:
            try:
                latest = hf_hub_download(repo_id=repo_id, filename="latest.json", token=token)
                step = int(json.loads(Path(latest).read_text())["step"])
            except Exception:
                pass

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
    except Exception as exc:
        logger.warning("[hf-hub] Impossible de télécharger le checkpoint depuis HF Hub (%s): %s", repo_id, exc)
        return None, None, cfg or Config(), 0


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


# ─────────────────────────────────────────────────────────────────────────────
# Async replay buffer push
# ─────────────────────────────────────────────────────────────────────────────
_buffer_push_thread: threading.Thread | None = None
_buffer_push_lock = threading.Lock()


def push_buffer_to_hub_async(
    buffer,
    cfg: Config,
    step: int,
) -> None:
    """Serialize the replay buffer and upload it to HuggingFace Hub **asynchronously**.

    The heavy work (pickle + upload) runs on a background daemon thread so the
    training loop is never blocked.  Only one upload can be in-flight at a time:
    if a previous push is still running the new request is silently skipped.
    """
    global _buffer_push_thread

    if not cfg.hf.enabled or not cfg.hf.repo_id:
        return

    with _buffer_push_lock:
        if _buffer_push_thread is not None and _buffer_push_thread.is_alive():
            logger.info(
                "[hf-buffer] Push précédent encore en cours — skip step %d.",
                step,
            )
            return

    # Serialize on the main thread (fast snapshot under the buffer lock)
    local_dir = Path(cfg.hf.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    buf_path = str(local_dir / "replay_buffer.pkl")
    meta_path = str(local_dir / "buffer_meta.json")
    try:
        buffer.serialize(buf_path, step=step)
    except Exception as exc:
        logger.error("[hf-buffer] Serialize failed (step=%d): %s", step, exc)
        return

    buf_size = len(buffer)
    max_size = getattr(buffer, "_max_size", buf_size)
    fill_pct = round(100.0 * buf_size / max(max_size, 1), 2)

    def _upload():
        try:
            from huggingface_hub import HfApi, create_repo

            token = get_hf_token(cfg)
            if not token:
                logger.warning("[hf-buffer] Pas de token HF — upload buffer ignoré.")
                return

            api = HfApi(token=token)
            try:
                create_repo(cfg.hf.repo_id, private=cfg.hf.private,
                            exist_ok=True, token=token)
            except Exception:
                pass

            # Upload metadata file first
            if os.path.exists(meta_path):
                try:
                    api.upload_file(
                        path_or_fileobj=meta_path,
                        path_in_repo="buffer_meta.json",
                        repo_id=cfg.hf.repo_id,
                        commit_message=f"Replay buffer metadata — step {step} ({buf_size}/{max_size} entries, {fill_pct}%)",
                        token=token,
                    )
                except Exception as meta_exc:
                    logger.debug("[hf-buffer] Failed to upload buffer_meta.json: %s", meta_exc)

            # Upload main pickle buffer
            api.upload_file(
                path_or_fileobj=buf_path,
                path_in_repo="replay_buffer.pkl",
                repo_id=cfg.hf.repo_id,
                commit_message=f"Replay buffer snapshot — step {step} ({buf_size}/{max_size} entries, {fill_pct}%)",
                token=token,
            )
            logger.info(
                "✓ Replay buffer pushed to HF Hub  repo=%s  step=%d  entries=%d/%d (%.1f%% rempli)",
                cfg.hf.repo_id, step, buf_size, max_size, fill_pct,
            )
        except Exception as exc:
            logger.error("[hf-buffer] Upload failed (step=%d): %s", step, exc)

    t = threading.Thread(target=_upload, daemon=True, name="hf-buffer-push")
    with _buffer_push_lock:
        _buffer_push_thread = t
    t.start()


def load_buffer_from_hub(
    repo_id: str,
    cfg_train,
    cfg_model,
    token: str | None = None,
    cfg: Config | None = None,
) -> tuple[Any | None, dict]:
    """Télécharge et restaure le replay buffer depuis HuggingFace Hub.

    Returns:
        tuple (buffer, meta_dict)
    """
    try:
        from huggingface_hub import hf_hub_download
        from training.replay_buffer import PrioritizedReplayBuffer
    except ImportError:
        logger.warning("[hf-buffer] huggingface_hub / PrioritizedReplayBuffer non disponible.")
        return None, {}

    if not token:
        token = get_hf_token(cfg, required=cfg.hf.enabled if cfg else True)
    else:
        if not verify_hf_token(token):
            logger.error("ERREUR FATALE : Le token Hugging Face fourni est invalide/inconnu. Interruption du script.")
            sys.exit(1)

    meta = {}
    # Tenter de télécharger le fichier de métadonnées buffer_meta.json
    try:
        meta_file = hf_hub_download(repo_id=repo_id, filename="buffer_meta.json", token=token)
        meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
        logger.info(
            "[hf-buffer] Métadonnées buffer trouvées sur HF Hub : étape %s, %s/%s entrées (%s%% rempli)",
            meta.get("step", "?"), meta.get("size", "?"), meta.get("max_size", "?"), meta.get("fill_percentage", "?")
        )
    except Exception as e:
        logger.debug("[hf-buffer] buffer_meta.json absent sur HF Hub : %s", e)

    # Télécharger le fichier pickle replay_buffer.pkl
    try:
        buf_file = hf_hub_download(repo_id=repo_id, filename="replay_buffer.pkl", token=token)
        buf = PrioritizedReplayBuffer.deserialize(buf_file, cfg_train, cfg_model)
        step_val = meta.get("step", getattr(buf, "loaded_step", 0))
        size_val = len(buf)
        max_size_val = buf._max_size
        fill_pct = round(100.0 * size_val / max(max_size_val, 1), 2)

        meta.setdefault("step", step_val)
        meta.setdefault("size", size_val)
        meta.setdefault("max_size", max_size_val)
        meta.setdefault("fill_percentage", fill_pct)

        logger.info(
            "=== Replay Buffer restauré depuis HF Hub (%s) : %d/%d entrées (%.1f%% rempli, étape %d) ===",
            repo_id, size_val, max_size_val, fill_pct, step_val
        )
        return buf, meta
    except Exception as e:
        logger.info("[hf-buffer] Échec du chargement du buffer depuis HF Hub (%s) : %s", repo_id, e)
        return None, meta

