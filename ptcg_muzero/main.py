"""
ptcg_muzero/main.py
====================
Point d'entrée principal — entraînement, évaluation, inférence Kaggle.

Modes
-----
  train       Lance la boucle d'entraînement complète.
  eval        Évalue un checkpoint contre un agent aléatoire.
  submit      Génère la fonction agent() à soumettre sur Kaggle.
  probe-diag  Affiche un rapport de précision des probing classifiers.

Flags communs
-------------
  --config     Chemin vers un config.json (optionnel, surcharge les défauts).
  --hf-repo    Surcharge HFConfig.repo_id.
  --no-hf      Désactive le push HuggingFace.
  --devices N  Nombre de GPUs (défaut : 2).
  --seed N     Graine aléatoire.
  --debug      Désactive JIT (pour debugger).

Exemples
--------
  python main.py train
  python main.py train --hf-repo my-org/ptcg-muzero --devices 2
  python main.py eval  --config checkpoints/config.json
  python main.py submit
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# NOTE: cabt / cg-lib path discovery is handled centrally in env.cabt_api
# (same glob pattern as the reference Kaggle notebook).


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ptcg_muzero.main")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PTCG MuZero — entraînement / évaluation / soumission Kaggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "mode",
        choices=["train", "eval", "submit", "probe-diag"],
        help="Mode d'exécution.",
    )
    p.add_argument("--config",   default=None,  help="Chemin vers config.json")
    p.add_argument("--hf-repo",  default=None,  help="Surcharge HFConfig.repo_id")
    p.add_argument("--no-hf",    action="store_true", help="Désactive le push HF")
    p.add_argument("--devices",  type=int, default=None, help="Nombre de GPUs")
    p.add_argument("--seed",     type=int, default=None, help="Graine aléatoire")
    p.add_argument("--debug",    action="store_true",    help="Désactive JIT")
    p.add_argument("--ckpt",     default=None, help="Chemin checkpoint (eval/probe-diag)")
    p.add_argument("--eval-games", type=int, default=None, help="Nb parties d'éval")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de configuration
# ─────────────────────────────────────────────────────────────────────────────
def load_config(args) -> "Config":
    from config import Config
    if args.config and Path(args.config).exists():
        cfg = Config.load(args.config)
        logger.info("Config chargée depuis %s", args.config)
    else:
        cfg = Config()
        logger.info("Config par défaut")

    # Surcharges CLI
    if args.hf_repo:
        cfg.hf.repo_id = args.hf_repo
    if args.no_hf:
        cfg.hf.enabled = False
    if args.devices:
        cfg.infra.num_devices = args.devices
    if args.seed is not None:
        cfg.infra.seed = args.seed
    if args.debug:
        cfg.infra.debug_no_jit = True
        os.environ["JAX_DISABLE_JIT"] = "1"

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Mode : train
# ─────────────────────────────────────────────────────────────────────────────
def cmd_train(args) -> None:
    cfg = load_config(args)
    _check_devices(cfg)

    # Sauvegarde config pour reproductibilité
    Path(cfg.infra.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(Path(cfg.infra.checkpoint_dir) / "config.json")

    # Import du tracker d'activité explicite et propre
    import threading
    import time
    from training.activity import tracker, dump_all_stacks

    # Thread heartbeat basé sur le temps réel (toutes les 15 secondes)
    def _heartbeat():
        while True:
            time.sleep(15.0)
            now = time.time()
            inactive_dur = now - tracker.last_activity_time
            
            target_size = cfg.train.min_replay_size if tracker.buffer_size < cfg.train.min_replay_size else cfg.train.replay_buffer_size
            
            # Détection de freeze (aucune mise à jour d'activité depuis plus de 240s)
            if inactive_dur > 240.0:
                freeze_warning = " ⚠️ ATTENTION : Activité suspecte, possible freeze !"
                logger.warning(
                    "[heartbeat] Phase: %s | Buffer: %d/%d | Erreurs Deck: %d | Étape jeu en cours: %d | Inactif depuis: %.1fs%s",
                    tracker.phase, tracker.buffer_size, target_size, tracker.deck_errors, tracker.current_game_steps, inactive_dur, freeze_warning
                )
                try:
                    dump_all_stacks()
                except Exception as e:
                    logger.error("Impossible de dumper la stack trace : %s", e)
            else:
                logger.info(
                    "[heartbeat] Phase: %s | Buffer: %d/%d | Erreurs Deck: %d | Étape jeu en cours: %d | Inactif depuis: %.1fs",
                    tracker.phase, tracker.buffer_size, target_size, tracker.deck_errors, tracker.current_game_steps, inactive_dur
                )
            
    h_thread = threading.Thread(target=_heartbeat, daemon=True)
    h_thread.start()

    from training.trainer import train
    train(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Mode : eval
# ─────────────────────────────────────────────────────────────────────────────
def cmd_eval(args) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    cfg = load_config(args)
    if args.eval_games:
        cfg.train.eval_games = args.eval_games

    # Charge checkpoint
    params, deck_params = _load_checkpoint(args.ckpt, cfg)

    from cards.encoder import CardStaticFeatures
    from models.networks import MuZeroNetwork
    from models.deck_builder import DeckBuilderNetwork, sample_deck, set_energy_ids, set_ace_spec_ids
    from training.trainer import make_agent_fn
    from env.wrapper import run_self_play_game

    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    energy_ids = [
        cid for cid in card_data.card_ids
        if "Energy" in card_data._cards[cid].get("stage", "")
    ]
    set_energy_ids(energy_ids)
    set_ace_spec_ids(card_data.ace_spec_ids)

    network  = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    deck_net = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)

    rng = jax.random.PRNGKey(cfg.infra.seed)
    wins = 0
    total = cfg.train.eval_games

    for g in range(total):
        rng, rng_d, rng_ag = jax.random.split(rng, 3)
        deck_logits, _ = deck_net.apply(deck_params)
        deck0, _ = sample_deck(deck_logits[0], rng_d, num_card_ids, energy_ids)
        # Agent 1 : aléatoire
        deck1 = deck0[:]  # même deck, adversaire aléatoire

        agent = make_agent_fn(network, params["muzero"], cfg, rng_ag, train_mode=False)

        def random_agent(obs_dict, player_idx, _cfg):
            opts = (obs_dict.get("select") or {}).get("option", [])
            n = max(len(opts), 1)
            return [np.random.randint(0, n)], np.ones(cfg.model.max_actions) / cfg.model.max_actions, 0.0

        from env.wrapper import run_self_play_game, GameHistory
        h0, h1 = run_self_play_game(
            (agent, random_agent), deck0, deck1, cfg, np.random.default_rng()
        )
        if h0.game_won:
            wins += 1
        logger.info("Game %d/%d — won=%s", g + 1, total, h0.game_won)

    wr = wins / total
    logger.info("=== Eval result : %d/%d  win_rate=%.2f ===", wins, total, wr)


# ─────────────────────────────────────────────────────────────────────────────
# Mode : submit  (génère agent.py pour Kaggle)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_submit(args) -> None:
    """
    Génère un fichier agent_submit.py qui peut être soumis directement
    sur la compétition Kaggle PTCG.  Il télécharge les poids depuis le Hub.
    """
    cfg = load_config(args)
    out = Path("agent_submit.py")
    out.write_text(_SUBMIT_TEMPLATE.format(
        repo_id=cfg.hf.repo_id,
        num_card_ids=cfg.model.num_card_ids,
        latent_dim=cfg.model.latent_dim,
    ))
    logger.info("Agent de soumission généré → %s", out)
    logger.info("Assurez-vous que '%s' est public sur le Hub.", cfg.hf.repo_id)


_SUBMIT_TEMPLATE = '''"""
PTCG MuZero — agent de soumission Kaggle.
Générée automatiquement par main.py submit.
"""
import glob, sys
# Replicate the exact path setup from the reference Kaggle notebook:
#   sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0])
_cg_hits = glob.glob('/kaggle/input/**/cg-lib', recursive=True)
if _cg_hits:
    sys.path.append(_cg_hits[0])

HF_REPO = "{repo_id}"

def _load():
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    import json, jax, jax.numpy as jnp

    mz_path  = hf_hub_download(HF_REPO, "muzero.safetensors")
    cfg_path = hf_hub_download(HF_REPO, "config.json")
    dk_path  = hf_hub_download(HF_REPO, "deck_builder.safetensors")

    from export.hub import _unflatten_params
    from config import Config
    cfg = Config.from_json(open(cfg_path).read())

    mz_params  = _unflatten_params(load_file(mz_path))
    dk_params  = _unflatten_params(load_file(dk_path))
    return mz_params, dk_params, cfg

_mz_params, _dk_params, _cfg = _load()

from cards.encoder import CardStaticFeatures
from models.networks import MuZeroNetwork
from models.deck_builder import DeckBuilderNetwork, sample_deck, set_energy_ids
import jax, jax.numpy as jnp

_card_data = CardStaticFeatures("/kaggle/input/cards.csv")
_n = max(_card_data.max_card_id + 1, _cfg.model.num_card_ids)
_cfg.model.num_card_ids = _n
_static = jnp.array(_card_data.feature_matrix(_n))
_energy_ids = [c for c in _card_data.card_ids
               if "Energy" in _card_data._cards[c].get("stage","")]
set_energy_ids(_energy_ids)

_network  = MuZeroNetwork(cfg=_cfg.model, static_features=_static)
_deck_net = DeckBuilderNetwork(cfg=_cfg.model, static_features=_static)

_rng = jax.random.PRNGKey(42)
_logits, _ = _deck_net.apply(_dk_params)
_deck, _   = sample_deck(_logits[0], _rng, _n, _energy_ids, temperature=0.1)

from cg.game import battle_start, battle_select, battle_finish
from search.ismcts import ismcts_action
from env.encoding import encode_observation

def agent(obs_dict):
    global _rng
    _rng, rng_act = jax.random.split(_rng)
    select  = obs_dict.get("select") or {{}}
    options = select.get("option", [])
    if not options:
        return []
    enc = encode_observation(obs_dict, obs_dict["current"]["yourIndex"], _cfg.model)
    mask = enc["option_mask"]
    best, _, _ = ismcts_action(_network, _mz_params, enc, mask, rng_act, _cfg)
    max_cnt = int(select.get("maxCount", 1))
    import numpy as np
    if max_cnt > 1:
        return sorted(np.argsort(-np.where(mask, enc["option_mask"].astype(float), -1e9))[:max_cnt].tolist())
    return [best]

def deck_builder():
    return _deck
'''


# ─────────────────────────────────────────────────────────────────────────────
# Mode : probe-diag
# ─────────────────────────────────────────────────────────────────────────────
def cmd_probe_diag(args) -> None:
    """Affiche les précisions des probing classifiers sur un checkpoint."""
    import jax.numpy as jnp
    import numpy as np

    cfg = load_config(args)
    params, _ = _load_checkpoint(args.ckpt, cfg)

    from cards.encoder import CardStaticFeatures
    from models.networks import MuZeroNetwork
    from interpretability.probes import (
        ProbeHeads, probe_accuracy, probe_report, extract_probe_targets
    )

    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    probes  = ProbeHeads(cfg=cfg.model)

    # Génère quelques obs dummy pour le diagnostic
    from training.trainer import _make_dummy_obs
    dummy = _make_dummy_obs(cfg.model)
    obs_batch = {k: jnp.array(v[None]) for k, v in dummy.items()}

    z = network.apply(params["muzero"], obs_batch, method=network.represent)
    probe_logits = probes.apply(params["probes"], z)

    # Targets factices (all 0) juste pour vérifier les shapes
    import numpy as np
    tgts = jnp.zeros((1, 5), dtype=jnp.int32)
    accs = np.array(probe_accuracy(probe_logits, tgts))
    losses = np.zeros(5)
    print(probe_report(accs, losses))
    print("\n(Précisions sur targets=0 factices — lancez sur vraies parties pour des métriques réelles)")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────
def _check_devices(cfg: "Config") -> None:
    import jax
    n = len(jax.devices())
    if n < cfg.infra.num_devices:
        logger.warning(
            "%d GPU(s) demandé(s) mais seulement %d disponible(s). "
            "Ajustement automatique.",
            cfg.infra.num_devices, n,
        )
        cfg.infra.num_devices = n


def _load_checkpoint(ckpt_path: str | None, cfg: "Config") -> tuple:
    import pickle
    import numpy as np
    import jax

    if ckpt_path and Path(ckpt_path).exists():
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
        logger.info("Checkpoint chargé : %s  (step=%d)", ckpt_path, data.get("step", -1))
        return (
            jax.tree_util.tree_map(jax.device_put, data["params"]),
            jax.tree_util.tree_map(jax.device_put, data.get("deck", {})),
        )

    logger.warning("Aucun checkpoint fourni — paramètres aléatoires.")
    return {}, {}


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "train":      cmd_train,
        "eval":       cmd_eval,
        "submit":     cmd_submit,
        "probe-diag": cmd_probe_diag,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
