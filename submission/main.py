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
        description="PTCG MuZero — training / evaluation / Kaggle submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "mode",
        choices=["train", "eval", "test", "submit", "probe-diag"],
        help="Execution mode.",
    )

    # ── Arguments globaux (train, test, eval, submit) ────────────────────────
    p.add_argument("-s", "--safetensors", default=None,
                   help="Weights or checkpoint to load: path to .safetensors/.pkl, or a HuggingFace reference — "
                        "'HF' (latest checkpoint of cfg.hf.repo_id), 'hf@120000' (specific step), "
                        "'hf:owner/repo', 'hf:owner/repo@120000'.")
    p.add_argument("--step",     type=int, default=None, help="Specific HF step number to load (e.g. --step 135000)")
    p.add_argument("--ckpt",     default=None, help="Explicit local checkpoint path (.pkl/.safetensors)")

    # ── Mode test : mesurer la FORCE du modèle contre un étalon extérieur ────
    # En self-play le taux de victoire vaut 50 % par construction (même réseau,
    # même deck des deux côtés) : aucune courbe d'entraînement ne dit si l'agent
    # joue mieux.  Ce mode fournit le signal manquant.
    t = p.add_argument_group("test mode")
    t.add_argument("-o", "--opponent", default="greedy",
                   choices=["greedy", "random", "self", "checkpoint"],
                   help="Opponent. 'greedy' = heuristic by option type (stable baseline), "
                        "'random' = baseline floor, 'self' = mirror of tested model, "
                        "'checkpoint' = other weights via --opponent-weights.")
    t.add_argument("--opponent-weights", default=None,
                   help="Opponent weights when --opponent=checkpoint. Accepts the same "
                        "HF references as -s: e.g. 'hf@120000' to battle an earlier "
                        "step of the same repository (AlphaZero-style relative progression).")
    t.add_argument("-e", "--epsilon", type=float, default=0.0,
                   help="Greedy opponent difficulty knob: 0.0 = full strength, "
                        "1.0 = equivalent to random. Tuning up to achieve a 50 %% "
                        "winrate measures network strength.")
    t.add_argument("-n", "--games", type=int, default=20, help="Number of games.")
    t.add_argument("--sims", type=int, default=None,
                   help="MCTS simulations (overrides config; lowering speeds up test).")
    t.add_argument("--belief", type=int, default=None, help="ISMCTS belief samples.")
    t.add_argument("--no-swap", action="store_true",
                   help="Do not alternate sides. Default alternates to neutralize "
                        "first-player advantage.")
    t.add_argument("--json-out", default=None, help="Write JSON report to this path.")
    t.add_argument("--policy-only", action="store_true",
                   help="Run evaluation with raw policy network (0 MCTS simulations). Fast and tests intuitive prior.")
    t.add_argument("--iree", action="store_true", help="Use IREE acceleration (Vulkan GPU / CPU VMFB) for ultra-fast testing.")
    t.add_argument("--device", "--device-uri", default="vulkan", choices=["vulkan", "cpu"], help="Hardware device for IREE acceleration ('vulkan' or 'cpu').")
    t.add_argument("--vmfb", default=None, help="Path to compiled .vmfb module (compiled automatically from HF if missing).")
    p.add_argument("--config",   default=None,  help="Path to config.json")

    p.add_argument("--hf-repo",  default=None,  help="Override HFConfig.repo_id")
    p.add_argument("--no-hf",    action="store_true", help="Disable HF push")
    p.add_argument("-w", "--workers", type=int, default=None, help="Number of CPU self-play worker processes (default: auto-scale based on os.cpu_count)")

    p.add_argument("--devices",  type=int, default=None, help="Number of GPUs")

    p.add_argument("--seed",     type=int, default=None, help="Random seed")

    p.add_argument("--hot-fix", "--hot_fix", action="store_true", help="Hot-Fix mode: resets f (policy/val), g (50D dynamics), Adam and Replay Buffer while keeping h(s) at 94%%.")
    p.add_argument("--reset-policy", action="store_true", help="Resets the full policy/value head (f)")
    p.add_argument("--reset-value-head", action="store_true",
                   help="Surgical reset: v_dense + rdet_fc2 only (preserves h, policy, and transition g). "
                        "Use with --fresh-buffer after value target fixes.")
    p.add_argument("--fresh-buffer", action="store_true", help="Start with an empty Replay Buffer (ignores older passive games)")
    p.add_argument("--freeze-h-steps", type=int, default=None, help="Number of steps with h(s) 100%% frozen")
    p.add_argument("--unfreeze-ramp-steps", type=int, default=None, help="Number of transition steps for gradual unfreezing of h(s)")
    p.add_argument("--no-reward-shaping", action="store_true", help="Disable reward shaping (reverts to sparse ±1.0 rewards)")
    p.add_argument("--debug",    action="store_true",    help="Disable JIT")
    p.add_argument("--eval-games", type=int, default=None, help="Number of eval games")
    p.add_argument("--out-dir",  default="submission", help="Output directory for submission")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de configuration
# ─────────────────────────────────────────────────────────────────────────────
def load_config(args) -> "Config":
    from config import Config
    if args.config and Path(args.config).exists():
        cfg = Config.load(args.config)
        logger.info("Config loaded from %s", args.config)
    else:
        cfg = Config()
        logger.info("Default config")

    # Surcharges de checkpoint et reprise
    if getattr(args, "safetensors", None):
        parsed = _parse_hf_spec(args.safetensors)
        if parsed is not None:
            repo, step = parsed
            if repo:
                cfg.hf.repo_id = repo
            if step is not None:
                cfg.train.resume_step = step
        else:
            cfg.train.resume_ckpt = args.safetensors
    if getattr(args, "step", None) is not None:
        cfg.train.resume_step = args.step
    if getattr(args, "ckpt", None) is not None:
        cfg.train.resume_ckpt = args.ckpt

    # Surcharges CLI
    if getattr(args, "hot_fix", False):
        cfg.train.hot_fix = True
        logger.info(">>> HOT-FIX MODE ACTIVE: f, g, Adam and Replay Buffer will be reset, h(s) will be kept with progressive unfreezing <<<")
    if getattr(args, "reset_policy", False):
        cfg.train.reset_policy_head = True
    if getattr(args, "reset_value_head", False):
        cfg.train.reset_value_head = True
        logger.info(">>> SURGICAL RESET: only v_dense and rdet_fc2 are reset "
                    "(h, policy and transition g preserved) <<<")
        if not getattr(args, "fresh_buffer", False):
            logger.warning(
                "--reset-value-head without --fresh-buffer: existing buffer contains "
                "target_pol from contaminated MCTS and hallucinated opponent hand obs. "
                "Add --fresh-buffer unless specifically intended."
            )
    if getattr(args, "fresh_buffer", False):
        cfg.train.fresh_buffer = True
    if getattr(args, "freeze_h_steps", None) is not None:
        cfg.train.freeze_representation_steps = args.freeze_h_steps
    if getattr(args, "unfreeze_ramp_steps", None) is not None:
        cfg.train.unfreeze_ramp_steps = args.unfreeze_ramp_steps
    if getattr(args, "no_reward_shaping", False):
        cfg.train.enable_reward_shaping = False
        logger.info("Reward shaping désactivé (mode sparse ±1.0)")
    if getattr(args, "workers", None) is not None:
        cfg.train.num_workers = int(args.workers)
        logger.info("Workers self-play surchargés par CLI : %d", cfg.train.num_workers)
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
    if cfg.hf.enabled:
        from export.hub import get_hf_token
        get_hf_token(cfg, required=True)
    _check_devices(cfg)

    # Sauvegarde config pour reproductibilité
    Path(cfg.infra.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(Path(cfg.infra.checkpoint_dir) / "config.json")

    # Import du tracker d'activité explicite et propre
    import threading
    import time
    from training.activity import tracker, dump_all_stacks

    # Initialisation propre des temps du tracker pour l'estimation de l'ETA
    tracker.start_time = time.time()
    tracker.last_activity_time = time.time()
    tracker.games_completed = 0

    # Thread heartbeat basé sur le temps réel (toutes les 15 secondes)
    def _heartbeat():
        def format_time(seconds: float) -> str:
            if seconds <= 0:
                return "0s"
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            if h > 0:
                return f"{h}h {m}m {s}s"
            return f"{m}m {s}s"

        while True:
            time.sleep(15.0)
            now = time.time()
            inactive_dur = now - tracker.last_activity_time
            
            target_size = cfg.train.min_replay_size if tracker.buffer_size < cfg.train.min_replay_size else cfg.train.replay_buffer_size
            
            elapsed = now - tracker.start_time
            # Le temps par partie (durée moyenne d'une session de self-play de 8 parties parallèles)
            sec_per_game = tracker.avg_self_play_time
            
            current_size = tracker.buffer_size
            next_milestone = ((current_size // 10000) + 1) * 10000
            milestone_k = next_milestone // 1000
            
            is_seeding = current_size < cfg.train.min_replay_size
            avg_trans_game = tracker.avg_transitions_per_game
            avg_sp_time = tracker.avg_self_play_time
            avg_step_time = tracker.avg_train_step_time
            
            if is_seeding:
                games_per_cycle = 8
                transitions_per_cycle = games_per_cycle * avg_trans_game
                time_per_cycle = avg_sp_time
            else:
                games_per_cycle = cfg.train.games_per_self_play
                transitions_per_cycle = games_per_cycle * avg_trans_game
                time_per_cycle = avg_sp_time + cfg.train.self_play_interval * avg_step_time

            rate = transitions_per_cycle / time_per_cycle if time_per_cycle > 0 else 1.0
            remaining = next_milestone - current_size
            eta_s = remaining / rate if rate > 0 else 0.0
            eta_str = format_time(eta_s)

            step_str = f"Step: {tracker.current_step} (nouveau: {tracker.new_step}) | " if tracker.current_step > 0 else ""
            act = tracker.get_action_percentages()
            act_str = f"Actions: ATK={act.get('attack',0):.0f}% ATT={act.get('attach',0):.0f}% END={act.get('end',0):.0f}% | " if tracker.total_actions_tracked > 0 else ""
            pol_str = f"H_norm: {tracker.policy_entropy_norm:.2f} (p_max: {tracker.policy_p_max:.2f}) | " if tracker.policy_entropy_norm > 0 else ""
            h_str = f"h_scale: {tracker.h_grad_scale:.2f} | " if tracker.h_grad_scale < 1.0 else ""

            # Détection de freeze (aucune mise à jour d'activité depuis plus de 240s)
            if inactive_dur > 240.0:
                freeze_warning = " ⚠️ ATTENTION : Activité suspecte, possible freeze !"
                logger.warning(
                    "[heartbeat] %s%s%s%sPhase: %s | Buffer: %d/%d | Erreurs Deck: %d | Étape jeu: %d | Sec/partie: %.1fs | ETA (%dk): %s | Inactif depuis: %.1fs%s",
                    step_str, pol_str, act_str, h_str, tracker.phase, tracker.buffer_size, target_size, tracker.deck_errors, tracker.current_game_steps, sec_per_game, milestone_k, eta_str, inactive_dur, freeze_warning
                )
                try:
                    dump_all_stacks()
                except Exception as e:
                    logger.error("Impossible de dumper la stack trace : %s", e)
            else:
                logger.info(
                    "[heartbeat] %s%s%s%sPhase: %s | Buffer: %d/%d | Erreurs Deck: %d | Étape jeu: %d | Sec/partie: %.1fs | ETA (%dk): %s | Inactif depuis: %.1fs",
                    step_str, pol_str, act_str, h_str, tracker.phase, tracker.buffer_size, target_size, tracker.deck_errors, tracker.current_game_steps, sec_per_game, milestone_k, eta_str, inactive_dur
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
    if cfg.hf.enabled:
        from export.hub import get_hf_token
        get_hf_token(cfg, required=True)
    if args.eval_games:
        cfg.train.eval_games = args.eval_games

    # Charge checkpoint
    params, deck_params = _load_checkpoint(args.ckpt, cfg)

    from cards.encoder import CardStaticFeatures
    from models.networks import MuZeroNetwork
    from models.deck_builder import DeckBuilderNetwork, sample_deck, set_basic_pokemon_ids, set_energy_ids, set_ace_spec_ids
    from training.trainer import make_agent_fn
    from env.wrapper import run_self_play_game

    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    # AUDIT §3.10 — même définition qu'à l'entraînement (trainer.py) : seules les
    # Basic Energy sont illimitées.  L'ancien filtre `"Energy" in stage` incluait
    # aussi les Special Energy, produisant deux jeux d'IDs différents entre
    # entraînement et évaluation.
    energy_ids = [
        cid for cid in card_data.card_ids
        if card_data._cards[cid].get("stage", "").strip().lower() == "basic energy"
    ]
    set_energy_ids(energy_ids)
    set_basic_pokemon_ids([
        cid for cid in card_data.card_ids
        if card_data._cards[cid].get("stage", "").strip().lower()
        in ("basic pokémon", "basic pokemon")
    ])
    set_ace_spec_ids(card_data.ace_spec_ids)

    network  = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    deck_net = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)

    rng = jax.random.PRNGKey(cfg.infra.seed)
    wins = 0
    total = cfg.train.eval_games

    # Initialisation robuste avec fallback si aucun checkpoint n'est fourni ou valide
    if not params or not deck_params:
        logger.info("Initialisation de paramètres aléatoires pour l'évaluation...")
        from training.trainer import _make_dummy_obs
        from interpretability.probes import ProbeHeads

        dummy_obs = _make_dummy_obs(cfg.model)
        batch_obs = {k: jnp.array(v[None]) for k, v in dummy_obs.items()}
        rng, rng_mz, rng_pr, rng_dk = jax.random.split(rng, 4)

        if not params:
            probe_heads = ProbeHeads(cfg=cfg.model)
            mz_params = network.init(rng_mz, batch_obs, method=network.init_all)
            z_dummy = jnp.zeros((1, cfg.model.latent_dim))
            pr_params = probe_heads.init(rng_pr, z_dummy)
            params = {"muzero": mz_params, "probes": pr_params}

    from models.deck_builder import DEFAULT_COMPETITIVE_DECK

    for g in range(total):
        rng, rng_ag = jax.random.split(rng)
        deck0 = list(DEFAULT_COMPETITIVE_DECK)
        deck1 = list(DEFAULT_COMPETITIVE_DECK)

        agent = make_agent_fn(network, params, cfg, rng_ag, train_mode=False)

        def random_agent(obs_dict, player_idx, _cfg):
            opts = (obs_dict.get("select") or {}).get("option", [])
            n = max(len(opts), 1)
            return [np.random.randint(0, n)], np.ones(cfg.model.max_actions) / cfg.model.max_actions, 0.0

        h0, h1 = run_self_play_game(
            (agent, random_agent), deck0, deck1, cfg, np.random.default_rng()
        )
        if h0.game_won:
            wins += 1
        logger.info("Game %d/%d — won=%s", g + 1, total, h0.game_won)

    wr = wins / total
    logger.info("=== Eval result : %d/%d  win_rate=%.2f ===", wins, total, wr)


# ─────────────────────────────────────────────────────────────────────────────
# Mode : test — force du modèle contre un étalon extérieur
# ─────────────────────────────────────────────────────────────────────────────
def _parse_hf_spec(spec):
    """Reconnaît une référence HuggingFace passée à ``-s`` / ``--opponent-weights``.

    Formes acceptées (insensible à la casse) ::

        HF                       → dernier checkpoint de cfg.hf.repo_id
        hf@120000                → étape précise de cfg.hf.repo_id
        hf:owner/repo            → dernier checkpoint d'un autre dépôt
        hf:owner/repo@120000     → étape précise d'un autre dépôt

    Retourne ``(repo_id | None, step | None)`` ou ``None`` si ce n'est pas une
    référence HF.
    """
    if spec is None:
        return None
    s = str(spec).strip()
    low = s.lower()
    if low != "hf" and not (low.startswith("hf:") or low.startswith("hf@")):
        return None
    body = s[2:]
    if body.startswith(":"):
        body = body[1:]
    step = None
    if "@" in body:
        body, _, raw_step = body.rpartition("@")
        raw_step = raw_step.strip()
        if raw_step.isdigit():
            step = int(raw_step)
        elif raw_step:
            raise ValueError(f"Invalid HF step in -s {spec!r}: {raw_step!r} is not an integer.")
    body = body.strip()
    return (body or None, step)


_HF_CACHE: dict = {}


def _fetch_hf_weights(spec, cfg):
    """Télécharge un checkpoint depuis HuggingFace Hub.

    Retourne ``(params_bruts, cfg_du_checkpoint, step)``.  Le résultat est mis en
    cache : tester le modèle courant contre une étape antérieure du même dépôt
    ne déclenche qu'un téléchargement par référence.
    """
    parsed = _parse_hf_spec(spec)
    if parsed is None:
        return None
    repo, step = parsed
    repo = repo or cfg.hf.repo_id
    if not repo:
        raise ValueError("No HF repository: configure cfg.hf.repo_id or use -s hf:owner/repo")

    key = (repo, step)
    if key in _HF_CACHE:
        return _HF_CACHE[key]

    from export.hub import get_hf_token, load_from_hub

    token = get_hf_token(cfg, required=False)
    if not token:
        raise RuntimeError(
            f"HuggingFace token not found — unable to read {repo} (private repo).\n"
            f"  export {cfg.hf.token_env_var}=hf_xxx   or   Kaggle secret '{cfg.hf.token_env_var}'."
        )

    logger.info("[hf] Fetching %s (%s)...", repo,
                f"step {step}" if step is not None else "latest checkpoint via latest.json")
    mz, _dk, ckpt_cfg, step_val = load_from_hub(repo, step=step, token=token, cfg=cfg)
    if mz is None:
        raise RuntimeError(
            f"No checkpoint retrieved from {repo}. Check repository name, "
            "token, and presence of latest.json / muzero.safetensors."
        )
    logger.info("[hf] Loaded checkpoint: %s @ step %s", repo, step_val)
    _HF_CACHE[key] = (mz, ckpt_cfg, step_val)
    return _HF_CACHE[key]


def _reconcile_model_cfg(cfg, ckpt_cfg) -> None:
    """Aligne cfg.model sur celui du checkpoint.

    Sans ça, un checkpoint entraîné avec d'autres dimensions serait chargé sur un
    réseau construit aux dimensions locales : ``_merge_params`` écarterait
    silencieusement toutes les couches incompatibles et on testerait un modèle
    quasi aléatoire en croyant tester le modèle entraîné.
    """
    if ckpt_cfg is None or not hasattr(ckpt_cfg, "model"):
        return
    import dataclasses
    changed = []
    for f in dataclasses.fields(cfg.model):
        local, remote = getattr(cfg.model, f.name), getattr(ckpt_cfg.model, f.name, None)
        if remote is not None and local != remote:
            changed.append(f"{f.name}: {local} → {remote}")
            setattr(cfg.model, f.name, remote)
    if changed:
        logger.warning(
            "[hf] Local architecture differs from checkpoint — aligning with checkpoint: %s",
            "; ".join(changed),
        )


def _load_muzero_weights(path, network, cfg, rng):
    """Charge des poids depuis HF, .safetensors ou .pkl, fusionnés sur une init fraîche."""
    import jax
    import jax.numpy as jnp
    from training.trainer import _merge_params, _make_dummy_obs

    dummy = _make_dummy_obs(cfg.model)
    batch = {k: jnp.array(v[None]) for k, v in dummy.items()}
    fresh = network.init(rng, batch, method=network.init_all)
    if path is None:
        logger.warning("No weights provided — tested model is RANDOM.")
        return fresh

    # ── Référence HuggingFace ────────────────────────────────────────────────
    hf = _fetch_hf_weights(path, cfg)
    if hf is not None:
        raw, _ckpt_cfg, _step = hf
        raw = raw.get("muzero", raw) if isinstance(raw, dict) else raw
        merged = _merge_params(fresh, jax.tree_util.tree_map(jax.device_put, raw))
        _warn_if_mostly_fresh(merged, fresh)
        return merged

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Weights not found: {p}")
    if p.suffix == ".pkl":
        import pickle
        with open(p, "rb") as f:
            raw = pickle.load(f).get("params", {})
    else:
        from safetensors.numpy import load_file
        from export.hub import _unflatten_params
        raw = _unflatten_params(load_file(str(p)))
    raw = raw.get("muzero", raw) if isinstance(raw, dict) else raw
    merged = _merge_params(fresh, jax.tree_util.tree_map(jax.device_put, raw))
    logger.info("Loaded weights: %s", p)
    _warn_if_mostly_fresh(merged, fresh)
    return merged


def _warn_if_mostly_fresh(merged, fresh) -> None:
    """Alerte si la fusion a conservé surtout des poids frais.

    Un checkpoint dont l'architecture ne correspond pas se charge « avec succès »
    mais laisse la quasi-totalité des couches à leur initialisation aléatoire.
    Sans ce contrôle on teste un modèle vierge en croyant tester le modèle
    entraîné — et on conclut que l'entraînement ne sert à rien.
    """
    import jax
    import jax.numpy as jnp

    m = jax.tree_util.tree_leaves(merged)
    f = jax.tree_util.tree_leaves(fresh)
    if not m or len(m) != len(f):
        return
    same = sum(1 for a, b in zip(m, f)
               if a.shape == b.shape and bool(jnp.array_equal(a, b)))
    pct = 100.0 * same / len(m)
    if pct > 50.0:
        logger.error(
            "⚠ %.0f %% of layers (%d/%d) remained at RANDOM initialization: "
            "checkpoint does not match current architecture. Results "
            "do NOT reflect trained model.", pct, same, len(m),
        )
    elif pct > 10.0:
        logger.warning(
            "%.0f %% of layers (%d/%d) were not loaded from checkpoint "
            "(layers removed or shape mismatch).", pct, same, len(m),
        )


def cmd_test(args) -> None:
    """Fait jouer le modèle contre un adversaire de référence et rapporte sa force.

    C'est le signal absent des courbes d'entraînement : `p_max`, l'entropie et la
    distribution d'actions décrivent COMMENT l'agent joue, jamais s'il joue MIEUX.
    """
    import json
    import jax
    import jax.numpy as jnp
    import numpy as np

    cfg = load_config(args)
    uses_hf = (_parse_hf_spec(args.safetensors) is not None
               or _parse_hf_spec(args.opponent_weights) is not None)
    if not uses_hf:
        cfg.hf.enabled = False
    if getattr(args, "policy_only", False):
        cfg.search.num_simulations = 0
    elif args.sims is not None:
        cfg.search.num_simulations = args.sims
    if args.belief is not None:
        cfg.search.num_belief_samples = args.belief

    from cards.encoder import CardStaticFeatures
    from models.networks import MuZeroNetwork
    from training.trainer import make_agent_fn
    from env.wrapper import run_self_play_game, DeckError
    from env.baselines import make_baseline_agent
    from models.deck_builder import (
        DEFAULT_COMPETITIVE_DECK, set_ace_spec_ids, set_basic_pokemon_ids,
        set_card_names, set_energy_ids,
    )
    from evaluation import analyse_history, aggregate, format_report

    # ── Poids HF : télécharger AVANT de construire le réseau ─────────────────
    # L'architecture doit être alignée sur le checkpoint, sinon le réseau est
    # construit aux dimensions locales et toutes les couches incompatibles sont
    # écartées en silence par _merge_params.
    weights_path = args.safetensors or args.ckpt
    for spec in (weights_path, args.opponent_weights):
        hf = _fetch_hf_weights(spec, cfg)
        if hf is not None:
            _reconcile_model_cfg(cfg, hf[1])

    # ── Cartes ───────────────────────────────────────────────────────────────
    card_data = CardStaticFeatures(cfg.infra.card_csv)
    n_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = n_ids
    static = jnp.array(card_data.feature_matrix(n_ids))
    set_energy_ids([c for c in card_data.card_ids
                    if card_data._cards[c].get("stage", "").strip().lower() == "basic energy"])
    set_basic_pokemon_ids([c for c in card_data.card_ids
                           if card_data._cards[c].get("stage", "").strip().lower()
                           in ("basic pokémon", "basic pokemon")])
    set_card_names({c: card_data.card_name(c) for c in card_data.card_ids})
    set_ace_spec_ids(card_data.ace_spec_ids)

    # ── Deck ─────────────────────────────────────────────────────────────────
    if args.deck and Path(args.deck).exists():
        deck = [int(x) for x in Path(args.deck).read_text().split() if x.strip().isdigit()][:60]
        logger.info("Deck loaded from %s (%d cards)", args.deck, len(deck))
    else:
        deck = list(DEFAULT_COMPETITIVE_DECK)
    if len(deck) != 60:
        raise ValueError(f"Deck must contain exactly 60 cards (received {len(deck)}).")

    # La déterminisation ISMCTS doit puiser dans le deck réellement joué.
    from search.ismcts import set_belief_deck
    set_belief_deck(deck)

    network = MuZeroNetwork(cfg=cfg.model, static_features=static)
    rng = jax.random.PRNGKey(cfg.infra.seed)
    rng, r_w, r_o, r_a = jax.random.split(rng, 4)

    use_iree = bool(getattr(args, "iree", False) or getattr(args, "vmfb", None) or (weights_path and str(weights_path).endswith(".vmfb")))
    device_uri = getattr(args, "device", "vulkan") or "vulkan"
    my_step = None

    if use_iree:
        from models.iree_agent import IREEMuZeroAgent
        from env.encoding import encode_observation

        vmfb_path = Path(args.vmfb) if args.vmfb else (Path(weights_path) if weights_path and str(weights_path).endswith(".vmfb") else Path(f"muzero_{device_uri}.vmfb"))
        if not vmfb_path.exists():
            for cand in [Path.cwd() / vmfb_path.name, Path(__file__).resolve().parent.parent / vmfb_path.name]:
                if cand.exists():
                    vmfb_path = cand
                    break

        if not vmfb_path.exists():
            logger.info("IREE VMFB bytecode not found (%s). Compiling online from HuggingFace Hub...", vmfb_path.name)
            import subprocess, sys
            export_script = Path(__file__).resolve().parent.parent / "export_iree.py"
            if not export_script.exists():
                export_script = Path("export_iree.py")
            spec = str(weights_path) if weights_path and not str(weights_path).endswith(".vmfb") else "HF"
            cmd = [
                sys.executable, str(export_script),
                "-m", spec,
                "-o", str(vmfb_path),
                "--target", device_uri,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error("IREE compilation failed:\n%s", res.stderr)
                raise RuntimeError(f"Failed to compile IREE module: {res.stderr}")
            logger.info("✓ IREE module compiled successfully to %s", vmfb_path)

        my_iree_agent = IREEMuZeroAgent(vmfb_path=vmfb_path, device_uri=device_uri, cfg=cfg)

        def make_iree_test_agent_fn(agent, _cfg):
            def agent_fn(obs_dict, player_idx, __cfg):
                select = obs_dict.get("select") or {}
                options = select.get("option", [])
                if not options:
                    return [0], np.zeros(_cfg.model.max_actions, dtype=np.float32), 0.0
                probs, val, _ = agent.evaluate(obs_dict, player_idx)
                max_count = int(select.get("maxCount", 1))
                enc_obs = encode_observation(obs_dict, player_idx, _cfg.model)
                option_mask = enc_obs["option_mask"]
                masked = np.where(option_mask, probs, -1e9)
                if max_count > 1:
                    sel_indices = np.argsort(masked)[::-1][:max_count].tolist()
                else:
                    sel_indices = [int(np.argmax(masked))]
                return sel_indices, probs, float(val)
            return agent_fn

        my_sims = 0
        my_title = f"IREE ({device_uri.upper()})"
    else:
        hf_my = _fetch_hf_weights(weights_path, cfg) if weights_path else None
        if hf_my is not None:
            my_step = hf_my[2]
        mz = _load_muzero_weights(weights_path, network, cfg, r_w)
        params = {"muzero": mz}
        my_sims = cfg.search.num_simulations
        my_title = f"step {my_step}" if my_step is not None else "Model"

    # ── Adversaire ───────────────────────────────────────────────────────────
    opp_sims = 0
    if args.opponent in ("greedy", "random"):
        opp_agent = make_baseline_agent(args.opponent, cfg, epsilon=args.epsilon,
                                        seed=cfg.infra.seed)
        opp_label = (f"greedy(ε={args.epsilon:.2f})" if args.opponent == "greedy" else "random")
    elif args.opponent == "self":
        if use_iree:
            opp_agent = make_iree_test_agent_fn(my_iree_agent, cfg)
            opp_label = f"self (IREE {device_uri.upper()})"
            opp_sims = 0
        else:
            opp_agent = make_agent_fn(network, params, cfg, r_o, train_mode=False)
            opp_label = "self (mirror)"
            opp_sims = cfg.search.num_simulations
    else:
        if not args.opponent_weights:
            raise ValueError("--opponent=checkpoint requires --opponent-weights.")
        opp_step = None
        if str(args.opponent_weights).endswith(".vmfb"):
            from models.iree_agent import IREEMuZeroAgent
            opp_iree_agent = IREEMuZeroAgent(vmfb_path=args.opponent_weights, device_uri=device_uri, cfg=cfg)
            opp_agent = make_iree_test_agent_fn(opp_iree_agent, cfg)
            opp_label = f"checkpoint(IREE {Path(args.opponent_weights).name})"
            opp_sims = 0
        else:
            hf_opp = _fetch_hf_weights(args.opponent_weights, cfg)
            if hf_opp is not None:
                opp_step = hf_opp[2]
            opp_mz = _load_muzero_weights(args.opponent_weights, network, cfg, r_o)
            
            # Si le modèle testé est en Policy-Only (ex: IREE à 0-sim) et que l'utilisateur n'a pas fixé --sims,
            # on aligne l'adversaire checkpoint en Policy-Only (0-sim) pour une comparaison équitable.
            if use_iree and args.sims is None:
                logger.info(
                    "[test] Modèle testé en Policy-Only (IREE 0-sim). Alignement de l'adversaire checkpoint "
                    "en Policy-Only (0-sim) pour une comparaison équitable. (Passez --sims N pour forcer MCTS)."
                )
                from copy import deepcopy
                opp_cfg = deepcopy(cfg)
                opp_cfg.search.num_simulations = 0
                opp_agent = make_agent_fn(network, {"muzero": opp_mz}, opp_cfg, r_o, train_mode=False)
                opp_sims = 0
            else:
                opp_agent = make_agent_fn(network, {"muzero": opp_mz}, cfg, r_o, train_mode=False)
                opp_sims = cfg.search.num_simulations

            if opp_step is not None:
                opp_label = f"checkpoint(HF@{opp_step})"
            else:
                opp_label = f"checkpoint({Path(args.opponent_weights).name})"

    if my_sims != opp_sims:
        logger.warning(
            "⚠️ ASYMÉTRIE DÉTECTÉE : %s joue avec %d sims MCTS vs %s avec %d sims MCTS.",
            my_title, my_sims, opp_label, opp_sims
        )

    logger.info(
        "=== TEST : %d games | model=%s [sims=%d] | opponent=%s [sims=%d] | belief=%d | swap=%s ===",
        args.games, my_title, my_sims, opp_label, opp_sims,
        cfg.search.num_belief_samples if (my_sims > 0 or opp_sims > 0) else 0, not args.no_swap,
    )

    # ── Parties ──────────────────────────────────────────────────────────────
    stats, errors = [], 0
    rng_test = jax.random.PRNGKey(cfg.infra.seed if not use_iree else 42)
    for g in range(args.games):
        if use_iree:
            me = make_iree_test_agent_fn(my_iree_agent, cfg)
        else:
            rng_test, r_g = jax.random.split(rng_test)
            me = make_agent_fn(network, params, cfg, r_g, train_mode=False)
        # Alterner les côtés : sans ça, tout l'écart mesuré peut n'être que
        # l'avantage (ou le désavantage) structurel du premier joueur.
        my_idx = 0 if (args.no_swap or g % 2 == 0) else 1
        pair = (me, opp_agent) if my_idx == 0 else (opp_agent, me)
        try:
            h0, h1 = run_self_play_game(pair, deck, deck, cfg, np.random.default_rng(g))
        except DeckError as e:
            logger.error("Deck rejected by engine: %s", e)
            raise
        except Exception as e:
            errors += 1
            logger.warning("Game %d aborted: %s", g + 1, e)
            continue
        mine = h0 if my_idx == 0 else h1
        st = analyse_history(mine)
        stats.append(st)
        logger.info("  game %d/%d — side %d — %s (%d decisions, %.0f prizes)",
                    g + 1, args.games, my_idx,
                    {True: "WIN", False: "loss"}.get(st.won, "draw"),
                    st.decisions, st.prizes_taken)

    agg = aggregate(stats)
    agg["opponent"] = opp_label
    agg["weights"] = str(weights_path)
    agg["num_simulations"] = cfg.search.num_simulations
    agg["aborted_games"] = errors

    print()
    print(format_report(f"{my_title} vs {opp_label}", agg))
    print()
    if agg.get("games"):
        wr = agg["win_rate_pct"]
        lo, hi = agg["win_rate_ci95_pct"]
        if not agg["conclusive"]:
            # Ne jamais conclure sur un intervalle qui contient 50 % : c'est
            # exactement ainsi qu'on prend du bruit pour un résultat.
            print(f"  → NON CONCLUSIF : {wr:.0f} % mais IC95 [{lo:.0f} %, {hi:.0f} %] "
                  f"contient 50 %. Relancez avec -n plus grand avant d'interpréter.")
        elif args.opponent == "random" and hi < 80:
            print("  ⚠ Significativement sous 80 % contre l'agent ALÉATOIRE : "
                  "le modèle ne joue pas.")
        elif lo > 50:
            print(f"  → Bat {opp_label} de façon significative. Monter --epsilon "
                  "jusqu'à revenir vers 50 % pour situer sa force.")
        else:
            print(f"  → Perd significativement contre {opp_label}. "
                  "Baisser --epsilon pour trouver le palier atteint.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(agg, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
        logger.info("Rapport JSON écrit : %s", args.json_out)


# ─────────────────────────────────────────────────────────────────────────────
# Mode : submit (génère le dossier de soumission Kaggle autonome)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_submit(args) -> None:
    """
    Génère un dossier de soumission autonome pour Kaggle (contenant main.py, deck.csv,
    les poids safetensors téléchargés depuis HF et les dépendances du projet).
    """
    import shutil
    import json
    from pathlib import Path

    cfg = load_config(args)
    out_dir = Path(getattr(args, "out_dir", None) or "submission")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing submission folder → %s", out_dir.resolve())

    # 1. Téléchargement ou récupération des poids safetensors (HF Hub ou local)
    step_num = getattr(args, "step", None)
    mz_params, dk_params = None, None
    loaded_cfg = cfg

    from config import Config
    from export.hub import load_from_hub, get_hf_token, _unflatten_params, _flatten_params
    from safetensors.numpy import load_file

    download_success = False
    if cfg.hf.enabled:
        token = get_hf_token(cfg, required=True)
        logger.info("Attempting to load from HuggingFace Hub (%s)...", cfg.hf.repo_id)

        mz_params, dk_params, loaded_cfg, step_val = load_from_hub(
            repo_id=cfg.hf.repo_id,
            step=step_num,
            token=token,
            cfg=cfg,
        )
        if mz_params is not None:
            download_success = True
            logger.info("✓ HF checkpoint loaded successfully (step=%s)", step_val)

    # Si le téléchargement HF a échoué ou HF désactivé/non authentifié, chercher les poids locaux
    if not download_success:
        logger.info("Searching for local checkpoints and weights (hf_checkpoints, checkpoints, submission, root)...")

        # Option A : Fichiers safetensors déjà présents dans out_dir (submission/)
        mz_out = out_dir / "muzero.safetensors"
        dk_out = out_dir / "deck_builder.safetensors"
        cfg_out = out_dir / "config.json"
        if mz_out.exists() and dk_out.exists():
            try:
                mz_params = _unflatten_params(load_file(str(mz_out)))
                dk_params = _unflatten_params(load_file(str(dk_out)))
                if cfg_out.exists():
                    loaded_cfg = Config.from_json(cfg_out.read_text())
                download_success = True
                logger.info("✓ Local safetensors found in submission folder (%s)", out_dir)
            except Exception as e:
                logger.warning("Error reading safetensors in %s: %s", out_dir, e)

        # Option B : hf_checkpoints (step_*)
        if not download_success:
            local_dir = Path(cfg.hf.local_dir)
            if local_dir.exists():
                step_dirs = sorted([d for d in local_dir.glob("step_*") if d.is_dir()])
                if step_dirs:
                    target_dir = step_dirs[-1]
                    logger.info("Using local hf_checkpoints checkpoint → %s", target_dir)
                    try:
                        mz_path = target_dir / "muzero.safetensors"
                        dk_path = target_dir / "deck_builder.safetensors"
                        cfg_path = target_dir / "config.json"
                        if mz_path.exists():
                            mz_params = _unflatten_params(load_file(str(mz_path)))
                            shutil.copy2(mz_path, out_dir / "muzero.safetensors")
                        if dk_path.exists():
                            dk_params = _unflatten_params(load_file(str(dk_path)))
                            shutil.copy2(dk_path, out_dir / "deck_builder.safetensors")
                        if cfg_path.exists():
                            loaded_cfg = Config.from_json(cfg_path.read_text())
                            shutil.copy2(cfg_path, out_dir / "config.json")
                        download_success = True
                    except Exception as exc:
                        logger.error("Error reading local safetensors: %s", exc)

        # Option C : pickle checkpoint (ckpt_latest.pkl) dans checkpoint_dir
        if not download_success:
            ckpt_dir = Path(cfg.infra.checkpoint_dir)
            latest_pkl = ckpt_dir / "ckpt_latest.pkl"
            if not latest_pkl.exists() and ckpt_dir.exists():
                pkls = sorted(list(ckpt_dir.glob("ckpt_*.pkl")))
                if pkls:
                    latest_pkl = pkls[-1]

            if latest_pkl.exists():
                try:
                    import pickle
                    with open(latest_pkl, "rb") as f:
                        ckpt_data = pickle.load(f)
                    mz_params = ckpt_data.get("params", {})
                    dk_params = ckpt_data.get("deck", {})
                    download_success = True
                    logger.info("✓ Weights loaded from local pickle checkpoint (%s, step=%d)", latest_pkl, ckpt_data.get("step", 0))

                    if mz_params:
                        from safetensors.numpy import save_file
                        save_file(_flatten_params(mz_params), str(out_dir / "muzero.safetensors"))
                    if dk_params:
                        from safetensors.numpy import save_file
                        save_file(_flatten_params(dk_params, prefix="deck"), str(out_dir / "deck_builder.safetensors"))
                    (out_dir / "config.json").write_text(loaded_cfg.to_json())
                except Exception as e:
                    logger.warning("Failed to read pkl %s: %s", latest_pkl, e)

        # Option D : safetensors situés dans le dossier racine ou parent
        if not download_success:
            for root_path in [Path("."), Path(__file__).resolve().parent.parent]:
                mz_root = root_path / "muzero.safetensors"
                dk_root = root_path / "deck_builder.safetensors"
                if mz_root.exists() or dk_root.exists():
                    try:
                        if mz_root.exists():
                            mz_params = _unflatten_params(load_file(str(mz_root)))
                            shutil.copy2(mz_root, out_dir / "muzero.safetensors")
                        if dk_root.exists():
                            dk_params = _unflatten_params(load_file(str(dk_root)))
                            shutil.copy2(dk_root, out_dir / "deck_builder.safetensors")
                        download_success = True
                        logger.info("✓ Root safetensors (%s) copied to %s", root_path, out_dir)
                        break
                    except Exception as e:
                        logger.warning("Error reading root safetensors %s: %s", root_path, e)

    # Si téléchargé depuis HF avec succès, sauvegarder dans out_dir
    if download_success and mz_params is not None:
        try:
            from safetensors.numpy import save_file
            save_file(_flatten_params(mz_params), str(out_dir / "muzero.safetensors"))
            if dk_params:
                clean_dk = dk_params.get("deck", dk_params) if (isinstance(dk_params, dict) and "deck" in dk_params and "params" not in dk_params) else dk_params
                save_file(_flatten_params(clean_dk, prefix="deck"), str(out_dir / "deck_builder.safetensors"))
            (out_dir / "config.json").write_text(loaded_cfg.to_json())
        except Exception as exc:
            logger.warning("Warning during local safetensors save in %s: %s", out_dir, exc)

    # 2. Génération de deck.csv
    base_src = Path(__file__).resolve().parent
    card_csv_candidates = [
        "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN Card Data.csv",
        "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv",
        "/kaggle/input/cards.csv",
        str(base_src.parent / "competiton" / "EN Card Data.csv"),
        str(base_src.parent / "competiton" / "EN_Card_Data.csv"),
        str(base_src.parent / "competition" / "EN Card Data.csv"),
        str(base_src.parent / "competition" / "EN_Card_Data.csv"),
        str(base_src / "EN Card Data.csv"),
        str(base_src / "EN_Card_Data.csv"),
        "competiton/EN Card Data.csv",
        "competiton/EN_Card_Data.csv",
        "EN Card Data.csv",
        "EN_Card_Data.csv",
    ]
    card_csv_path = None
    for c in card_csv_candidates:
        if os.path.exists(c):
            card_csv_path = c
            break

    from models.deck_builder import DEFAULT_COMPETITIVE_DECK
    deck_csv_file = out_dir / "deck.csv"
    with open(deck_csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(str(cid) for cid in DEFAULT_COMPETITIVE_DECK) + "\n")
    logger.info("✓ deck.csv exported successfully (reference Ogerpon ex deck, %d cards)", len(DEFAULT_COMPETITIVE_DECK))

    # 3. Copie des modules Python du projet vers out_dir/
    base_src = Path(__file__).resolve().parent
    subdirs_to_copy = ["cards", "env", "models", "search", "export"]
    for s in subdirs_to_copy:
        src_path = base_src / s
        dst_path = out_dir / s
        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    # Copie de config.py
    shutil.copy2(base_src / "config.py", out_dir / "config.py")

    # Copie du dossier cg/ (fourni par Kaggle, requis par cg.api)
    cg_src_candidates = [
        base_src.parent / "competiton" / "sample_submission" / "sample_submission" / "cg",
        base_src.parent / "competition" / "sample_submission" / "sample_submission" / "cg",
        base_src.parent / "cg",
    ]
    cg_src = next((p for p in cg_src_candidates if p.exists()), None)
    if cg_src:
        cg_dst = out_dir / "cg"
        if cg_dst.exists():
            shutil.rmtree(cg_dst)
        shutil.copytree(cg_src, cg_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        logger.info("✓ cg/ folder copied to %s", out_dir)
    else:
        logger.warning("cg/ folder not found — please copy manually to %s", out_dir)

    logger.info("✓ Project source code copied to %s", out_dir)

    # Vendor mctx (non installé sur Kaggle)
    mctx_src_candidates = [
        base_src / "vendor" / "mctx",
        base_src.parent / "submission" / "mctx",
    ]
    mctx_src = next((p for p in mctx_src_candidates if p.is_dir()), None)
    if mctx_src:
        # Gère vendor/mctx et vendor/mctx/mctx (structure imbriquée)
        if (mctx_src / "mctx" / "__init__.py").is_file() and not (mctx_src / "__init__.py").is_file():
            mctx_src = mctx_src / "mctx"
        mctx_dst = out_dir / "mctx"
        if mctx_dst.exists():
            shutil.rmtree(mctx_dst)
        shutil.copytree(mctx_src, mctx_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
        logger.info("✓ mctx/ vendored in %s", out_dir)
    else:
        logger.warning("mctx/ not found — run: pip download mctx && unzip in ptcg_muzero/vendor/mctx")

    # Vendor chex (dépendance obligatoire de mctx, non installé sur Kaggle)
    chex_src_candidates = [
        base_src / "vendor" / "chex",
        base_src.parent / "submission" / "chex",
    ]
    chex_src = next((p for p in chex_src_candidates if p.is_dir()), None)
    if chex_src:
        # Gère vendor/chex et vendor/chex/chex (structure imbriquée)
        if (chex_src / "chex" / "__init__.py").is_file() and not (chex_src / "__init__.py").is_file():
            chex_src = chex_src / "chex"
        chex_dst = out_dir / "chex"
        if chex_dst.exists():
            shutil.rmtree(chex_dst)
        shutil.copytree(chex_src, chex_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
        logger.info("✓ chex/ vendored in %s", out_dir)
    else:
        logger.warning("chex/ not found — required by mctx, run: pip download chex && unzip in ptcg_muzero/vendor/chex")

    # Création des __init__.py manquants (requis par Kaggle qui utilise exec())
    for s in subdirs_to_copy:
        init_file = out_dir / s / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            logger.info("  created %s/__init__.py", s)

    # 4. Génération de main.py dans out_dir
    submit_main = Path(__file__).resolve().parent / "submit_main.py"
    (out_dir / "main.py").write_text(submit_main.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("✓ %s/main.py successfully generated", out_dir)
    logger.info("=== Ready for Kaggle submission ===")
    logger.info("1. Create a Kaggle dataset with ALL contents of %s/", out_dir)
    logger.info("2. Attach this dataset to your submission notebook")
    logger.info("3. Upload main.py (+ deck.csv) as Kaggle agent")
    logger.info("To archive: tar -czvf submission.tar.gz -C %s .", out_dir)


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
    print("\n(Accuracy on dummy targets=0 — run on real games for actual metrics)")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────
def _check_devices(cfg: "Config") -> None:
    import jax
    n = len(jax.devices())
    if n < cfg.infra.num_devices:
        logger.warning(
            "%d GPU(s) requested but only %d available. Automatic adjustment.",
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
        logger.info("Loaded checkpoint: %s  (step=%d)", ckpt_path, data.get("step", -1))
        return (
            jax.tree_util.tree_map(jax.device_put, data["params"]),
            jax.tree_util.tree_map(jax.device_put, data.get("deck", {})),
        )

    # 1. Vérifier si un checkpoint pickle local récent existe dans checkpoint_dir
    ckpt_dir = Path(cfg.infra.checkpoint_dir)
    latest_pkl = ckpt_dir / "ckpt_latest.pkl"
    if not latest_pkl.exists() and ckpt_dir.exists():
        pkls = sorted(list(ckpt_dir.glob("ckpt_*.pkl")))
        if pkls:
            latest_pkl = pkls[-1]

    if latest_pkl.exists():
        try:
            with open(latest_pkl, "rb") as f:
                data = pickle.load(f)
            logger.info("Loaded local checkpoint: %s (step=%d)", latest_pkl, data.get("step", -1))
            return (
                jax.tree_util.tree_map(jax.device_put, data.get("params", {})),
                jax.tree_util.tree_map(jax.device_put, data.get("deck", {})),
            )
        except Exception:
            pass

    # 2. Tenter le chargement depuis HF Hub si activé et token disponible
    if cfg.hf.enabled and cfg.hf.repo_id:
        try:
            from export.hub import load_from_hub
            logger.info("Attempting to load from HuggingFace Hub (%s)...", cfg.hf.repo_id)
            mz_params, deck_params, _, hf_step = load_from_hub(cfg.hf.repo_id, cfg=cfg)
            if mz_params:
                logger.info("HuggingFace Hub checkpoint loaded successfully (%s, step=%d)", cfg.hf.repo_id, hf_step)
                return mz_params, deck_params
        except Exception as e:
            logger.warning("Failed to load from HuggingFace Hub: %s", e)

    # 3. Safetensors locaux dans local_dir ou submission/ ou .
    for safetensors_dir in [Path(cfg.hf.local_dir), Path("submission"), Path(".")]:
        mz_file = safetensors_dir / "muzero.safetensors"
        dk_file = safetensors_dir / "deck_builder.safetensors"
        if mz_file.exists():
            try:
                from safetensors.numpy import load_file
                from export.hub import _unflatten_params
                mz_p = _unflatten_params(load_file(str(mz_file)))
                dk_p = _unflatten_params(load_file(str(dk_file))) if dk_file.exists() else {}
                logger.info("Local safetensors loaded from %s", safetensors_dir)
                return (
                    jax.tree_util.tree_map(jax.device_put, mz_p),
                    jax.tree_util.tree_map(jax.device_put, dk_p),
                )
            except Exception:
                pass

    logger.warning("No checkpoint found — using random parameters.")
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
        "test":       cmd_test,
        "submit":     cmd_submit,
        "probe-diag": cmd_probe_diag,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
