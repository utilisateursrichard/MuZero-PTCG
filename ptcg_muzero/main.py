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
    p.add_argument("--out-dir",  default="submission", help="Dossier de sortie pour la soumission")
    p.add_argument("--step",     type=int, default=None, help="Numéro d'étape HF spécifique")
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
            # Détection de freeze (aucune mise à jour d'activité depuis plus de 240s)
            if inactive_dur > 240.0:
                freeze_warning = " ⚠️ ATTENTION : Activité suspecte, possible freeze !"
                logger.warning(
                    "[heartbeat] %sPhase: %s | Buffer: %d/%d | h(s): %s | Erreurs Deck: %d | Étape jeu: %d | Sec/partie: %.1fs | ETA (%dk): %s | Inactif depuis: %.1fs%s",
                    step_str, tracker.phase, tracker.buffer_size, target_size, tracker.h_status, tracker.deck_errors, tracker.current_game_steps, sec_per_game, milestone_k, eta_str, inactive_dur, freeze_warning
                )
                try:
                    dump_all_stacks()
                except Exception as e:
                    logger.error("Impossible de dumper la stack trace : %s", e)
            else:
                logger.info(
                    "[heartbeat] %sPhase: %s | Buffer: %d/%d | h(s): %s | Erreurs Deck: %d | Étape jeu: %d | Sec/partie: %.1fs | ETA (%dk): %s | Inactif depuis: %.1fs",
                    step_str, tracker.phase, tracker.buffer_size, target_size, tracker.h_status, tracker.deck_errors, tracker.current_game_steps, sec_per_game, milestone_k, eta_str, inactive_dur
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
    from models.deck_builder import DeckBuilderNetwork, sample_deck, set_basic_pokemon_ids, set_energy_ids, set_ace_spec_ids
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
            mz_params = network.init(rng_mz, batch_obs)
            z_dummy = jnp.zeros((1, cfg.model.latent_dim))
            pr_params = probe_heads.init(rng_pr, z_dummy)
            params = {"muzero": mz_params, "probes": pr_params}

        if not deck_params:
            dummy_ctx = jnp.zeros((1, cfg.model.latent_dim))
            deck_params = deck_net.init(rng_dk, context=dummy_ctx)

    for g in range(total):
        rng, rng_d, rng_ag = jax.random.split(rng, 3)
        deck_logits, _ = deck_net.apply(deck_params)
        deck0, _ = sample_deck(deck_logits[0], rng_d, num_card_ids, energy_ids)
        deck1 = deck0[:]

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
    logger.info("Préparation du dossier de soumission → %s", out_dir.resolve())

    # 1. Téléchargement ou récupération des poids safetensors (HF Hub ou local)
    step_num = getattr(args, "step", None)
    mz_params, dk_params = None, None
    loaded_cfg = cfg

    from export.hub import load_from_hub, get_hf_token
    token = get_hf_token(cfg)

    download_success = False
    if cfg.hf.enabled:
        logger.info("Tentative de chargement depuis HuggingFace Hub (%s)...", cfg.hf.repo_id)
        try:
            mz_params, dk_params, loaded_cfg, step_val = load_from_hub(
                repo_id=cfg.hf.repo_id,
                step=step_num,
                token=token,
                cfg=cfg,
            )
            download_success = True
            logger.info("✓ Checkpoint HF chargé avec succès (step=%s)", step_val)
        except Exception as exc:
            logger.warning("Impossible de charger depuis le Hub HF : %s", exc)

    # Si le téléchargement HF a échoué ou HF désactivé, chercher les checkpoints locaux
    if not download_success:
        logger.info("Recherche de checkpoints locaux dans %s...", cfg.hf.local_dir)
        local_dir = Path(cfg.hf.local_dir)
        if local_dir.exists():
            step_dirs = sorted([d for d in local_dir.glob("step_*") if d.is_dir()])
            if step_dirs:
                target_dir = step_dirs[-1]
                logger.info("Utilisation du checkpoint local le plus récent → %s", target_dir)
                try:
                    from safetensors.numpy import load_file
                    from export.hub import _unflatten_params
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
                    logger.error("Erreur lors de la lecture des safetensors locaux : %s", exc)

    # Si on a téléchargé depuis HF, sauvegarder/copier les safetensors dans out_dir
    if download_success and mz_params is not None:
        try:
            from huggingface_hub import hf_hub_download
            subfolder = f"step_{step_val:07d}" if step_val is not None else ""
            def _dl(filename):
                return hf_hub_download(
                    repo_id=cfg.hf.repo_id,
                    filename=f"{subfolder}/{filename}" if subfolder else filename,
                    token=token,
                )
            try:
                shutil.copy2(_dl("muzero.safetensors"), out_dir / "muzero.safetensors")
                shutil.copy2(_dl("deck_builder.safetensors"), out_dir / "deck_builder.safetensors")
                shutil.copy2(_dl("config.json"), out_dir / "config.json")
                logger.info("✓ Fichiers safetensors copiés depuis le Hub vers %s", out_dir)
            except Exception as e:
                from safetensors.numpy import save_file
                from export.hub import _flatten_params
                save_file(_flatten_params(mz_params), str(out_dir / "muzero.safetensors"))
                save_file(_flatten_params(dk_params, prefix="deck"), str(out_dir / "deck_builder.safetensors"))
                (out_dir / "config.json").write_text(loaded_cfg.to_json())
        except Exception as exc:
            logger.warning("Avertissement lors de la sauvegarde locale des safetensors dans %s: %s", out_dir, exc)

    # 2. Génération de deck.csv
    card_csv_candidates = [
        "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv",
        "/kaggle/input/cards.csv",
        "competiton/EN_Card_Data.csv",
        "EN_Card_Data.csv",
    ]
    card_csv_path = None
    for c in card_csv_candidates:
        if os.path.exists(c):
            card_csv_path = c
            break

    deck_generated = False
    if dk_params is not None and card_csv_path is not None:
        try:
            import jax
            import jax.numpy as jnp
            from cards.encoder import CardStaticFeatures
            from models.deck_builder import (
                DeckBuilderNetwork,
                sample_deck,
                set_ace_spec_ids,
                set_basic_pokemon_ids,
                set_energy_ids,
            )

            card_data = CardStaticFeatures(card_csv_path)
            num_card_ids = max(card_data.max_card_id + 1, loaded_cfg.model.num_card_ids)
            loaded_cfg.model.num_card_ids = num_card_ids
            static_feats = jnp.array(card_data.feature_matrix(num_card_ids))

            energy_ids = [c for c in card_data.card_ids if "Energy" in card_data._cards[c].get("stage", "")]
            set_energy_ids(energy_ids)
            set_basic_pokemon_ids([
                cid for cid in card_data.card_ids
                if card_data._cards[cid].get("stage", "").strip().lower() in ("basic pokémon", "basic pokemon")
            ])
            set_ace_spec_ids(card_data.ace_spec_ids)

            deck_net = DeckBuilderNetwork(cfg=loaded_cfg.model, static_features=static_feats)
            eval_dk_params = dk_params.get("deck", dk_params) if isinstance(dk_params, dict) and "deck" in dk_params else dk_params
            logits, _ = deck_net.apply(eval_dk_params)
            rng = jax.random.PRNGKey(42)
            sampled_deck_ids, _ = sample_deck(logits[0], rng, num_card_ids, energy_ids, temperature=0.1)

            deck_csv_file = out_dir / "deck.csv"
            with open(deck_csv_file, "w", encoding="utf-8") as f:
                f.write("\n".join(str(cid) for cid in sampled_deck_ids) + "\n")
            logger.info("✓ deck.csv généré avec succès par le DeckBuilder (%d cartes)", len(sampled_deck_ids))
            deck_generated = True
        except Exception as exc:
            logger.warning("Échec de génération du deck par DeckBuilder : %s", exc)

    if not deck_generated:
        # Fallback sur le deck de référence
        base_src = Path(__file__).resolve().parent
        ref_deck_candidates = [
            base_src.parent / "competiton" / "sample_submission" / "sample_submission" / "deck.csv",
            base_src.parent / "competiton" / "sample_submission" / "deck.csv",
            base_src.parent / "deck.csv",
        ]
        ref_deck = next((p for p in ref_deck_candidates if p.exists()), None)
        if ref_deck:
            shutil.copy2(ref_deck, out_dir / "deck.csv")
            logger.info("✓ deck.csv copié depuis le deck de référence (%s)", ref_deck)
        else:
            logger.error("Aucun deck.csv de référence trouvé!")

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
        logger.info("✓ dossier cg/ copié dans %s", out_dir)
    else:
        logger.warning("dossier cg/ introuvable — à copier manuellement dans %s", out_dir)

    logger.info("✓ Code source du projet copié dans %s", out_dir)

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
        logger.info("✓ mctx/ vendored dans %s", out_dir)
    else:
        logger.warning("mctx/ introuvable — exécutez: pip download mctx && unzip dans ptcg_muzero/vendor/mctx")

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
        logger.info("✓ chex/ vendored dans %s", out_dir)
    else:
        logger.warning("chex/ introuvable — mctx en dépend, exécutez: pip download chex && unzip dans ptcg_muzero/vendor/chex")

    # Création des __init__.py manquants (requis par Kaggle qui utilise exec())
    for s in subdirs_to_copy:
        init_file = out_dir / s / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            logger.info("  créé %s/__init__.py", s)

    # 4. Génération de main.py dans out_dir
    submit_main = Path(__file__).resolve().parent / "submit_main.py"
    (out_dir / "main.py").write_text(submit_main.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("✓ %s/main.py généré avec succès", out_dir)
    logger.info("=== Prêt pour soumission Kaggle ===")
    logger.info("1. Créez un dataset Kaggle avec TOUT le contenu de %s/", out_dir)
    logger.info("2. Attachez ce dataset à votre notebook de soumission")
    logger.info("3. Uploadez main.py (+ deck.csv) comme agent Kaggle")
    logger.info("Pour archiver : tar -czvf submission.tar.gz -C %s .", out_dir)


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

    if cfg.hf.enabled and cfg.hf.repo_id:
        try:
            from export.hub import load_from_hub
            logger.info("Tentative de chargement depuis HuggingFace Hub (%s)...", cfg.hf.repo_id)
            mz_params, deck_params, _, hf_step = load_from_hub(cfg.hf.repo_id, cfg=cfg)
            if mz_params:
                logger.info("Checkpoint HuggingFace Hub chargé avec succès (%s, step=%d)", cfg.hf.repo_id, hf_step)
                return mz_params, deck_params
        except Exception as e:
            logger.warning("Échec du chargement depuis HuggingFace Hub : %s", e)

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
