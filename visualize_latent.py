#!/usr/bin/env python3
"""
visualize_latent.py
===================
Script standalone (hors du dossier ptcg_muzero) permettant de charger un fichier
`muzero.safetensors` (ou un checkpoint), de simuler des parties pour récolter
des états de jeu réels, et d'enregistrer une image PNG représentant l'espace latent (z)
compressé en 2D via t-SNE / PCA / UMAP avec TOUTES LES SONDES (11 sondes + Valeur Prédite + Valeur Réelle + Erreur).

Usage :
    python visualize_latent.py --safetensors /chemin/vers/muzero.safetensors --gpu 1 --games 6 --method umap
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
MUZERO_DIR = SCRIPT_DIR / "ptcg_muzero"
if MUZERO_DIR.exists():
    sys.path.insert(0, str(MUZERO_DIR))

# Forcer le flush en temps réel des logs dans le terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Configuration initiale des logs avant import JAX
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("visualize_latent")


def resolve_local_path(path_str: str) -> str:
    """Résout automatiquement les chemins Kaggle/relatifs vers les fichiers locaux du workspace."""
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())

    basename = p.name
    candidates = [
        SCRIPT_DIR / "competiton" / basename,
        SCRIPT_DIR / "competiton" / "sample_submission" / basename,
        SCRIPT_DIR / "competiton" / "sample_submission" / "sample_submission" / basename,
        SCRIPT_DIR / "ptcg_muzero" / "data" / basename,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())

    logger.warning("Fichier introuvable : %s (utilisation du chemin brut)", path_str)
    return path_str


def _unflatten_params(flat: dict) -> dict:
    """Reconstruit un pytree Flax à partir d'un dictionnaire plat safetensors."""
    nested: dict = {}
    for key, val in flat.items():
        parts = key.split("/")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return nested


def _run_single_game_process(args_dict: dict):
    """Exécute une partie de self-play dans un processus totalement isolé (évite les pointeurs C++ corrompus)."""
    g_idx = args_dict["g_idx"]
    sf_path = args_dict["sf_path"]
    cfg_path = args_dict["cfg_path"]
    gpu_id = args_dict["gpu_id"]
    seed = args_dict["seed"]

    # Forcer l'affichage temps réel dans les sous-processus
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # Isolation environnementale GPU pour ce processus (AMD Radeon RX 6500M / 680M gfx1034 / gfx1035)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Chemins des bibliothèques ROCm / PyTorch
    torch_lib = str(SCRIPT_DIR / ".venv_gpu" / "lib" / "python3.12" / "site-packages" / "torch" / "lib")
    compat_lib = str(SCRIPT_DIR / ".venv_gpu" / "lib" / "rocm_compat")
    curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{compat_lib}:{torch_lib}:/opt/rocm/lib:{curr_ld}"

    import jax
    import jax.numpy as jnp
    import numpy as np
    from safetensors.numpy import load_file

    from cards.encoder import CardStaticFeatures
    from config import Config
    from env.wrapper import run_self_play_game
    from models.deck_builder import (
        DeckBuilderNetwork,
        sample_deck,
        set_ace_spec_ids,
        set_basic_pokemon_ids,
        set_energy_ids,
    )
    from models.networks import MuZeroNetwork
    from training.trainer import make_agent_fn

    cfg = Config.load(cfg_path) if cfg_path and Path(cfg_path).exists() else Config()
    cfg.infra.card_csv = resolve_local_path(cfg.infra.card_csv)
    cfg.infra.reference_deck_csv = resolve_local_path(cfg.infra.reference_deck_csv)

    mz_flat = load_file(str(sf_path))
    mz_params = _unflatten_params(mz_flat)
    params = mz_params if "muzero" in mz_params else {"muzero": mz_params}

    device = jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params)

    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    energy_ids = [cid for cid in card_data.card_ids if "Energy" in card_data._cards[cid].get("stage", "")]
    set_energy_ids(energy_ids)
    basic_ids = [
        cid for cid in card_data.card_ids
        if card_data._cards[cid].get("stage", "").strip().lower() in ("basic pokémon", "basic pokemon")
    ]
    set_basic_pokemon_ids(basic_ids)

    ace_spec_set = set(card_data.ace_spec_ids)
    try:
        from env.cabt_api import all_card_data
        for c in all_card_data():
            if getattr(c, "aceSpec", False):
                ace_spec_set.add(c.cardId)
    except Exception:
        pass
    set_ace_spec_ids(list(ace_spec_set))

    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    deck_net = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)

    rng = jax.random.PRNGKey(seed + g_idx * 100)
    rng, r1, r2, r3 = jax.random.split(rng, 4)
    deck_params = deck_net.init(r1)

    agent_fn = make_agent_fn(network, params, cfg, r3, train_mode=False)
    np_rng = np.random.default_rng(seed + g_idx * 1000 + 1)

    deck_logits, _ = deck_net.apply(deck_params)
    deck0, _ = sample_deck(deck_logits[0], r2, num_card_ids, energy_ids)
    deck1 = deck0[:]

    h0, h1 = run_self_play_game(agent_fn, deck0, deck1, cfg, np_rng)

    res_obs, res_raw, res_ret, res_pidx, res_step = [], [], [], [], []
    for h in (h0, h1):
        if h and len(h.observations) > 0:
            outcome = 1.0 if h.game_won is True else (-1.0 if h.game_won is False else 0.0)
            for step_idx, (obs, raw) in enumerate(zip(h.observations, h.raw_states)):
                res_obs.append(obs)
                res_raw.append(raw)
                res_ret.append(outcome)
                res_pidx.append(h.player_idx)
                res_step.append(step_idx + 1)

    return res_obs, res_raw, res_ret, res_pidx, res_step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Génère une image PNG de l'espace latent 2D d'un MuZero avec TOUTES LES SONDES + VRAIE VALEUR."
    )
    parser.add_argument(
        "--safetensors",
        "-s",
        type=str,
        default=None,
        help="Chemin vers le fichier muzero.safetensors.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Chemin vers config.json (optionnel).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="latent_space_all_probes.png",
        help="Nom du fichier PNG de sortie.",
    )
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=4,
        help="Nombre de parties de self-play à jouer (par défaut : 4).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "pca", "umap", "phate"],
        default="tsne",
        help="Méthode de réduction de dimensionnalité (tsne, pca, umap, phate). Default: tsne",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index du GPU physique (ex: 0 ou 1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire (par défaut: 42).")

    args = parser.parse_args()

    gpu_idx = args.gpu
    if gpu_idx is None:
        if sys.stdin.isatty():
            try:
                ans = input("\n🎮 Quel GPU voulez-vous utiliser ? (0 ou 1) [défaut: 0] : ").strip()
                gpu_idx = int(ans) if ans in ("0", "1") else 0
            except Exception:
                gpu_idx = 0
        else:
            gpu_idx = 0

    return args, gpu_idx


def main():
    args, selected_gpu_id = parse_args()

    # Isolation GPU pour le processus principal
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_id)
    os.environ["HIP_VISIBLE_DEVICES"] = str(selected_gpu_id)
    os.environ["ROCR_VISIBLE_DEVICES"] = str(selected_gpu_id)
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    torch_lib = str(SCRIPT_DIR / ".venv_gpu" / "lib" / "python3.12" / "site-packages" / "torch" / "lib")
    compat_lib = str(SCRIPT_DIR / ".venv_gpu" / "lib" / "rocm_compat")
    curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{compat_lib}:{torch_lib}:/opt/rocm/lib:{curr_ld}"

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from safetensors.numpy import load_file

    from cards.encoder import CardStaticFeatures
    from config import Config
    from interpretability.probes import PROBE_DEFS, extract_probe_targets
    from models.networks import MuZeroNetwork

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    devices = jax.devices()
    logger.info("⚡ Device JAX principal : %s (GPU physique : %d)", devices[0], selected_gpu_id)

    # 1. Recherche du fichier .safetensors
    sf_path = args.safetensors
    if sf_path is None:
        ckpt_dir = SCRIPT_DIR / "checkpoints"
        found = list(ckpt_dir.glob("**/muzero.safetensors")) if ckpt_dir.exists() else []
        if found:
            sf_path = str(found[-1])
            logger.info("Fichier safetensors détecté automatiquement : %s", sf_path)
        else:
            logger.error("Veuillez fournir un fichier --safetensors")
            sys.exit(1)

    sf_file = Path(sf_path).resolve()
    if not sf_file.exists():
        logger.error("Le fichier safetensors n'existe pas : %s", sf_path)
        sys.exit(1)

    # 2. Config
    cfg_path = args.config or str(sf_file.parent / "config.json")
    if Path(cfg_path).exists():
        cfg = Config.load(cfg_path)
        logger.info("Configuration chargée depuis %s", cfg_path)
    else:
        cfg = Config()
        cfg_path = ""
        logger.info("Utilisation de la configuration par défaut.")

    cfg.infra.card_csv = resolve_local_path(cfg.infra.card_csv)

    # 3. Simulation des parties (Séquentielle sur 1 seul processus pour compiler JAX 1 seule fois en 2s)
    # 3. Simulation des parties en PARALLÈLE via Processus Isolés (Multiprocessing SPAWN)
    logger.info("⚡ Simulation de %d parties en PARALLÈLE (Processus isolés)...", args.games)

    ctx = mp.get_context("spawn")
    work_items = [
        {
            "g_idx": g_i,
            "sf_path": str(sf_file),
            "cfg_path": str(cfg_path),
            "gpu_id": selected_gpu_id,
            "seed": args.seed,
        }
        for g_i in range(args.games)
    ]

    num_workers = min(args.games, os.cpu_count() or 4)
    collected_obs = []
    collected_raw = []
    collected_returns = []
    collected_player_idx = []
    collected_steps = []

    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_run_single_game_process, work_items)

    for res_obs, res_raw, res_ret, res_pidx, res_step in results:
        collected_obs.extend(res_obs)
        collected_raw.extend(res_raw)
        collected_returns.extend(res_ret)
        collected_player_idx.extend(res_pidx)
        collected_steps.extend(res_step)

    if not collected_obs:
        logger.error("Aucune observation n'a pu être récoltée.")
        sys.exit(1)

    N = len(collected_obs)
    logger.info("Total d'états réels récoltés : %d", N)

    # 4. Chargement du modèle pour le calcul des états z
    mz_flat = load_file(str(sf_file))
    mz_params = _unflatten_params(mz_flat)
    params = mz_params if "muzero" in mz_params else {"muzero": mz_params}
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, devices[0]), params)

    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)

    # 5. Extraction des états latents z par micro-batches
    import gc
    logger.info("Calcul des représentations latentes z et des prédictions (par micro-batches)...")
    obs_keys = collected_obs[0].keys()
    mz_core_params = params["muzero"]

    z_chunks = []
    v_chunks = []
    micro_batch_size = 64

    for start_i in range(0, N, micro_batch_size):
        end_i = min(start_i + micro_batch_size, N)
        sub_obs = {
            k: jnp.array(np.stack([collected_obs[i][k] for i in range(start_i, end_i)]))
            for k in obs_keys
        }
        sub_z = network.apply(mz_core_params, sub_obs, method=network.represent)
        _, sub_v = network.apply(mz_core_params, sub_z, method=network.predict)
        z_chunks.append(np.array(sub_z))
        v_chunks.append(np.array(sub_v).reshape(-1))

    z_np = np.concatenate(z_chunks, axis=0)                  # [N, latent_dim]
    v_pred_np = np.concatenate(v_chunks, axis=0)              # [N]
    v_real_np = np.array(collected_returns, dtype=np.float32) # [N]
    v_err_np  = np.abs(v_pred_np - v_real_np)                 # [N]
    step_np   = np.array(collected_steps, dtype=np.int32)     # [N] (Numéro de tour)

    del collected_obs, z_chunks, v_chunks
    gc.collect()

    raw_targets = [
        extract_probe_targets(raw, p_idx)
        for raw, p_idx in zip(collected_raw, collected_player_idx)
    ]
    raw_targets_np = np.stack(raw_targets)

    # 6. Réduction 2D
    logger.info("Projection 2D des états latents via %s ...", args.method.upper())
    if args.method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, N // 10))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed)
            z_2d = reducer.fit_transform(z_np)
        except ImportError:
            z_centered = z_np - np.mean(z_np, axis=0)
            _, _, vh = np.linalg.svd(z_centered)
            z_2d = z_centered @ vh[:2].T

    elif args.method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, low_memory=True, random_state=args.seed)
            z_2d = reducer.fit_transform(z_np)
        except Exception as exc:
            logger.warning("Échec UMAP (%s). Fallback sur t-SNE.", exc)
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, N // 10))
            z_2d = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed).fit_transform(z_np)

    elif args.method == "phate":
        try:
            import phate
            logger.info("Exécution de PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)...")
            reducer = phate.PHATE(n_components=2, random_state=args.seed, n_jobs=-1)
            z_2d = reducer.fit_transform(z_np)
        except Exception as exc:
            logger.warning("Échec ou absence du package PHATE (%s). Pour l'utiliser : 'pip install phate'. Fallback sur t-SNE.", exc)
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, N // 10))
            z_2d = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed).fit_transform(z_np)

    else:
        try:
            from sklearn.decomposition import PCA
            z_2d = PCA(n_components=2).fit_transform(z_np)
        except ImportError:
            z_centered = z_np - np.mean(z_np, axis=0)
            _, _, vh = np.linalg.svd(z_centered)
            z_2d = z_centered @ vh[:2].T

    # 7. Génération du graphique Matplotlib 14 panneaux
    logger.info("Génération de la grille de graphiques 4x4...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 4, figsize=(22, 20), dpi=300)
    axes_flat = axes.flatten()

    fig.suptitle(
        f"Cartographie Complète de l'Espace Latent MuZero (z) — {N} États sur GPU {selected_gpu_id} ({args.method.upper()})",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    ax0 = axes_flat[0]
    sc0 = ax0.scatter(z_2d[:, 0], z_2d[:, 1], c=v_pred_np, cmap="coolwarm", s=20, alpha=0.85)
    ax0.set_title("0. Valeur Prédite par le Modèle (v_pred)", fontsize=11, fontweight="bold", pad=8)
    cbar0 = plt.colorbar(sc0, ax=ax0)
    cbar0.set_label("Pense perdre (-1) / Pense gagner (+1)", fontsize=8)
    ax0.grid(True, linestyle="--", alpha=0.25)

    ax1 = axes_flat[1]
    sc1 = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=v_real_np, cmap="coolwarm", s=20, alpha=0.85)
    ax1.set_title("1. Vraie Valeur Finale Réelle (v_real)", fontsize=11, fontweight="bold", pad=8)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("A perdu (-1) / A gagné (+1)", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.25)

    ax2 = axes_flat[2]
    sc2 = ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=v_err_np, cmap="plasma", s=20, alpha=0.85)
    ax2.set_title("2. Erreur d'Estimation (|v_pred - v_real|)", fontsize=11, fontweight="bold", pad=8)
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Erreur (0 = Précis, 2 = Erreur Totale)", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.25)

    binary_colors = {0: "#ff7043", 1: "#00e676", -1: "#555555"}
    binary_labels = {0: "Non (0)", 1: "Oui (1)", -1: "Inconnu"}

    tri_colors = {0: "#ff5252", 1: "#40c4ff", 2: "#b2ff59", -1: "#555555"}
    tri_labels = {0: "Désavantage / Retard (0)", 1: "Neutre / Égalité (1)", 2: "Avantage / Avance (2)", -1: "Inconnu"}

    for p_idx, pdef in enumerate(PROBE_DEFS):
        ax_target_idx = p_idx + 3
        if ax_target_idx >= len(axes_flat):
            break
        ax = axes_flat[ax_target_idx]
        name = pdef["name"]
        num_classes = pdef["num_classes"]

        targets_p = raw_targets_np[:, p_idx] if p_idx < raw_targets_np.shape[1] else np.full(N, -1, dtype=np.int32)
        colors_map = tri_colors if num_classes == 3 else binary_colors
        labels_map = tri_labels if num_classes == 3 else binary_labels
        possible_cats = [0, 1, 2] if num_classes == 3 else [0, 1]

        for cat in possible_cats:
            mask = targets_p == cat
            if np.any(mask):
                ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=colors_map[cat], label=labels_map[cat], s=20, alpha=0.85)

    # 14. Âge / Avancement dans la partie (Numéro de Tour)
    ax14 = axes_flat[14]
    sc14 = ax14.scatter(z_2d[:, 0], z_2d[:, 1], c=step_np, cmap="viridis", s=20, alpha=0.85)
    ax14.set_title("14. Âge / Avancement dans la Partie (Numéro de Tour)", fontsize=11, fontweight="bold", pad=8)
    cbar14 = plt.colorbar(sc14, ax=ax14)
    cbar14.set_label("Tour / Étape de la partie (1 → Fin)", fontsize=8)
    ax14.grid(True, linestyle="--", alpha=0.25)

    for unused_idx in range(15, len(axes_flat)):
        axes_flat[unused_idx].set_visible(False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_file = Path(args.output)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("✓ Grille complète sur GPU %d sauvegardée → %s", selected_gpu_id, out_file.resolve())


if __name__ == "__main__":
    main()
