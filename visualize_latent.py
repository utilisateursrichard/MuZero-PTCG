#!/usr/bin/env python3
"""
visualize_latent.py
===================
Standalone script to load a `muzero.safetensors` model (from local path or HuggingFace Hub),
simulate real self-play games to collect game states, and export a high-resolution PNG
visualizing the 2D projected latent space (z) via t-SNE, PCA, UMAP, or PHATE.

Features:
- Hugging Face Support: Load weights directly with `-m HF`, `-m hf@150000`, or `-m hf:owner/repo`.
- Default mode: 4x4 multi-panel grid displaying the latent space across all linear
  probes (11 probes + Predicted Value + True Game Outcome + Estimation Error + Game Step Progression).
- Simple / Core mode (`-s`, `--simple`): Disables probes and generates a clean 2x2 square
  grid containing the 4 foundational graphs (Predicted Value, True Game Outcome, Estimation Error, and Game Duration/Progression).

Usage:
    # 2x2 Core square grid from HuggingFace Hub:
    python visualize_latent.py -m HF -s --method phate -g 50 -p 8 -o showcase

    # Full 4x4 multi-probe grid with local checkpoint:
    python visualize_latent.py -m checkpoints/muzero.safetensors --gpu 1 --games 6 --method umap
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
MUZERO_DIR = SCRIPT_DIR / "ptcg_muzero"
if MUZERO_DIR.exists():
    sys.path.insert(0, str(MUZERO_DIR))

# Force JAX to use CPU backend cleanly when not using GPU
if "--vulkan" not in sys.argv and "--iree" not in sys.argv and "--gpu" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_PJRT_PLUGIN_DISCOVERY"] = "false"

# Configure CPU parallelism for UMAP, t-SNE, and linear algebra (Numba, OpenMP, BLAS)
_cpu_count = str(os.cpu_count() or 4)
os.environ.setdefault("NUMBA_NUM_THREADS", _cpu_count)
os.environ.setdefault("OMP_NUM_THREADS", _cpu_count)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _cpu_count)
os.environ.setdefault("MKL_NUM_THREADS", _cpu_count)

# Initial logging setup before importing JAX
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("visualize_latent")


def resolve_local_path(path_str: str) -> str:
    """Automatically resolves Kaggle/relative paths to local workspace files."""
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

    logger.warning("File not found: %s (using raw path)", path_str)
    return path_str


def _parse_hf_spec(spec: str | None) -> tuple[str | None, int | None] | None:
    """Parses HuggingFace model reference passed to -m / --safetensors.

    Accepted forms (case-insensitive):
        HF                       -> latest checkpoint from cfg.hf.repo_id
        hf@120000                -> specific step from cfg.hf.repo_id
        hf:owner/repo            -> latest checkpoint from another repo
        hf:owner/repo@120000     -> specific step from another repo
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
            raise ValueError(f"Invalid HF step in -m {spec!r}: {raw_step!r} is not an integer.")
    body = body.strip()
    return (body or None, step)


def _fetch_hf_checkpoint(spec: str, default_repo: str = "richard151111/muzero-V2") -> tuple[Path, Path | None]:
    """Downloads muzero.safetensors and config.json from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download, get_token

    parsed = _parse_hf_spec(spec)
    if parsed is None:
        raise ValueError(f"Not a valid HF spec: {spec}")

    repo, step = parsed
    repo = repo or default_repo
    if not repo:
        raise ValueError("No Hugging Face repository specified. Use -m hf:owner/repo")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        try:
            token = get_token()
        except Exception:
            token = None

    if step is None:
        try:
            latest_file = hf_hub_download(repo_id=repo, filename="latest.json", token=token)
            latest_data = json.loads(Path(latest_file).read_text())
            step = int(latest_data.get("step", 0))
            logger.info("[hf] Latest checkpoint on Hub is step %d", step)
        except Exception:
            pass

    subfolder = f"step_{step:07d}" if step is not None else ""
    logger.info("[hf] Fetching muzero.safetensors from %s (%s)...", repo, f"step {step}" if step is not None else "latest")

    sf_path_str = hf_hub_download(
        repo_id=repo,
        filename=f"{subfolder}/muzero.safetensors" if subfolder else "muzero.safetensors",
        token=token,
    )

    cfg_path = None
    try:
        cfg_path_str = hf_hub_download(
            repo_id=repo,
            filename=f"{subfolder}/config.json" if subfolder else "config.json",
            token=token,
        )
        cfg_path = Path(cfg_path_str)
    except Exception:
        pass

    return Path(sf_path_str), cfg_path


def _unflatten_params(flat: dict) -> dict:
    """Reconstructs a nested Flax pytree from a flat safetensors dictionary with backward-compatibility adaptation."""
    nested: dict = {}
    for key, val in flat.items():
        # Handle backward compatibility for option_proj kernel (legacy shape 160 vs modern 162)
        if "option_proj" in key and key.endswith("kernel") and getattr(val, "ndim", 0) == 2 and val.shape[0] < 162:
            logger.info("Adapting legacy checkpoint weight '%s' from %s to (162, %d)", key, val.shape, val.shape[1])
            pad_rows = 162 - val.shape[0]
            val = np.pad(val, ((0, pad_rows), (0, 0)), mode="constant", constant_values=0.0)

        parts = key.split("/")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return nested


# ─────────────────────────────────────────────────────────────────────────────
# Worker-level Persistent Singletons (Loaded ONCE per worker process)
# ─────────────────────────────────────────────────────────────────────────────
_G_CARD_DATA = None
_G_ENGINE = None
_G_NETWORK = None
_G_DECK_NET = None
_G_PARAMS = None
_G_CFG = None
_G_NUM_CARD_IDS = 0
_G_ENERGY_IDS = []
_G_SHARED_COUNTER = None

EXPECTED_IREE_KEYS = (
    "global_feat", "my_active_feat", "my_active_id", "my_bench_feat", "my_bench_ids", "my_bench_mask",
    "my_discard_ids", "my_discard_mask", "my_hand_ids", "my_hand_mask", "opp_active_feat", "opp_active_id",
    "opp_bench_feat", "opp_bench_ids", "opp_bench_mask", "opp_discard_ids", "opp_discard_mask",
    "opp_hand_ids", "opp_hand_mask", "option_feat", "option_ids", "option_mask"
)
INT_IREE_KEYS = {
    "my_active_id", "my_bench_ids", "my_discard_ids", "my_hand_ids",
    "opp_active_id", "opp_bench_ids", "opp_discard_ids", "opp_hand_ids", "option_ids"
}


def _run_batched_self_play(
    num_games: int,
    num_workers: int,
    sf_file: Path,
    cfg: Config,
    device,
    seed: int = 42,
):
    """Executes parallel self-play with CPU workers running game engine and central GPU/JAX batched inference."""
    import multiprocessing as mp
    import queue as _pyqueue
    import threading
    import time
    from tqdm import tqdm
    import numpy as np
    import jax
    import jax.numpy as jnp
    from safetensors.numpy import load_file
    from training.worker_bootstrap import run as _worker_bootstrap
    from models.deck_builder import DEFAULT_COMPETITIVE_DECK
    from cards.encoder import CardStaticFeatures
    from models.networks import MuZeroNetwork

    ctx = mp.get_context("spawn")
    pipes = []
    processes = []

    for i in range(num_workers):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_worker_bootstrap, args=(child_conn, i, cfg), daemon=True)
        p.start()
        pipes.append(parent_conn)
        processes.append(p)

    # Preload network parameters onto central inference device
    mz_flat = load_file(str(sf_file))
    mz_params = _unflatten_params(mz_flat)
    params = mz_params if "muzero" in mz_params else {"muzero": mz_params}
    gpu_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params)
    mz_core_params = gpu_params["muzero"] if "muzero" in gpu_params else gpu_params

    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))
    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)

    work_queue = _pyqueue.Queue()
    stop_event = threading.Event()
    pipe_locks = [threading.Lock() for _ in range(num_workers)]
    pipe_epoch = [0] * num_workers

    def _gpu_inference_worker():
        while not stop_event.is_set():
            try:
                first = work_queue.get(timeout=0.030)
            except _pyqueue.Empty:
                continue

            items = [first]
            t0 = time.perf_counter()
            while len(items) < num_workers and (time.perf_counter() - t0) < 0.008:
                try:
                    items.append(work_queue.get(timeout=0.001))
                except _pyqueue.Empty:
                    break

            na_indices = [it[0] for it in items]
            na_encs = [it[1] for it in items]
            na_masks = [it[2] for it in items]
            na_epochs = [it[3] for it in items]

            batch_keys = list(na_encs[0].keys())
            batched_obs = {
                k: jnp.array(np.stack([x[k] for x in na_encs], axis=0))
                for k in batch_keys
            }
            omasks_np = np.stack(na_masks, axis=0)

            try:
                z_b = network.apply(mz_core_params, batched_obs, method=network.represent)
                pi_logits_b, v_b = network.apply(mz_core_params, z_b, method=network.predict)
                pi_np = np.array(pi_logits_b)
                v_np = np.array(v_b).reshape(-1)

                masked = np.where(omasks_np, pi_np, -1e9)
                best_actions = np.argmax(masked, axis=1)

                for i_item, pidx in enumerate(na_indices):
                    with pipe_locks[pidx]:
                        if pipe_epoch[pidx] != na_epochs[i_item]:
                            continue
                        act = int(best_actions[i_item])
                        pipes[pidx].send({
                            "action_indices": [act],
                            "search_pol": np.zeros(cfg.model.max_actions, dtype=np.float32),
                            "search_val": float(v_np[i_item]),
                        })
            except Exception:
                for i_item, pidx in enumerate(na_indices):
                    with pipe_locks[pidx]:
                        if pipe_epoch[pidx] != na_epochs[i_item]:
                            continue
                        mask = na_masks[i_item]
                        valid_acts = np.where(mask)[0]
                        act = int(np.random.choice(valid_acts) if len(valid_acts) > 0 else 0)
                        pipes[pidx].send({
                            "action_indices": [act],
                            "search_pol": np.zeros(cfg.model.max_actions, dtype=np.float32),
                            "search_val": 0.0,
                        })

    gpu_thread = threading.Thread(target=_gpu_inference_worker, daemon=True, name="batched-gpu-inference")
    gpu_thread.start()

    deck0 = list(DEFAULT_COMPETITIVE_DECK)
    deck1 = list(DEFAULT_COMPETITIVE_DECK)

    games_started = 0
    games_completed = 0
    pipe_meta = [{} for _ in range(num_workers)]

    def _start_game(pidx):
        nonlocal games_started
        pipe_meta[pidx]["game_active"] = True
        with pipe_locks[pidx]:
            pipes[pidx].send({
                "cmd": "start",
                "deck0": list(deck0),
                "deck1": list(deck1),
                "num_belief_samples": 1,
            })
        games_started += 1

    for i in range(num_workers):
        if games_started < num_games:
            _start_game(i)
        else:
            pipe_meta[i]["game_active"] = False

    collected_obs = []
    collected_returns = []
    collected_player_idx = []
    collected_steps = []
    collected_probes = []

    pbar_games = tqdm(total=num_games, desc="🎮 Games       ", unit="game", position=0, leave=True, dynamic_ncols=True)
    pbar_trans = tqdm(total=num_games * 200, desc="🔄 Transitions ", unit="trans", position=1, leave=True, dynamic_ncols=True)

    t_sim_start = time.perf_counter()

    try:
        while games_completed < num_games:
            active_pipes = [pipes[i] for i in range(num_workers) if pipe_meta[i].get("game_active")]
            if not active_pipes:
                break

            ready = list(mp.connection.wait(active_pipes, timeout=2.0))
            for p in active_pipes:
                if p not in ready and p.poll():
                    ready.append(p)

            for pipe in ready:
                pidx = pipes.index(pipe)
                if not pipe.poll():
                    continue

                try:
                    msg = pipe.recv()
                except Exception:
                    pipe_meta[pidx]["game_active"] = False
                    games_completed += 1
                    pbar_games.update(1)
                    continue

                status = msg.get("status")
                if status == "need_action":
                    work_queue.put((pidx, msg["batched_enc"], msg["option_mask"], pipe_epoch[pidx]))

                elif status == "game_over":
                    h0 = msg.get("hist0")
                    h1 = msg.get("hist1")
                    for h in (h0, h1):
                        if h and len(h.observations) > 0:
                            ret = 1.0 if h.game_won is True else (-1.0 if h.game_won is False else 0.0)
                            for s_idx, (obs, prb) in enumerate(zip(h.observations, h.probe_targets)):
                                collected_obs.append(obs)
                                collected_returns.append(ret)
                                collected_player_idx.append(h.player_idx)
                                collected_steps.append(s_idx + 1)
                                collected_probes.append(prb)

                    games_completed += 1
                    pbar_games.update(1)
                    total_c = len(collected_obs)
                    pbar_trans.n = total_c
                    elapsed = max(1e-5, time.perf_counter() - t_sim_start)
                    trans_rate = total_c / elapsed
                    avg_g = total_c / games_completed
                    pbar_trans.total = max(total_c, int(round(avg_g * num_games)))
                    pbar_trans.set_postfix({"avg/game": f"{avg_g:.0f}", "trans/s": f"{trans_rate:.1f}"})
                    pbar_trans.refresh()

                    if games_started < num_games:
                        _start_game(pidx)
                    else:
                        pipe_meta[pidx]["game_active"] = False

                elif status == "error":
                    if games_started < num_games:
                        _start_game(pidx)
                    else:
                        pipe_meta[pidx]["game_active"] = False
                        games_completed += 1
                        pbar_games.update(1)

    finally:
        stop_event.set()
        pbar_trans.n = len(collected_obs)
        pbar_trans.total = len(collected_obs)
        pbar_trans.refresh()
        pbar_games.close()
        pbar_trans.close()

        # Stop workers cleanly
        for pidx, pipe in enumerate(pipes):
            try:
                with pipe_locks[pidx]:
                    pipe.send({"cmd": "stop"})
            except Exception:
                pass

        for p in processes:
            p.join(timeout=0.5)
            if p.is_alive():
                p.terminate()

    return collected_obs, collected_returns, collected_player_idx, collected_steps, collected_probes, games_completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the 2D projected MuZero latent space (z) from self-play games."
    )
    parser.add_argument(
        "--safetensors",
        "-m",
        type=str,
        default=None,
        help="Path to muzero.safetensors or HuggingFace spec ('HF', 'hf@150000', 'hf:owner/repo').",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config.json (optional).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output PNG image path (default: 'latent_space.png' in single mode, 'latent_space_all_probes.png' in full mode).",
    )
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=4,
        help="Number of self-play games to simulate (default: 4).",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        "--workers",
        "-w",
        type=int,
        default=4,
        dest="workers",
        help="Number of parallel game workers running concurrently (default: 4).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "pca", "umap", "phate"],
        default="tsne",
        help="Dimensionality reduction method ('tsne', 'pca', 'umap', 'phate'). Default: 'tsne'.",
    )
    # UMAP tuning flags
    parser.add_argument(
        "--umap-neighbors",
        "--n-neighbors",
        type=int,
        default=15,
        dest="umap_neighbors",
        help="UMAP number of nearest neighbors (default: 15). Try 30-50 for more global structure, or 5-10 for fine local clusters.",
    )
    parser.add_argument(
        "--umap-min-dist",
        "--min-dist",
        type=float,
        default=0.1,
        dest="umap_min_dist",
        help="UMAP minimum distance between points (default: 0.1). Use 0.01 or 0.001 to pack points into tight archipelagos/clusters.",
    )
    parser.add_argument(
        "--umap-metric",
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "correlation", "manhattan"],
        dest="umap_metric",
        help="UMAP distance metric ('euclidean', 'cosine', 'correlation', 'manhattan'). Default: 'euclidean'.",
    )
    parser.add_argument(
        "--umap-init",
        "--init",
        type=str,
        default="spectral",
        choices=["spectral", "random", "pca"],
        dest="umap_init",
        help="UMAP initialization method ('spectral', 'random', 'pca'). Default: 'spectral'. Use 'pca' or 'random' to avoid spherical artifacts.",
    )
    parser.add_argument(
        "--umap-spread",
        type=float,
        default=1.0,
        help="UMAP effective scale/spread of embedded points (default: 1.0).",
    )
    # t-SNE tuning flags
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="t-SNE perplexity (default: auto-computed min(30, max(5, N // 10))).",
    )
    # Subsampling flag
    parser.add_argument(
        "--max-states",
        "--subsample",
        type=int,
        default=None,
        dest="max_states",
        help="Maximum number of states to plot (randomly subsamples if N > max_states to avoid dense over-plotting).",
    )
    parser.add_argument(
        "--simple",
        "-s",
        action="store_true",
        help="Core 2x2 grid mode: omit linear probes and produce a 2x2 square grid of the 4 foundational graphs (3 value metrics + turn duration).",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["v_pred", "v_real", "error", "step"],
        default="v_pred",
        help="Attribute to color points by in single plot mode: 'v_pred', 'v_real', 'error', 'step' (default: 'v_pred').",
    )
    parser.add_argument(
        "--vulkan",
        "--iree",
        action="store_true",
        help="Use IREE Vulkan GPU acceleration for self-play game simulation and latent state (z) computation on AMD Radeon GPU.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index to use (e.g., 0 or 1, default: None). Enabling this activates GPU acceleration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation and projection (default: 42).",
    )

    args = parser.parse_args()
    return args, args.gpu


def main():
    args, gpu_id = parse_args()

    if gpu_id is not None:
        os.environ["MESA_VK_DEVICE_SELECT"] = str(gpu_id)
        os.environ["DRI_PRIME"] = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("cards").setLevel(logging.WARNING)
    logging.getLogger("probes").setLevel(logging.WARNING)
    logging.getLogger("iree_engine").setLevel(logging.WARNING)
    logging.getLogger("ptcg_muzero").setLevel(logging.WARNING)

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from safetensors.numpy import load_file

    from cards.encoder import CardStaticFeatures
    from config import Config
    from interpretability.probes import PROBE_DEFS, extract_probe_targets
    from models.networks import MuZeroNetwork

    devices = jax.devices()
    logger.info("⚡ Main JAX Device: %s", devices[0])

    # 1. Model and Config resolution (Local file or HuggingFace Hub)
    sf_input = args.safetensors
    hf_cfg_path = None

    if sf_input is not None and _parse_hf_spec(sf_input) is not None:
        # Load from HuggingFace Hub
        sf_file, hf_cfg_path = _fetch_hf_checkpoint(sf_input)
    elif sf_input is not None:
        # Local file path
        sf_file = Path(sf_input).resolve()
        if not sf_file.exists():
            logger.error("Safetensors file does not exist: %s", sf_input)
            sys.exit(1)
    else:
        # Auto-detect local checkpoints
        ckpt_dir = SCRIPT_DIR / "checkpoints"
        found = list(ckpt_dir.glob("**/muzero.safetensors")) if ckpt_dir.exists() else []
        if found:
            sf_file = found[-1].resolve()
            logger.info("Automatically detected local safetensors file: %s", sf_file)
        else:
            logger.info("No local safetensors found. Attempting to fetch default HuggingFace Hub model...")
            sf_file, hf_cfg_path = _fetch_hf_checkpoint("HF")

    # 2. Config loading
    cfg_path = args.config or (str(hf_cfg_path) if hf_cfg_path else str(sf_file.parent / "config.json"))
    if Path(cfg_path).exists():
        cfg = Config.load(cfg_path)
        logger.info("Configuration loaded from %s", cfg_path)
    else:
        cfg = Config()
        cfg_path = ""
        logger.info("Using default configuration.")

    cfg.infra.card_csv = resolve_local_path(cfg.infra.card_csv)

    # 3. GPU / Vulkan VMFB Module check & compilation
    use_vulkan = args.vulkan or (gpu_id is not None)
    vmfb_path = SCRIPT_DIR / "muzero_vulkan.vmfb"

    if use_vulkan:
        if not vmfb_path.exists():
            logger.info("⚡ Compiling muzero_vulkan.vmfb for GPU self-play & latent inference on-the-fly...")
            from export_iree import export_and_compile
            class CompileArgs:
                safetensors = str(sf_file)
                config = str(cfg_path)
                output = str(vmfb_path)
                target = "vulkan"
                save_mlir = None
                batch_size = 1
            export_and_compile(CompileArgs())
        logger.info("⚡ GPU Acceleration enabled (IREE Vulkan) for both Self-Play simulation and Latent space computation!")

    # 4. Batched Game simulation (CPU workers + central batched GPU/JAX inference)
    num_workers = max(1, min(args.games, args.workers))
    sim_device_desc = f"JAX ({devices[0]})"
    logger.info("⚡ Simulating %d games using %d parallel CPU workers + Batched inference on %s...", args.games, num_workers, sim_device_desc)

    collected_obs, collected_returns, collected_player_idx, collected_steps, collected_probes, completed = _run_batched_self_play(
        num_games=args.games,
        num_workers=num_workers,
        sf_file=sf_file,
        cfg=cfg,
        device=devices[0],
        seed=args.seed,
    )

    if not collected_obs:
        logger.error("No observations could be collected.")
        sys.exit(1)

    N = len(collected_obs)
    logger.info("Total real game states collected: %d (from %d completed games)", N, completed)

    # 5. Extract latent states z and predictions v via vectorized JAX micro-batches
    import gc
    z_chunks = []
    v_chunks = []
    logger.info("⚡ Computing %d latent representations z and predictions (batched on %s)...", N, devices[0])
    mz_flat = load_file(str(sf_file))
    mz_params = _unflatten_params(mz_flat)
    params = mz_params if "muzero" in mz_params else {"muzero": mz_params}
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, devices[0]), params)

    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))
    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)

    obs_keys = collected_obs[0].keys()
    mz_core_params = params["muzero"] if "muzero" in params else params
    micro_batch_size = 1024
    with tqdm(total=N, desc="⚡ Latent states (z)", unit="state", dynamic_ncols=True) as pbar:
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
            pbar.update(end_i - start_i)

    z_np = np.concatenate(z_chunks, axis=0)                  # [N, latent_dim]
    v_pred_np = np.concatenate(v_chunks, axis=0)              # [N]
    v_real_np = np.array(collected_returns, dtype=np.float32) # [N]
    v_err_np  = np.abs(v_pred_np - v_real_np)                 # [N]
    step_np   = np.array(collected_steps, dtype=np.int32)     # [N] (Step / Turn number)

    del collected_obs, z_chunks, v_chunks
    gc.collect()

    raw_targets_np = np.stack(collected_probes)

    # Optional subsampling
    if args.max_states and args.max_states < N:
        logger.info("Subsampling %d states out of %d for clearer visualization...", args.max_states, N)
        rng_sub = np.random.default_rng(args.seed)
        sub_indices = rng_sub.choice(N, size=args.max_states, replace=False)
        sub_indices.sort()
        z_np = z_np[sub_indices]
        v_pred_np = v_pred_np[sub_indices]
        v_real_np = v_real_np[sub_indices]
        v_err_np = v_err_np[sub_indices]
        step_np = step_np[sub_indices]
        raw_targets_np = raw_targets_np[sub_indices]
        N = len(z_np)

    # 6. 2D Dimensionality Reduction
    logger.info("Projecting %d latent states to 2D via %s...", N, args.method.upper())
    if args.method == "umap":
        try:
            import umap
            try:
                import numba
                numba.set_num_threads(mp.cpu_count())
            except Exception:
                pass
            n_cores = mp.cpu_count()
            logger.info(
                "Computing UMAP on CPU (%d threads, n_neighbors=%d, min_dist=%g, metric='%s', init='%s')...",
                n_cores,
                args.umap_neighbors,
                args.umap_min_dist,
                args.umap_metric,
                args.umap_init,
            )
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                init=args.umap_init,
                spread=args.umap_spread,
                n_jobs=-1,
                random_state=None,  # Required by UMAP to enable full multi-core CPU parallelism
            )
            z_2d = reducer.fit_transform(z_np)
        except Exception as exc:
            logger.warning("UMAP execution failed (%s). Falling back to t-SNE on CPU.", exc)
            from sklearn.manifold import TSNE
            perplexity = args.perplexity or min(30, max(5, N // 10))
            z_2d = TSNE(n_components=2, perplexity=perplexity, n_jobs=-1, random_state=args.seed).fit_transform(z_np)

    elif args.method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = args.perplexity or min(30, max(5, N // 10))
            logger.info("Computing t-SNE on CPU (perplexity=%g)...", perplexity)
            reducer = TSNE(n_components=2, perplexity=perplexity, n_jobs=-1, random_state=args.seed)
            z_2d = reducer.fit_transform(z_np)
        except ImportError:
            z_centered = z_np - np.mean(z_np, axis=0)
            _, _, vh = np.linalg.svd(z_centered)
            z_2d = z_centered @ vh[:2].T

    elif args.method == "phate":
        try:
            import phate
            logger.info("Running PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)...")
            reducer = phate.PHATE(n_components=2, random_state=args.seed, n_jobs=-1, verbose=0)
            z_2d = reducer.fit_transform(z_np)
        except Exception as exc:
            logger.warning("PHATE execution failed (%s). Falling back to t-SNE.", exc)
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

    # 7. Visualization Plot Generation
    plt.style.use("dark_background")

    # Output file handling
    out_name = args.output
    if out_name is None:
        out_name = "latent_space.png" if args.simple else "latent_space_all_probes.png"
    elif not out_name.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        out_name = f"{out_name}.png"
    out_file = Path(out_name).resolve()

    if args.simple:
        logger.info("Generating 2x2 square grid plot (3 Value metrics + Game Duration, probes omitted)...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=300)
        axes_flat = axes.flatten()

        fig.suptitle(
            f"MuZero Latent Space 2D Projection ({args.method.upper()}) — {N:,} States from {completed} Games",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # 0. Model Predicted Value (v_pred)
        ax0 = axes_flat[0]
        sc0 = ax0.scatter(z_2d[:, 0], z_2d[:, 1], c=v_pred_np, cmap="coolwarm", s=24, alpha=0.85, edgecolors="none")
        ax0.set_title("0. Model Predicted Value (v_pred)", fontsize=12, fontweight="bold", pad=8)
        ax0.set_xlabel(f"{args.method.upper()} Dim 1", fontsize=9)
        ax0.set_ylabel(f"{args.method.upper()} Dim 2", fontsize=9)
        ax0.grid(True, linestyle="--", alpha=0.25)
        ax0.set_box_aspect(1)
        cbar0 = fig.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.04)
        cbar0.set_label("Expects Loss (-1) / Expects Win (+1)", fontsize=9)
        cbar0.set_ticks([-1, -0.5, 0, 0.5, 1])

        # 1. True Final Value (v_real)
        ax1 = axes_flat[1]
        sc1 = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=v_real_np, cmap="coolwarm", s=24, alpha=0.85, edgecolors="none")
        ax1.set_title("1. True Game Outcome (v_real)", fontsize=12, fontweight="bold", pad=8)
        ax1.set_xlabel(f"{args.method.upper()} Dim 1", fontsize=9)
        ax1.set_ylabel(f"{args.method.upper()} Dim 2", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.25)
        ax1.set_box_aspect(1)
        cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Lost (-1) / Won (+1)", fontsize=9)
        cbar1.set_ticks([-1, 0, 1])

        # 2. Estimation Error (|v_pred - v_real|)
        ax2 = axes_flat[2]
        sc2 = ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=v_err_np, cmap="plasma", s=24, alpha=0.85, edgecolors="none")
        ax2.set_title("2. Estimation Error (|v_pred - v_real|)", fontsize=12, fontweight="bold", pad=8)
        ax2.set_xlabel(f"{args.method.upper()} Dim 1", fontsize=9)
        ax2.set_ylabel(f"{args.method.upper()} Dim 2", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.25)
        ax2.set_box_aspect(1)
        cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Error (0 = Exact, 2 = Max Error)", fontsize=9)

        # 3. Game Progression / Duration (Turn / Step)
        ax3 = axes_flat[3]
        sc3 = ax3.scatter(z_2d[:, 0], z_2d[:, 1], c=step_np, cmap="viridis", s=24, alpha=0.85, edgecolors="none")
        ax3.set_title("3. Game Progression / Duration (Turn / Step)", fontsize=12, fontweight="bold", pad=8)
        ax3.set_xlabel(f"{args.method.upper()} Dim 1", fontsize=9)
        ax3.set_ylabel(f"{args.method.upper()} Dim 2", fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.25)
        ax3.set_box_aspect(1)
        cbar3 = fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("Turn / Step in Game (1 → End)", fontsize=9)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    else:
        logger.info("Generating 4x4 probe grid plot...")
        fig, axes = plt.subplots(4, 4, figsize=(22, 20), dpi=300)
        axes_flat = axes.flatten()

        fig.suptitle(
            f"MuZero Latent Space Mapping (z) — {N:,} States from {completed} Games ({args.method.upper()})",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # 0. Model Predicted Value
        ax0 = axes_flat[0]
        sc0 = ax0.scatter(z_2d[:, 0], z_2d[:, 1], c=v_pred_np, cmap="coolwarm", s=20, alpha=0.85)
        ax0.set_title("0. Model Predicted Value (v_pred)", fontsize=11, fontweight="bold", pad=8)
        cbar0 = plt.colorbar(sc0, ax=ax0)
        cbar0.set_label("Expects Loss (-1) / Expects Win (+1)", fontsize=8)
        ax0.grid(True, linestyle="--", alpha=0.25)

        # 1. True Final Value
        ax1 = axes_flat[1]
        sc1 = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=v_real_np, cmap="coolwarm", s=20, alpha=0.85)
        ax1.set_title("1. True Game Outcome (v_real)", fontsize=11, fontweight="bold", pad=8)
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label("Lost (-1) / Won (+1)", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.25)

        # 2. Estimation Error
        ax2 = axes_flat[2]
        sc2 = ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=v_err_np, cmap="plasma", s=20, alpha=0.85)
        ax2.set_title("2. Estimation Error (|v_pred - v_real|)", fontsize=11, fontweight="bold", pad=8)
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label("Error (0 = Exact, 2 = Max Error)", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.25)

        binary_colors = {0: "#ff7043", 1: "#00e676", -1: "#555555"}
        binary_labels = {0: "No (0)", 1: "Yes (1)", -1: "Unknown"}

        tri_colors = {0: "#ff5252", 1: "#40c4ff", 2: "#b2ff59", -1: "#555555"}
        tri_labels = {0: "Disadvantage (0)", 1: "Neutral (1)", 2: "Advantage (2)", -1: "Unknown"}

        probe_title_map = {
            "active_in_ko_range": "Active in KO Range",
            "type_advantage": "Type Advantage",
            "prize_lead": "Prize Lead",
            "hand_advantage": "Hand Advantage",
            "opp_energy_ready": "Opp. Energy Ready",
            "opp_bench_attacker_ready": "Opp. Bench Attacker Ready",
            "gust_ko_opportunity": "Gust KO Opportunity",
            "deck_out_risk": "Deck-Out Risk",
            "evolution_in_hand": "Evolution in Hand",
            "ko_next_turn_probable": "KO Next Turn Probable",
            "energy_attachment_available": "Energy Attachment Available",
        }

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

            p_title = probe_title_map.get(name, name.replace("_", " ").title())
            ax.set_title(f"{ax_target_idx}. Probe: {p_title}", fontsize=11, fontweight="bold", pad=8)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
            ax.grid(True, linestyle="--", alpha=0.25)

        # 14. Game Progression (Turn / Step)
        ax14 = axes_flat[14]
        sc14 = ax14.scatter(z_2d[:, 0], z_2d[:, 1], c=step_np, cmap="viridis", s=20, alpha=0.85)
        ax14.set_title("14. Game Progression (Turn / Step)", fontsize=11, fontweight="bold", pad=8)
        cbar14 = plt.colorbar(sc14, ax=ax14)
        cbar14.set_label("Turn / Step in Game (1 → End)", fontsize=8)
        ax14.grid(True, linestyle="--", alpha=0.25)

        for unused_idx in range(15, len(axes_flat)):
            axes_flat[unused_idx].set_visible(False)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("✓ Plot successfully saved → %s", out_file)


if __name__ == "__main__":
    main()
