#!/usr/bin/env python3
"""
benchmark_iree_vulkan.py
========================
Benchmarks inference latency, throughput, and numerical precision
between native JAX (CPU) and compiled IREE (Vulkan GPU on AMD Radeon).

Usage:
    python benchmark_iree_vulkan.py --vmfb muzero_vulkan.vmfb -m HF -n 100
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_PJRT_PLUGIN_DISCOVERY"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_iree")

SCRIPT_DIR = Path(__file__).parent.resolve()
MUZERO_DIR = SCRIPT_DIR / "ptcg_muzero"
if MUZERO_DIR.exists():
    sys.path.insert(0, str(MUZERO_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from cards.encoder import CardStaticFeatures
from config import Config
from env.encoding import (
    GLOBAL_FEAT_DIM,
    OPTION_FEAT_DIM,
    POKEMON_FEAT_DIM,
)
from models.iree_engine import IREEMuZeroEngine
from models.networks import MuZeroNetwork


def resolve_local_path(path_str: str) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())
    basename = p.name
    candidates = [
        SCRIPT_DIR / "competiton" / basename,
        SCRIPT_DIR / "competition" / basename,
        SCRIPT_DIR / basename,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(p)


def _unflatten_params(flat: dict) -> dict:
    nested: dict = {}
    for key, val in flat.items():
        parts = key.split("/")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return nested


def _fetch_hf_checkpoint(hf_spec: str):
    from huggingface_hub import hf_hub_download
    spec = hf_spec.strip()
    repo_id = "richard151111/muzero-V2"
    step = 198000
    step_prefix = f"step_{step:07d}"
    sf_path = hf_hub_download(repo_id=repo_id, filename=f"{step_prefix}/muzero.safetensors")
    cfg_path = hf_hub_download(repo_id=repo_id, filename=f"{step_prefix}/config.json")
    return Path(sf_path), Path(cfg_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX CPU vs IREE Vulkan GPU.")
    parser.add_argument("--vmfb", type=str, default="muzero_vulkan.vmfb", help="Path to .vmfb module.")
    parser.add_argument("-m", "--safetensors", type=str, default="HF", help="Safetensors checkpoint path or 'HF'.")
    parser.add_argument("-n", "--iters", type=int, default=100, help="Number of benchmark iterations (default: 100).")
    parser.add_argument("--device", type=str, default="vulkan", help="IREE device ('vulkan', 'local-task').")
    args = parser.parse_args()

    vmfb_path = Path(args.vmfb).resolve()
    if not vmfb_path.exists():
        logger.error("VMFB file not found: %s. Please run export_iree.py first.", vmfb_path)
        sys.exit(1)

    # 1. Load Model & Config
    if args.safetensors.startswith("hf") or args.safetensors.startswith("HF"):
        sf_file, hf_cfg = _fetch_hf_checkpoint(args.safetensors)
        cfg = Config.load(str(hf_cfg))
    else:
        sf_file = Path(args.safetensors).resolve()
        cfg_file = sf_file.parent / "config.json"
        cfg = Config.load(str(cfg_file)) if cfg_file.exists() else Config()

    cfg.infra.card_csv = resolve_local_path(cfg.infra.card_csv)
    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    mz_flat = load_file(str(sf_file))
    mz_params = _unflatten_params(mz_flat)
    params = mz_params["muzero"] if "muzero" in mz_params else mz_params

    # 2. Prepare Sample Input
    B = 1
    sample_obs = {
        "global_feat": np.random.randn(B, GLOBAL_FEAT_DIM).astype(np.float32),
        "my_active_id": np.random.randint(1, 100, size=(B, 1), dtype=np.int32),
        "my_active_feat": np.random.randn(B, 1, POKEMON_FEAT_DIM).astype(np.float32),
        "my_bench_ids": np.random.randint(0, 100, size=(B, 5), dtype=np.int32),
        "my_bench_feat": np.random.randn(B, 5, POKEMON_FEAT_DIM).astype(np.float32),
        "my_bench_mask": np.ones((B, 5), dtype=np.float32),
        "my_hand_ids": np.random.randint(0, 100, size=(B, cfg.model.max_hand_size), dtype=np.int32),
        "my_hand_mask": np.ones((B, cfg.model.max_hand_size), dtype=np.float32),
        "my_discard_ids": np.random.randint(0, 100, size=(B, cfg.model.max_discard_size), dtype=np.int32),
        "my_discard_mask": np.ones((B, cfg.model.max_discard_size), dtype=np.float32),
        "opp_active_id": np.random.randint(1, 100, size=(B, 1), dtype=np.int32),
        "opp_active_feat": np.random.randn(B, 1, POKEMON_FEAT_DIM).astype(np.float32),
        "opp_bench_ids": np.random.randint(0, 100, size=(B, 5), dtype=np.int32),
        "opp_bench_feat": np.random.randn(B, 5, POKEMON_FEAT_DIM).astype(np.float32),
        "opp_bench_mask": np.ones((B, 5), dtype=np.float32),
        "opp_discard_ids": np.random.randint(0, 100, size=(B, cfg.model.max_discard_size), dtype=np.int32),
        "opp_discard_mask": np.ones((B, cfg.model.max_discard_size), dtype=np.float32),
        "opp_hand_ids": np.random.randint(0, 100, size=(B, cfg.model.max_hand_size), dtype=np.int32),
        "opp_hand_mask": np.ones((B, cfg.model.max_hand_size), dtype=np.float32),
        "option_ids": np.random.randint(0, 50, size=(B, cfg.model.max_actions), dtype=np.int32),
        "option_feat": np.random.randn(B, cfg.model.max_actions, OPTION_FEAT_DIM).astype(np.float32),
        "option_mask": np.ones((B, cfg.model.max_actions), dtype=np.float32),
    }

    # 3. Load IREE Engine
    logger.info("Initializing IREE Runtime engine (%s)...", args.device)
    iree_engine = IREEMuZeroEngine(vmfb_path=vmfb_path, device_uri=args.device)

    # 4. JAX Warmup & Inference
    logger.info("Running JAX CPU reference...")
    sample_obs_jax = {k: jnp.array(v) for k, v in sample_obs.items()}

    @jax.jit
    def jax_forward(obs):
        return network.apply(params, obs, deterministic=True)

    # Warmup
    z_jax, pi_jax, v_jax = jax_forward(sample_obs_jax)
    _ = jax.block_until_ready(z_jax)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        z_j, pi_j, v_j = jax_forward(sample_obs_jax)
        _ = jax.block_until_ready(z_j)
    t_jax = (time.perf_counter() - t0) / args.iters * 1000.0

    # 5. IREE Warmup & Inference
    logger.info("Running IREE Vulkan GPU inference...")
    z_iree, pi_iree, v_iree = iree_engine.forward(sample_obs)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        z_i, pi_i, v_i = iree_engine.forward(sample_obs)
    t_iree = (time.perf_counter() - t0) / args.iters * 1000.0

    # 6. Precision / Accuracy Check
    mae_z = float(np.mean(np.abs(np.array(z_jax) - z_iree)))
    mae_pi = float(np.mean(np.abs(np.array(pi_jax) - pi_iree)))
    mae_v = float(np.mean(np.abs(np.array(v_jax) - v_iree)))

    print("\n" + "=" * 65)
    print("📊 BENCHMARK REPORT: JAX CPU vs IREE VULKAN GPU")
    print("=" * 65)
    print(f"Device:                 {iree_engine.config}")
    print(f"Iterations:             {args.iters}")
    print("-" * 65)
    print(f"JAX CPU Latency:        {t_jax:.2f} ms / forward pass ({1000.0/t_jax:.1f} inferences/sec)")
    print(f"IREE GPU Latency:       {t_iree:.2f} ms / forward pass ({1000.0/t_iree:.1f} inferences/sec)")
    print("-" * 65)
    print("Numerical Precision (Difference between JAX and IREE):")
    print(f"  • Latent State (z) MAE:      {mae_z:.6e}")
    print(f"  • Policy (pi) MAE:           {mae_pi:.6e}")
    print(f"  • Value (v) MAE:             {mae_v:.6e}")
    print("=" * 65)
    if mae_z < 1e-4 and mae_v < 1e-4:
        print("✅ Numerical Verification: PASSED (Outputs are identical!)")
    else:
        print("⚠️ Warning: Significant deviation detected.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
