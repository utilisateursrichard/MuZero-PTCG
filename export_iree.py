#!/usr/bin/env python3
"""
export_iree.py
==============
Exports JAX / Flax MuZero neural network models into StableHLO MLIR,
and compiles them with IREE into Vulkan SPIR-V (.vmfb) bytecode modules
ready for lightweight, native GPU inference on AMD Radeon GPUs without ROCm.

Usage:
    # 1. Export and compile for Vulkan GPU (AMD Radeon RX 6500M / 680M):
    python export_iree.py -m HF -o muzero_vulkan.vmfb --target vulkan

    # 2. Export with intermediate StableHLO MLIR file saved:
    python export_iree.py -m checkpoints/muzero.safetensors -o muzero.vmfb --save-mlir muzero.mlir

    # 3. Compile for CPU (LLVM-CPU fallback):
    python export_iree.py -m HF -o muzero_cpu.vmfb --target cpu
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Prevent noisy JAX backend logs during export
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_PJRT_PLUGIN_DISCOVERY"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_iree")

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


def _fetch_hf_checkpoint(hf_spec: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download
    spec = hf_spec.strip()
    if spec.upper() == "HF" or spec.lower() == "hf":
        repo_id = "richard151111/muzero-V2"
        step = None
    elif "@" in spec:
        left, right = spec.split("@", 1)
        repo_id = "richard151111/muzero-V2" if left.lower() == "hf" else left.removeprefix("hf:")
        step = int(right)
    elif ":" in spec:
        repo_id = spec.removeprefix("hf:")
        step = None
    else:
        repo_id = "richard151111/muzero-V2"
        step = None

    import json
    if step is None:
        latest_file = hf_hub_download(repo_id=repo_id, filename="latest.json")
        with open(latest_file, "r") as f:
            meta = json.load(f)
        step = meta.get("latest_step", 198000)

    step_prefix = f"step_{step:07d}"
    logger.info("Fetching weights from HuggingFace Hub: %s (%s)...", repo_id, step_prefix)
    sf_path = hf_hub_download(repo_id=repo_id, filename=f"{step_prefix}/muzero.safetensors")
    cfg_path = hf_hub_download(repo_id=repo_id, filename=f"{step_prefix}/config.json")
    return Path(sf_path), Path(cfg_path)


def create_sample_observations(cfg: Config, batch_size: int = 1) -> Dict[str, jnp.ndarray]:
    """Generates dummy observation arrays with accurate tensor shapes and dtypes."""
    B = batch_size
    H = cfg.model.max_hand_size
    D = cfg.model.max_discard_size
    A = cfg.model.max_actions

    return {
        "global_feat": jnp.zeros((B, GLOBAL_FEAT_DIM), dtype=jnp.float32),
        "my_active_id": jnp.zeros((B, 1), dtype=jnp.int32),
        "my_active_feat": jnp.zeros((B, 1, POKEMON_FEAT_DIM), dtype=jnp.float32),
        "my_bench_ids": jnp.zeros((B, 5), dtype=jnp.int32),
        "my_bench_feat": jnp.zeros((B, 5, POKEMON_FEAT_DIM), dtype=jnp.float32),
        "my_bench_mask": jnp.zeros((B, 5), dtype=jnp.float32),
        "my_hand_ids": jnp.zeros((B, H), dtype=jnp.int32),
        "my_hand_mask": jnp.zeros((B, H), dtype=jnp.float32),
        "my_discard_ids": jnp.zeros((B, D), dtype=jnp.int32),
        "my_discard_mask": jnp.zeros((B, D), dtype=jnp.float32),
        "opp_active_id": jnp.zeros((B, 1), dtype=jnp.int32),
        "opp_active_feat": jnp.zeros((B, 1, POKEMON_FEAT_DIM), dtype=jnp.float32),
        "opp_bench_ids": jnp.zeros((B, 5), dtype=jnp.int32),
        "opp_bench_feat": jnp.zeros((B, 5, POKEMON_FEAT_DIM), dtype=jnp.float32),
        "opp_bench_mask": jnp.zeros((B, 5), dtype=jnp.float32),
        "opp_discard_ids": jnp.zeros((B, D), dtype=jnp.int32),
        "opp_discard_mask": jnp.zeros((B, D), dtype=jnp.float32),
        "opp_hand_ids": jnp.zeros((B, H), dtype=jnp.int32),
        "opp_hand_mask": jnp.zeros((B, H), dtype=jnp.float32),
        "option_ids": jnp.zeros((B, A), dtype=jnp.int32),
        "option_feat": jnp.zeros((B, A, OPTION_FEAT_DIM), dtype=jnp.float32),
        "option_mask": jnp.zeros((B, A), dtype=jnp.float32),
    }


def export_and_compile(args):
    # 1. Model resolution
    sf_input = args.safetensors
    if sf_input.startswith("hf") or sf_input.startswith("HF"):
        sf_file, hf_cfg = _fetch_hf_checkpoint(sf_input)
        cfg_path = args.config or str(hf_cfg)
    else:
        sf_file = Path(sf_input).resolve()
        if not sf_file.exists():
            logger.error("Safetensors file not found: %s", sf_input)
            sys.exit(1)
        cfg_path = args.config or str(sf_file.parent / "config.json")

    # 2. Config & Static Features
    if Path(cfg_path).exists():
        cfg = Config.load(cfg_path)
        logger.info("Loaded config from %s", cfg_path)
    else:
        cfg = Config()
        logger.info("Using default config.")

    cfg.infra.card_csv = resolve_local_path(cfg.infra.card_csv)
    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    # 3. Model instantiation & weights loading
    logger.info("Instantiating Flax MuZeroNetwork...")
    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    mz_flat = load_file(str(sf_file))
    mz_params = _unflatten_params(mz_flat)
    params = mz_params["muzero"] if "muzero" in mz_params else mz_params

    # 4. Generate sample inputs
    sample_obs = create_sample_observations(cfg, batch_size=args.batch_size)
    ordered_keys = sorted(sample_obs.keys())
    sample_args = [sample_obs[k] for k in ordered_keys]

    logger.info("Lowering MuZero forward pass to StableHLO MLIR...")

    def forward_fn(*flat_args):
        obs_dict = {k: v for k, v in zip(ordered_keys, flat_args)}
        z, pi, v = network.apply(params, obs_dict, deterministic=True)
        return z, pi, v

    lowered = jax.jit(forward_fn).lower(*sample_args)
    mlir_module = lowered.compiler_ir(dialect="stablehlo")

    if args.save_mlir:
        mlir_path = Path(args.save_mlir).resolve()
        mlir_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mlir_path, "w") as f:
            f.write(str(mlir_module))
        logger.info("Saved intermediate StableHLO MLIR to: %s", mlir_path)

    # 5. IREE Compilation
    logger.info("Compiling StableHLO MLIR with IREE (target: %s)...", args.target)
    try:
        import iree.compiler as ireec
    except ImportError:
        logger.error(
            "iree-base-compiler is not installed. Please run: pip install iree-base-compiler iree-base-runtime"
        )
        sys.exit(1)

    extra_args = ["--iree-opt-level=O3"]
    if args.target == "vulkan":
        target_backend = "vulkan-spirv"
    elif args.target == "cpu":
        target_backend = "llvm-cpu"
    else:
        target_backend = args.target

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        compiled_vmfb = ireec.compile_str(
            str(mlir_module),
            target_backends=[target_backend],
            extra_args=extra_args,
            input_type="stablehlo",
        )
        with open(out_path, "wb") as f:
            f.write(compiled_vmfb)
        logger.info("✅ Successfully compiled IREE bytecode module to: %s (%d KB)", out_path, len(compiled_vmfb) // 1024)
    except Exception as exc:
        logger.warning("Target-specific compilation failed with error: %s", exc)
        logger.info("Retrying with generic %s backend...", target_backend)
        compiled_vmfb = ireec.compile_str(
            str(mlir_module),
            target_backends=[target_backend],
            input_type="stablehlo",
        )
        with open(out_path, "wb") as f:
            f.write(compiled_vmfb)
        logger.info("✅ Successfully compiled IREE bytecode module to: %s (%d KB)", out_path, len(compiled_vmfb) // 1024)


def parse_args():
    parser = argparse.ArgumentParser(description="Export JAX MuZero network to MLIR and compile with IREE.")
    parser.add_argument(
        "-m",
        "--safetensors",
        type=str,
        default="HF",
        help="Path to muzero.safetensors or HuggingFace spec (e.g., 'HF', 'hf@198000').",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="muzero_vulkan.vmfb",
        help="Output path for compiled .vmfb bytecode module.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="vulkan",
        choices=["vulkan", "cpu", "vulkan-spirv", "llvm-cpu"],
        help="IREE target compilation backend (default: 'vulkan').",
    )
    parser.add_argument(
        "--save-mlir",
        type=str,
        default=None,
        help="Optional path to save the intermediate StableHLO .mlir file.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Static inference batch size (default: 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_and_compile(args)
