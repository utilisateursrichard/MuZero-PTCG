"""
modal_train.py
==============
Script d'entraînement Cloud sur Modal.com (Modal 1.5+) avec GPU L4 (1 GPU).

Utilisation :
-------------
1. Se connecter à Modal (si pas encore fait) :
       modal setup

2. Lancer l'entraînement en mode détaché (recommandé pour éviter les coupures gRPC/heartbeat local) :
       modal run --detach modal_train.py
"""

import os
import sys
from pathlib import Path
import modal

local_dir = Path(__file__).parent.resolve()
req_path = local_dir / "ptcg_muzero" / "requirements.txt"

# Exclusion des dossiers lourds pour un upload ultra-rapide en 2 secondes
def _ignore_heavy_files(p: Path) -> bool:
    for part in p.parts:
        if part in (".venv", ".venv_gpu", ".git", "__pycache__", ".ipynb_checkpoints"):
            return True
    if p.suffix in (".zip", ".tar.gz"):
        return True
    return False

# ── 1. Image Docker avec JAX CUDA 12 GPU et requirements.txt ──────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "g++", "make", "curl")
    .pip_install(
        "jax[cuda12]",
        "jax-cuda12-plugin",
        "jax-cuda12-pjrt",
        find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        gpu="L4",
    )
    .pip_install_from_requirements(str(req_path))
    .add_local_dir(local_dir, remote_path="/root/workspace", ignore=_ignore_heavy_files)
)

# ── 2. Volume persistant pour sauvegarder les checkpoints dans le Cloud ──────
checkpoint_volume = modal.Volume.from_name("ptcg-muzero-checkpoints", create_if_missing=True)

# ── 3. Application Modal ──────────────────────────────────────────────────────
app = modal.App("ptcg-muzero-training")


@app.function(
    image=image,
    gpu="L4",
    timeout=86400,  # 24 heures max de runtime
    volumes={"/root/workspace/checkpoints": checkpoint_volume},
    secrets=[
        modal.Secret.from_dict({
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "") or os.environ.get("WANDB", ""),
            "WANDB": os.environ.get("WANDB", "") or os.environ.get("WANDB_API_KEY", ""),
        })
    ],
)
def run_training(extra_args: str = ""):
    """
    Fonction distante exécutée sur GPU L4 (1 GPU) dans Modal Cloud.
    Équivalent de : python ptcg_muzero/main.py train --devices 1
    """
    import subprocess

    workspace = Path("/root/workspace")
    os.chdir(str(workspace))

    # Ajout du workspace et de ptcg_muzero au PYTHONPATH
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    if str(workspace / "ptcg_muzero") not in sys.path:
        sys.path.insert(0, str(workspace / "ptcg_muzero"))

    # Vérification et garantie d'initialisation du GPU par JAX
    import jax
    if jax.devices()[0].platform == "cpu":
        print("⚠️ CUDA GPU inactive. Immediately installing/activating JAX CUDA 12 (jax-cuda12-plugin)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-U",
            "jax[cuda12]", "jax-cuda12-plugin", "jax-cuda12-pjrt",
            "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        ], check=True)
        import importlib
        import jax as jax_reloaded
        importlib.reload(jax_reloaded)
        jax = jax_reloaded

    devices = jax.devices()
    print("=========================================================")
    print(f"⚡ Modal Cloud GPU detected: {devices}")
    print(f"⚡ Device Platform: {devices[0].platform}")
    print("=========================================================")

    # Équivalent exact de votre cellule Kaggle : main.py train --devices 1
    cmd = [sys.executable, "ptcg_muzero/main.py", "train", "--devices", "1"]
    if extra_args:
        cmd.extend(extra_args.split())

    print(f"🚀 Launching command: {' '.join(cmd)}")
    sys.stdout.flush()

    # Force unbuffered output pour un streaming instantané des logs Modal
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    # Lancement du process
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    retcode = process.wait()

    # Garantir la persistance des checkpoints créés
    checkpoint_volume.commit()

    if retcode != 0:
        raise RuntimeError(f"Training exited with error code: {retcode}")

    print("✅ Training completed successfully on Modal Cloud (1 L4 GPU)!")


@app.local_entrypoint()
def main(extra_args: str = ""):
    """Point d'entrée local appelé par `modal run modal_train.py`."""
    print("=========================================================")
    print("🚀 Uploading code and building Modal Cloud container...")
    print("=========================================================")
    run_training.remote(extra_args=extra_args)
