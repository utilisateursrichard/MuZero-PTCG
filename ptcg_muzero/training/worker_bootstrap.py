"""
ptcg_muzero/training/worker_bootstrap.py
=========================================
Point d'entrée des sous-processus self-play.

CRITIQUE : Ce module ne doit PAS importer JAX au niveau module.
Les variables d'environnement doivent être posées AVANT le premier import JAX,
sinon JAX s'initialise avec le GPU (ce qui cause des conflits CUDA avec le
coordinateur et des allocations mémoire inutiles dans les workers CPU).
"""


def run(child_conn, worker_id, cfg):
    """
    Bootstrap exécuté dans le sous-processus spawné.
    Pose les env vars en tout premier, avant tout import JAX.
    """
    import os

    # ── Forcer JAX CPU-only AVANT le moindre import JAX ──────────────────────
    os.environ["JAX_PLATFORMS"]              = "cpu"   # nouvelle syntaxe JAX >= 0.4
    os.environ["JAX_PLATFORM_NAME"]          = "cpu"   # rétro-compat
    os.environ["CUDA_VISIBLE_DEVICES"]       = ""      # cacher les GPUs
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Désactiver les plugins PJRT CUDA pour éviter que XLA tente d'initialiser CUDA
    os.environ["JAX_EXCLUDE_PJRT_PLUGINS"] = "cuda,tpu,rocm,tpu_driver"

    # Limiter les pools de threads du CPU pour éviter la surcharge et la contention entre les workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"

    # Mocker le module jax_plugins de CUDA pour empêcher son chargement
    import sys
    sys.modules['jax_plugins.xla_cuda12'] = None



    import logging
    logging.getLogger("probes").setLevel(logging.WARNING)
    logging.getLogger("cards").setLevel(logging.WARNING)

    # ── Maintenant on peut importer sans déclencher CUDA ─────────────────────
    from env.wrapper import self_play_worker_fn
    self_play_worker_fn(child_conn, worker_id, cfg)
