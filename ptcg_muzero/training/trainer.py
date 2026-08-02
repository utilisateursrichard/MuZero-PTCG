"""
ptcg_muzero/training/trainer.py
=================================
Boucle d'entraînement principale.

Architecture dual-GPU
---------------------
* Tous les paramètres sont répliqués sur les 2 devices via
  ``flax.jax_utils.replicate``.
* ``train_step`` est décoré avec ``jax.pmap``.  Chaque device traite
  ``batch_size // num_devices`` exemples.  Les gradients sont moyennés
  via ``jax.lax.pmean``.
* Le replay buffer, la self-play et le deck builder tournent sur CPU.

Flux global
-----------
1. Initialisation des réseaux + optimiseurs
2. Boucle :
   a. self-play  → remplir le replay buffer
   b. sample batch → collate → shard → train_step
   c. logging + checkpoint + HF push
"""
from __future__ import annotations

import functools
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.jax_utils
from flax.training import train_state

from cards.encoder import CardStaticFeatures, CardEmbedding, CARD_STATIC_DIM
from config import Config
from env.encoding import encode_observation
from env.wrapper import GameHistory, run_self_play_game, DeckError
from export.hub import push_to_hub
from interpretability.probes import (
    ProbeHeads,
    extract_probe_targets,
    probe_accuracy,
    probe_report,
)
from env.cabt_api import (
    all_card_data,
)
from models.deck_builder import (
    DeckBuilderNetwork,
    create_deck_train_state,
    deck_reinforce_update,
    sample_deck,
    set_ace_spec_ids,
    set_basic_pokemon_ids,
    set_energy_ids,
)
from models.networks import MuZeroNetwork
from search.ismcts import add_exploration_noise, ismcts_action, ismcts_action_batched, reanalyze_root
from training.loss import collate_batch, muzero_loss
from training.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)

_active_push_thread = None


# ─────────────────────────────────────────────────────────────────────────────
# TrainState étendu (MuZero + Probes dans un arbre unifié)
# ─────────────────────────────────────────────────────────────────────────────
class MuZeroTrainState(train_state.TrainState):
    """Étend TrainState pour stocker les métriques de step."""
    step_metrics: Dict = None


# ─────────────────────────────────────────────────────────────────────────────
# pmap'd train step
# ─────────────────────────────────────────────────────────────────────────────
@functools.partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
def _train_step(
    state: train_state.TrainState,
    batch: dict,
    # Les modules passés via closure (static) : voir make_train_step
) -> Tuple[train_state.TrainState, dict, jnp.ndarray]:
    """
    Un pas de gradient sur un shard du batch.
    Retourne (new_state, metrics, td_errors).

    Note : ``network`` et ``probe_heads`` sont dans la fermeture via
    ``make_train_step``; on ne les passe pas comme argument pour éviter
    les recompilations.
    """
    # ← Défini dynamiquement par make_train_step
    raise NotImplementedError("Use make_train_step() to build this function.")


def make_train_step(
    network: MuZeroNetwork,
    probe_heads: ProbeHeads,
    cfg: Config,
):
    """
    Fabrique le train_step pmappé, en fermant sur ``network`` et ``probe_heads``.
    Les modules Flax n'ont pas besoin d'être passés comme arguments — on les
    capture dans la closure avant jit/pmap.

    Paramètres complémentaires par rapport à l'ancienne version :
      rng : jnp.ndarray  — clé PRNG passée depuis le coordinateur et répliquée
            sur tous les devices.  À l'intérieur du pmap on utilise
            ``jax.random.fold_in(rng, axis_index)`` pour différencier les devices.
    """
    reanalyze_sims     = cfg.train.reanalyze_num_simulations
    reanalyze_consider = cfg.search.max_num_considered_actions

    def _step(
        state: train_state.TrainState,
        batch: dict,
        rng: jnp.ndarray,
        freeze_rep: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, dict, jnp.ndarray]:

        mz_params = state.params["muzero"]

        # ── Reanalyze In-Pipeline GPU ───────────────────────────────────────
        rng_device = jax.random.fold_in(rng, jax.lax.axis_index("devices"))

        # Extraire l'observation racine (step k=0) — déjà sur le GPU
        obs0 = {k: v[:, 0] for k, v in batch["obs_seq"].items()}  # [B, ...]

        z_root     = network.apply(mz_params, obs0, method=network.represent)
        pi_root, v_root = network.apply(mz_params, z_root, method=network.predict)

        mask_root = (batch["target_pol"][:, 0] > 0).astype(jnp.bool_)  # [B, A]

        fresh_pol, fresh_val = reanalyze_root(
            mz_params, network, z_root, pi_root, v_root, mask_root,
            rng_device, reanalyze_sims, reanalyze_consider,
        )

        batch = {
            **batch,
            "target_pol": batch["target_pol"].at[:, 0].set(jax.lax.stop_gradient(fresh_pol)),
            "target_val": batch["target_val"].at[:, 0].set(jax.lax.stop_gradient(fresh_val)),
        }

        # ── Gradient step ──────────────────────────────────────────────────────
        def loss_fn(params):
            loss, metrics, td_err = muzero_loss(
                params, network, probe_heads, batch,
                cfg.train, cfg.model,
            )
            return loss, (metrics, td_err)

        (loss, (metrics, td_err)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        # Annulation des gradients de h(s) si gelé
        def _zero_h_grads(g):
            if isinstance(g, dict) and "muzero" in g:
                mz_g = dict(g["muzero"])
                if "h" in mz_g:
                    mz_g["h"] = jax.tree_util.tree_map(jnp.zeros_like, mz_g["h"])
                g = {**g, "muzero": mz_g}
            return g

        grads = jax.lax.cond(
            freeze_rep,
            _zero_h_grads,
            lambda g: g,
            grads,
        )

        # Synchronise gradients across devices
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss  = jax.lax.pmean(loss,  axis_name="devices")

        # Never let one non-finite gradient contaminate Adam's moments
        grads_finite = jax.tree_util.tree_reduce(
            lambda ok, x: ok & jnp.all(jnp.isfinite(x)), grads, initializer=True
        )
        update_is_finite = jnp.isfinite(loss) & grads_finite
        update_is_finite = jax.lax.pmin(
            update_is_finite.astype(jnp.int32), axis_name="devices"
        ).astype(jnp.bool_)
        new_state = jax.lax.cond(
            update_is_finite,
            lambda _: state.apply_gradients(grads=grads),
            lambda _: state,
            operand=None,
        )
        metrics = {**metrics, "update_is_finite": update_is_finite.astype(jnp.float32)}
        return new_state, metrics, td_err

    return jax.pmap(_step, axis_name="devices", donate_argnums=(0,))



# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────
def create_muzero_train_state(
    network:     MuZeroNetwork,
    probe_heads: ProbeHeads,
    cfg:         Config,
    rng:         jnp.ndarray,
    dummy_obs:   dict,
) -> train_state.TrainState:
    """
    Initialise tous les paramètres et l'optimiseur.
    Les paramètres vivent dans un dict ``{"muzero": ..., "probes": ...}``.
    """
    rng_mz, rng_pr = jax.random.split(rng)

    # Init MuZero
    batch_obs = {k: v[None] for k, v in dummy_obs.items()}  # add batch dim
    mz_params = network.init(rng_mz, batch_obs)

    # Init ProbeHeads
    z_dummy  = jnp.zeros((1, cfg.model.latent_dim))
    pr_params = probe_heads.init(rng_pr, z_dummy)

    params = {"muzero": mz_params, "probes": pr_params}

    # LR schedule : linear warmup + cosine decay
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.train.learning_rate,
        warmup_steps=cfg.train.lr_warmup_steps,
        decay_steps=cfg.train.num_total_steps,
        end_value=cfg.train.learning_rate * 0.05,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad_norm),
        optax.adamw(lr_schedule, weight_decay=cfg.train.weight_decay),
    )

    return train_state.TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )


def _merge_params(defaults, loaded):
    """Conserve les poids/états restaurés et réinitialise les parties absentes ou incompatibles (forme / structure)."""
    is_defaults_map = isinstance(defaults, Mapping)
    is_loaded_map = isinstance(loaded, Mapping)

    if is_defaults_map and is_loaded_map:
        res = {}
        for k, v in defaults.items():
            if k in loaded:
                res[k] = _merge_params(v, loaded[k])
            else:
                res[k] = v
        for k, v in loaded.items():
            if k not in defaults:
                res[k] = v
        return res
    elif is_defaults_map != is_loaded_map:
        logger.warning(
            "Structure de couche/état incompatible détectée entre modèle et checkpoint ; réinitialisation de cette partie."
        )
        return defaults

    is_defaults_seq = isinstance(defaults, (tuple, list))
    is_loaded_seq = isinstance(loaded, (tuple, list))
    if is_defaults_seq and is_loaded_seq:
        if len(defaults) != len(loaded):
            return defaults
        merged_elems = [_merge_params(d, l) for d, l in zip(defaults, loaded)]
        if hasattr(defaults, "_fields"):  # NamedTuple
            return type(defaults)(*merged_elems)
        return type(defaults)(merged_elems)

    if hasattr(defaults, "shape") and hasattr(loaded, "shape"):
        if defaults.shape != loaded.shape:
            logger.warning(
                "Incompatibilité de forme détectée pour la couche (modèle %s vs checkpoint %s) ; réinitialisation aléatoire de cette couche.",
                defaults.shape, loaded.shape
            )
            return defaults
        return loaded

    return loaded



# ─────────────────────────────────────────────────────────────────────────────
# Agent function (utilisée pendant la self-play)
# ─────────────────────────────────────────────────────────────────────────────
def make_agent_fn(
    network:    MuZeroNetwork,
    params_cpu: dict,   # params non-répliqués (sur device 0)
    cfg:        Config,
    rng:        jnp.ndarray,
    train_mode: bool = True,
):
    """
    Retourne une AgentFn compatible avec ``run_self_play_game``.
    Les paramètres sont ceux du device 0 (non shardés).
    """
    _rng = rng

    def agent_fn(obs_dict, player_idx, _cfg):
        nonlocal _rng
        _rng, rng_act, rng_noise = jax.random.split(_rng, 3)

        select  = obs_dict.get("select") or {}
        options = select.get("option", [])
        n_opts  = len(options)
        if n_opts == 0:
            return [0], np.zeros(cfg.model.max_actions), 0.0

        enc_obs     = encode_observation(obs_dict, player_idx, cfg.model)
        option_mask = enc_obs["option_mask"]

        best_action, avg_policy, avg_value = ismcts_action(
            network, params_cpu["muzero"], enc_obs, option_mask, rng_act, cfg
        )

        if train_mode:
            avg_policy = np.array(add_exploration_noise(
                jnp.array(avg_policy), jnp.array(option_mask),
                rng_noise, cfg.search.dirichlet_alpha,
                cfg.search.dirichlet_epsilon,
            ))

        # maxCount > 1 : top-k par logit
        max_count = int(select.get("maxCount", 1))
        if max_count > 1:
            masked = np.where(option_mask, avg_policy, -1e9)
            sel_indices = np.argsort(masked)[::-1][:max_count].tolist()
        else:
            sel_indices = [best_action]

        return sel_indices, avg_policy, avg_value

    return agent_fn


# ─────────────────────────────────────────────────────────────────────────────
# Shard / unshard helpers
# ─────────────────────────────────────────────────────────────────────────────
def shard_batch(batch: dict, num_devices: int) -> dict:
    """Reshape each value [B, ...] → [D, B//D, ...]."""
    def _shard(x):
        B = x.shape[0]
        assert B % num_devices == 0, f"batch size {B} not divisible by {num_devices}"
        return x.reshape(num_devices, B // num_devices, *x.shape[1:])

    sharded = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            sharded[k] = {kk: _shard(vv) for kk, vv in v.items()}
        else:
            sharded[k] = _shard(v)
    return sharded


def unshard(x: jnp.ndarray) -> jnp.ndarray:
    """Reshape [D, B//D, ...] → [B, ...]."""
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def run_parallel_self_play(
    num_games_to_play: int,
    num_workers: int,
    deck_net,
    deck_params,
    network,
    state_params,
    cfg: Config,
    rng: jnp.ndarray,
    num_card_ids: int,
    energy_ids,
    ace_spec_ids,
    pipes: list,
    processes: list,
    is_seeding: bool = False,
    min_batch_threshold: int = -1,   # -1 = utiliser num_workers (batch GPU toujours complet)
    card_data = None,
):
    if is_seeding:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.search.num_simulations = 15
        cfg.search.num_belief_samples = 1
        logger.info("[self-play] Mode seeding active: acceleration MCTS (sims=15, belief_samples=1)")

    # Résoudre min_batch_threshold=-1 → num_workers (batch GPU toujours plein)
    if min_batch_threshold < 0:
        min_batch_threshold = num_workers

    import multiprocessing as mp
    from training.worker_bootstrap import run as _worker_bootstrap
    import numpy as np
    from training.activity import tracker

    # Pré-transférer les params sur les GPUs UNE SEULE FOIS (évite CPU→GPU à chaque inférence)
    devices = jax.devices()
    accel_devices = [d for d in devices if d.platform in ("gpu", "tpu")]
    if not accel_devices:
        accel_devices = devices  # fallback CPU
    logger.info("[self-play] Accelerators (%s) pour MCTS : %s", accel_devices[0].platform, accel_devices)
    _gpu_params = [
        jax.device_put(state_params, dev) for dev in accel_devices
    ]
    _inference_device_idx = [0]  # compteur pour alterner les GPUs

    ctx = mp.get_context("spawn")

    games_started = 0
    games_completed = 0
    completed_histories = []
    deck_builder_updates = []
    deck_errors_count = 0

    from models.deck_builder import sample_deck
    deck_logits, _ = deck_net.apply(deck_params)

    pipe_meta = [{} for _ in range(num_workers)]
    worker_steps = [0] * num_workers
    last_msg_time = [time.time()] * num_workers

    def start_game_on_worker(pipe_idx, local_rng):
        nonlocal games_started
        worker_steps[pipe_idx] = 0
        local_rng, r_d0, r_d1 = jax.random.split(local_rng, 3)
        deck0, ids0 = sample_deck(deck_logits[0], r_d0, num_card_ids, energy_ids, ace_spec_ids=ace_spec_ids)
        deck1, ids1 = sample_deck(deck_logits[0], r_d1, num_card_ids, energy_ids, ace_spec_ids=ace_spec_ids)
        
        # Réanimation robuste du worker s'il s'est arrêté (ex. suite à une exception)
        p = processes[pipe_idx]
        if not p.is_alive():
            logger.info(f"[coordinateur] Relance du worker {pipe_idx} qui s'était arrêté...")
            try:
                pipes[pipe_idx].close()
            except Exception:
                pass
            parent_conn, child_conn = ctx.Pipe()
            new_p = ctx.Process(target=_worker_bootstrap, args=(child_conn, pipe_idx, cfg), daemon=True)
            new_p.start()
            pipes[pipe_idx] = parent_conn
            processes[pipe_idx] = new_p

        pipe_meta[pipe_idx] = {
            "ids0": ids0,
            "ids1": ids1,
            "game_active": True
        }
        last_msg_time[pipe_idx] = time.time()
        try:
            pipes[pipe_idx].send({
                "cmd": "start",
                "deck0": list(deck0),
                "deck1": list(deck1),
                "num_belief_samples": int(cfg.search.num_belief_samples)
            })
            games_started += 1
        except Exception as e:
            logger.error(f"[coordinateur] Échec d'envoi au worker {pipe_idx} relancé : {e}")
            pipe_meta[pipe_idx]["game_active"] = False


    for i in range(num_workers):
        if games_started < num_games_to_play:
            rng, rng_start = jax.random.split(rng)
            start_game_on_worker(i, rng_start)
        else:
            pipe_meta[i] = {"game_active": False}

    # ── Architecture producer-consumer asynchrone ─────────────────────────
    # • Thread principal  : lit les pipes, gère game_over/deck_error,
    #                       met les need_action dans work_queue
    # • GPU thread i      : bloque sur work_queue, collecte 20ms,
    #                       lance l'inférence, envoie les résultats
    # Les deux GPUs fonctionnent totalement en parallèle sans se bloquer.
    # ──────────────────────────────────────────────────────────────────────
    import queue as _pyqueue
    import threading

    work_queue: "_pyqueue.Queue" = _pyqueue.Queue()
    stop_event = threading.Event()

    # RNG séparé par GPU thread (évite tout partage de state JAX entre threads)
    rng, _rng_gpu0, _rng_gpu1 = jax.random.split(rng, 3)
    _gpu_thread_rngs = [_rng_gpu0, _rng_gpu1]

    def _gpu_inference_worker(gpu_idx: int):
        _rng = _gpu_thread_rngs[gpu_idx]
        while not stop_event.is_set():
            # ── Attendre le premier item disponible (timeout court pour re-vérifier stop_event) ──
            try:
                first = work_queue.get(timeout=0.050)
            except _pyqueue.Empty:
                continue

            items = [first]

            # ── Collecte rapide (max 2ms) : agréger tous les workers déjà prêts ──
            _t0 = time.time()
            while time.time() - _t0 < 0.002:
                try:
                    items.append(work_queue.get_nowait())
                except _pyqueue.Empty:
                    break

            # ── Build batch padded à num_workers (taille fixe → pas de recompilation JAX) ──
            na_indices  = [it[0] for it in items]
            na_encs     = [it[1] for it in items]
            na_masks    = [it[2] for it in items]

            _pe = list(na_encs)
            _pm = list(na_masks)
            while len(_pe) < num_workers:
                _pe.append(na_encs[-1])
                _pm.append(na_masks[-1])

            _enc = {}
            for k in _pe[0].keys():
                _arr = np.stack([x[k] for x in _pe], axis=0)
                _enc[k] = jax.device_put(_arr, accel_devices[gpu_idx])
            _omasks = np.stack(_pm, axis=0)

            _rng, _rng_act = jax.random.split(_rng)
            _n = len(na_indices)

            try:
                _ba, _ap, _av = ismcts_action_batched(
                    network, _gpu_params[gpu_idx], _enc, _omasks, _rng_act, cfg
                )
                ba_np = np.array(_ba)
                ap_np = np.array(_ap)
                av_np = np.array(_av)
            except Exception as _e:
                logger.error("[Acc%d] Erreur inférence (batch=%d) : %s", gpu_idx, _n, _e, exc_info=True)
                ba_np = np.array([int(np.random.choice(np.where(na_masks[i])[0])) for i in range(_n)])
                ap_np = np.zeros((_n, int(cfg.model.max_actions)), dtype=np.float32)
                av_np = np.zeros(_n, dtype=np.float32)

            # ── Envoyer les résultats aux workers (chaque pipe est écrit par ce thread uniquement) ──
            for _i, _pidx in enumerate(na_indices):
                try:
                    pipes[_pidx].send({
                        "action_indices": [int(ba_np[_i])],
                        "search_pol":     ap_np[_i],
                        "search_val":     float(av_np[_i]),
                    })
                except Exception as _e:
                    logger.error("[Acc%d] Erreur envoi action worker %d : %s", gpu_idx, _pidx, _e)

    # Lancer autant de threads GPU/TPU qu'il y a d'accélérateurs
    gpu_threads = []
    for _gidx in range(len(_gpu_params)):
        _t = threading.Thread(target=_gpu_inference_worker, args=(_gidx,), daemon=True,
                              name=f"accel-inference-{_gidx}")
        _t.start()
        gpu_threads.append(_t)
        logger.info("[coordinateur] Accelerator inference thread %d démarré (device: %s)", _gidx, accel_devices[_gidx])

    # ── Boucle principale : I/O uniquement, pas d'inférence GPU ───────────
    while games_completed < num_games_to_play:
        active_pipes_list = [pipes[i] for i in range(num_workers) if pipe_meta[i].get("game_active")]
        if not active_pipes_list:
            break

        ready_pipes = list(mp.connection.wait(active_pipes_list, timeout=10.0))
        for p in active_pipes_list:
            if p not in ready_pipes and p.poll():
                ready_pipes.append(p)

        for pipe in ready_pipes:
            pipe_idx = pipes.index(pipe)
            if not pipe.poll():
                continue

            try:
                msg = pipe.recv()
                last_msg_time[pipe_idx] = time.time()
            except Exception as e:
                logger.error(f"Failed to read from worker {pipe_idx}: {e}")
                pipe_meta[pipe_idx]["game_active"] = False
                games_completed += 1
                continue

            status = msg.get("status")
            if status == "need_action":
                worker_steps[pipe_idx] = msg.get("step_count", 0)
                tracker.update(current_game_steps=int(np.max(worker_steps)))
                # → GPU thread se charge de l'inférence et de l'envoi du résultat
                work_queue.put((pipe_idx, msg["batched_enc"], msg["option_mask"]))

            else:
                tracker.update()

            if status == "deck_error":
                deck_errors_count += 1
                logger.warning(f"[coordinateur] Erreur de deck sur le worker {pipe_idx} : {msg.get('error')}")
                meta = pipe_meta[pipe_idx]
                for p_idx, ids in [("joueur 0", meta.get("ids0")), ("joueur 1", meta.get("ids1"))]:
                    if ids is not None:
                        from collections import Counter
                        counts = Counter(list(np.array(ids)))
                        deck_summary = []
                        for cid, count in counts.items():
                            name = card_data.card_name(int(cid)) if card_data else f"ID {cid}"
                            is_ace = int(cid) in ace_spec_ids
                            deck_summary.append(f"{name} (ID: {cid}, count: {count}{', ACE' if is_ace else ''})")
                        logger.info(f"  Deck de {p_idx} impliqué : {', '.join(deck_summary)}")
                if games_started < num_games_to_play:
                    rng, rng_restart = jax.random.split(rng)
                    start_game_on_worker(pipe_idx, rng_restart)
                else:
                    pipe_meta[pipe_idx]["game_active"] = False

            elif status == "game_over":
                h0, h1 = msg["hist0"], msg["hist1"]
                completed_histories.extend([h0, h1])
                if not is_seeding:
                    reward_0 = float(h0.game_won or False) * 2 - 1
                    reward_1 = float(h1.game_won or False) * 2 - 1
                    deck_builder_updates.append((pipe_meta[pipe_idx]["ids0"], reward_0))
                    deck_builder_updates.append((pipe_meta[pipe_idx]["ids1"], reward_1))
                games_completed += 1
                tracker.update(games_completed=tracker.games_completed + 1)
                logger.info(f"[coordinateur] Partie completee sur worker {pipe_idx} ! ({games_completed}/{num_games_to_play} completes, {games_started}/{num_games_to_play} lancees)")
                if games_started < num_games_to_play:
                    rng, rng_next = jax.random.split(rng)
                    start_game_on_worker(pipe_idx, rng_next)
                else:
                    pipe_meta[pipe_idx]["game_active"] = False

            elif status == "error":
                logger.error(f"[coordinateur] Le worker {pipe_idx} a envoye une erreur fatale : {msg.get('error')}")
                pipe_meta[pipe_idx]["game_active"] = False
                games_completed += 1
                if games_started < num_games_to_play:
                    logger.info(f"[coordinateur] Tentative de remplacement pour le worker {pipe_idx} en échec...")
                    rng, rng_restart = jax.random.split(rng)
                    start_game_on_worker(pipe_idx, rng_restart)

        # Détecter et relancer les workers gelés (inactivité > 120s)
        now = time.time()
        for idx in range(num_workers):
            if pipe_meta[idx].get("game_active") and (now - last_msg_time[idx] > 120.0):
                logger.warning(
                    f"[coordinateur] Le worker {idx} semble gele (inactif depuis {now - last_msg_time[idx]:.1f}s, etape {worker_steps[idx]}). Force restart..."
                )
                try:
                    processes[idx].terminate()
                    processes[idx].join(timeout=2.0)
                    if processes[idx].is_alive():
                        processes[idx].kill()
                except Exception as e:
                    logger.error(f"[coordinateur] Erreur lors de la coupure du worker {idx} : {e}")
                try:
                    pipes[idx].close()
                except Exception:
                    pass
                parent_conn, child_conn = ctx.Pipe()
                new_p = ctx.Process(target=_worker_bootstrap, args=(child_conn, idx, cfg), daemon=True)
                new_p.start()
                pipes[idx] = parent_conn
                processes[idx] = new_p
                pipe_meta[idx] = {"game_active": False}
                last_msg_time[idx] = now
                worker_steps[idx] = 0
                if games_started < num_games_to_play:
                    rng, rng_restart = jax.random.split(rng)
                    start_game_on_worker(idx, rng_restart)
                else:
                    games_completed += 1

    # ── Arrêt propre des GPU threads ──────────────────────────────────────
    stop_event.set()
    for _t in gpu_threads:
        _t.join(timeout=5.0)

    return completed_histories, deck_builder_updates, deck_errors_count



PROBE_NAMES = [
    "active_in_ko_range",
    "type_advantage",
    "prize_lead",
    "hand_advantage",
    "opp_energy_ready",
    "opp_bench_attacker_ready",
    "gust_ko_opportunity",
    "deck_out_risk",
    "evolution_in_hand",
    "ko_next_turn_probable",
    "energy_attachment_available",
]


def _init_wandb(cfg: Config):
    if not getattr(cfg, "wandb", None) or not cfg.wandb.enabled:
        return None
    import os
    
    # 1. Vérifier si un Secret Kaggle existe (nommé 'WANDB')
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret(cfg.wandb.kaggle_secret_name)
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            logger.info("[wandb] Clé API récupérée depuis le secret Kaggle '%s'", cfg.wandb.kaggle_secret_name)
    except Exception:
        pass

    # 2. Fallback si WANDB est défini comme variable d'environnement au lieu de WANDB_API_KEY
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB"):
        os.environ["WANDB_API_KEY"] = os.environ.get("WANDB")

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("[wandb] Aucune clé WANDB_API_KEY ou secret Kaggle '%s' trouvé. WandB sera désactivé.", cfg.wandb.kaggle_secret_name)
        return None

    try:
        import wandb
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=asdict(cfg),
            mode=cfg.wandb.mode,
            reinit=True,
        )
        url = run.url if hasattr(run, "url") else ""
        logger.info("[wandb] wandb.init() réussi ! (project=%s, run_id=%s, url=%s)", cfg.wandb.project, run.id, url)
        return run
    except Exception as e:
        logger.warning("[wandb] Impossible d'initialiser WandB : %s", e)
        return None


def _log_wandb_full_metrics(
    metrics: dict,
    buffer,
    batch_cpu: dict,
    td_err_flat: np.ndarray,
    step: int,
    elapsed: float,
    rep_frozen: bool,
    cfg: Config,
):
    if not getattr(cfg, "wandb", None) or not cfg.wandb.enabled:
        return
    try:
        import wandb
        if wandb.run is None:
            return

        w_dict = {}

        # ── 1. Losses ──────────────────────────────────────────────────────────
        w_dict["loss/total"]       = float(metrics["loss_total"][0])
        w_dict["loss/policy"]      = float(metrics["loss_policy"][0])
        w_dict["loss/value"]       = float(metrics["loss_value"][0])
        w_dict["loss/reward"]      = float(metrics["loss_reward"][0])
        w_dict["loss/probes"]      = float(metrics["loss_probes"][0])
        if "loss_consistency" in metrics:
            w_dict["loss/consistency"] = float(metrics["loss_consistency"][0])

        # ── 2. TD Error & Priority Stats ───────────────────────────────────────
        if td_err_flat is not None and len(td_err_flat) > 0:
            w_dict["td_error/mean"] = float(np.mean(td_err_flat))
            w_dict["td_error/max"]  = float(np.max(td_err_flat))
            w_dict["td_error/min"]  = float(np.min(td_err_flat))
            w_dict["td_error/std"]  = float(np.std(td_err_flat))

        # ── 3. Value Target Distribution Stats ─────────────────────────────────
        if "target_val" in batch_cpu:
            v_targets = np.array(batch_cpu["target_val"])
            w_dict["targets/val_mean"] = float(np.mean(v_targets))
            w_dict["targets/val_std"]  = float(np.std(v_targets))
            w_dict["targets/val_min"]  = float(np.min(v_targets))
            w_dict["targets/val_max"]  = float(np.max(v_targets))

        # ── 4. Detailed Probe Metrics (Per Task) ──────────────────────────────
        if "probe_per_task" in metrics:
            p_losses = np.array(metrics["probe_per_task"][0])
            for i, name in enumerate(PROBE_NAMES):
                if i < len(p_losses):
                    w_dict[f"probes_loss/{name}"] = float(p_losses[i])

        if "probe_acc_per_task" in metrics:
            p_accs = np.array(metrics["probe_acc_per_task"][0]) * 100.0
            w_dict["probes/accuracy_mean"] = float(np.mean(p_accs))
            for i, name in enumerate(PROBE_NAMES):
                if i < len(p_accs):
                    w_dict[f"probes_acc/{name}"] = float(p_accs[i])

        # ── 5. Replay Buffer Stats ─────────────────────────────────────────────
        w_dict["buffer/size"] = len(buffer)
        if hasattr(buffer, "_sum_tree"):
            total_prio = float(buffer._sum_tree.query_all())
            w_dict["buffer/total_priority"] = total_prio
            w_dict["buffer/avg_priority"]   = total_prio / max(len(buffer), 1)
        if hasattr(buffer, "get_max_priority"):
            w_dict["buffer/max_priority"]   = float(buffer.get_max_priority())

        # ── 6. Performance & System Metrics ────────────────────────────────────
        w_dict["perf/step_time_sec"] = elapsed
        w_dict["perf/steps_per_sec"] = 1.0 / (elapsed + 1e-8)
        w_dict["perf/rep_frozen"]    = 1.0 if rep_frozen else 0.0

        wandb.log(w_dict, step=step)

    except Exception as e:
        logger.debug("[wandb] Erreur lors du log étendu : %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: Config) -> None:
    """Point d'entrée principal d'entraînement."""
    logger.info("=== PTCG MuZero training start ===")
    logger.info("Devices: %s", jax.devices())

    Path(cfg.infra.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.infra.log_dir).mkdir(parents=True, exist_ok=True)

    _init_wandb(cfg)

    # ── 1. Card data ──────────────────────────────────────────────────────
    card_data    = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids

    static_np    = card_data.feature_matrix(num_card_ids)
    static_jax   = jnp.array(static_np)   # frozen, never in params

    # Mark basic energy IDs
    energy_ids = [
        cid for cid in card_data.card_ids
        if "Energy" in card_data._cards[cid].get("stage", "")
    ]
    set_energy_ids(energy_ids)
    basic_pokemon_ids = [
        cid for cid in card_data.card_ids
        if card_data._cards[cid].get("stage", "").strip().lower()
        in ("basic pokémon", "basic pokemon")
    ]
    set_basic_pokemon_ids(basic_pokemon_ids)

    # ── Détection des cartes Ace Spec ─────────────────────────────────────
    # Only explicit card metadata is trustworthy here.  A card appearing once
    # in a reference deck is not evidence that it is an Ace Spec.
    ace_spec_set = set(card_data.ace_spec_ids)
    logger.info("[ace-spec] %d Ace Spec card(s) détecté(s) via cards.csv : %s", len(ace_spec_set), list(ace_spec_set))

    # Complétion via l'API de l'engine si disponible
    try:
        from env.cabt_api import all_card_data
        cg_cards = all_card_data()
        items = list(cg_cards.items() if isinstance(cg_cards, dict) else enumerate(cg_cards))

        def _card_is_ace(card) -> bool:
            for attr in ('is_ace_spec', 'isAceSpec', 'ace_spec', 'is_ace', 'isAce'):
                v = getattr(card, attr, None)
                if isinstance(v, bool) and v:
                    return True
            try:
                d = vars(card)
            except TypeError:
                d = {a: getattr(card, a, None) for a in dir(card) if not a.startswith('_')}
            for k, v in d.items():
                if 'ace' in k.lower() and isinstance(v, bool) and v:
                    return True
                if 'rule' in k.lower():
                    texts = [v] if isinstance(v, str) else (list(v) if isinstance(v, (list, tuple)) else [])
                    for t in texts:
                        if isinstance(t, str) and 'ace spec' in t.lower():
                            return True
            return False

        def _card_id(idx, card) -> int:
            for attr in ('id', 'card_id', 'cardId'):
                v = getattr(card, attr, None)
                if v is not None:
                    return int(v)
            return int(idx)

        engine_ace = [_card_id(i, c) for i, c in items if _card_is_ace(c)]
        ace_spec_set.update(engine_ace)
        logger.info("[ace-spec] Après fusion avec all_card_data() : %d Ace Spec cards", len(ace_spec_set))
    except Exception as e:
        logger.warning("[ace-spec] all_card_data() failed: %s", e)

    ace_spec_ids = sorted(list(ace_spec_set))
    if not ace_spec_ids:
        logger.warning("[ace-spec] Aucune carte Ace Spec détectée !")

    set_ace_spec_ids(ace_spec_ids)
    logger.info("Card pool: %d ids, %d energy ids, %d Basic Pokémon, %d ace spec ids",
                num_card_ids, len(energy_ids), len(basic_pokemon_ids), len(ace_spec_ids))

    # ── 2. Build modules ──────────────────────────────────────────────────
    network    = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    probes     = ProbeHeads(cfg=cfg.model)
    deck_net   = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)

    # ── 3. Dummy obs for init ─────────────────────────────────────────────
    dummy_obs = _make_dummy_obs(cfg.model)

    # ── 4. Init train states (avec support de reprise depuis HF Hub et checkpoint local) ──
    rng      = jax.random.PRNGKey(cfg.infra.seed)
    rng, r1, r2, r3 = jax.random.split(rng, 4)

    latest_path = Path(cfg.infra.checkpoint_dir) / "ckpt_latest.pkl"
    start_step = 0
    loaded_params = None
    loaded_opt_state = None
    loaded_deck_params = None
    loaded_deck_opt_state = None

    # 1. Tenter de charger le dernier modèle depuis HF Hub si activé
    if cfg.hf.enabled and cfg.hf.repo_id:
        try:
            from export.hub import load_from_hub
            logger.info("Vérification du modèle le plus récent sur HuggingFace Hub (%s)...", cfg.hf.repo_id)
            hf_mz, hf_dk, hf_cfg, hf_step = load_from_hub(cfg.hf.repo_id, cfg=cfg)
            if hf_mz:
                loaded_params = hf_mz
                loaded_deck_params = hf_dk
                start_step = hf_step
                logger.info("=== Modèle le plus récent chargé depuis HF Hub : %s (étape %d) ===", cfg.hf.repo_id, start_step)
        except Exception as e:
            logger.info("Aucun modèle HuggingFace Hub téléchargé ou échec de vérification (%s) : %s", cfg.hf.repo_id, e)

    # 2. Vérifier le checkpoint local et conserver le plus avancé (HF vs local)
    if latest_path.exists():
        try:
            import pickle
            with open(latest_path, "rb") as f:
                ckpt_data = pickle.load(f)
            local_step = ckpt_data.get("step", 0)
            if loaded_params is None or local_step > start_step:
                start_step = local_step
                loaded_params = ckpt_data["params"]
                loaded_opt_state = ckpt_data.get("opt_state", None)
                loaded_deck_params = ckpt_data.get("deck", {})
                loaded_deck_opt_state = ckpt_data.get("deck_opt_state", None)
                logger.info("=== Reprise depuis le checkpoint local plus récent : %s (étape %d) ===", latest_path, start_step)
            elif loaded_params is not None:
                logger.info("=== Checkpoint local (étape %d) <= HF Hub (étape %d). Conservation des poids HF Hub. ===", local_step, start_step)
        except Exception as e:
            logger.warning("Échec de la lecture du checkpoint local existant : %s.", e)

    if loaded_params is not None:
        state = create_muzero_train_state(network, probes, cfg, r1, dummy_obs)
        params_jax = jax.tree_util.tree_map(jax.device_put, loaded_params)
        params_jax = _merge_params(state.params, params_jax)
        state = state.replace(params=params_jax, step=jnp.array(start_step, dtype=jnp.int32))

        fresh_opt_state = state.tx.init(params_jax)
        if loaded_opt_state is not None:
            try:
                opt_state_jax = jax.tree_util.tree_map(jax.device_put, loaded_opt_state)
                opt_state_merged = _merge_params(fresh_opt_state, opt_state_jax)
                # Validation de compatibilité du PyTree avec les gradients du nouveau modèle
                jax.tree.map(lambda a, b: None, fresh_opt_state, opt_state_merged)
                state = state.replace(opt_state=opt_state_merged)
                logger.info("=== État de l'optimiseur MuZero restauré et fusionné avec succès à l'étape %d ===", start_step)
            except Exception as e:
                logger.warning(
                    "État d'optimiseur du checkpoint incompatible avec la nouvelle architecture (%s). Réinitialisation propre de l'optimiseur à l'étape %d.",
                    e, start_step
                )
                state = state.replace(opt_state=fresh_opt_state)
        else:
            state = state.replace(opt_state=fresh_opt_state)
    else:
        state = create_muzero_train_state(network, probes, cfg, r1, dummy_obs)

    if loaded_deck_params is not None:
        if isinstance(loaded_deck_params, dict) and "deck" in loaded_deck_params and len(loaded_deck_params) == 1:
            loaded_deck_params = loaded_deck_params["deck"]
        deck_params = jax.tree_util.tree_map(jax.device_put, loaded_deck_params)
        if loaded_deck_opt_state is not None:
            try:
                deck_opt_state = jax.tree_util.tree_map(jax.device_put, loaded_deck_opt_state)
                logger.info("=== État de l'optimiseur deck builder restauré avec succès ===")
            except Exception as e:
                logger.warning("Impossible de restaurer l'état de l'optimiseur deck builder : %s", e)
                deck_opt_state = optax.adam(cfg.train.deck_lr).init(deck_params)
        else:
            deck_opt_state = optax.adam(cfg.train.deck_lr).init(deck_params)
        deck_baseline = 0.0
    else:
        deck_params, deck_opt_state, deck_baseline = create_deck_train_state(
            deck_net, cfg, r2
        )
    deck_optimizer = optax.adam(cfg.train.deck_lr)

    # ── 5. Replicate onto 2 devices ───────────────────────────────────────
    devices = jax.devices()[:cfg.infra.num_devices]
    state   = flax.jax_utils.replicate(state, devices)

    train_step_fn = make_train_step(network, probes, cfg)

    # ── 6. Replay buffer ──────────────────────────────────────────────────
    buffer = PrioritizedReplayBuffer(cfg.train, cfg.model)

    # ── 7. Parallel Self-play seed games ──────────────────────────────────
    logger.info("Seeding replay buffer (min=%d)…", cfg.train.min_replay_size)
    rng, rng_selfplay = jax.random.split(rng)
    _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)

    from training.activity import tracker, format_h_status
    tracker.attach_buffer(buffer)  # lecture en direct de len(buffer) dans le heartbeat
    tracker.update(phase="Seeding Replay Buffer", buffer_size=len(buffer), deck_errors=0)

    _start_seeding = time.time()
    # A self-play wave has at most eight games.  More workers add no useful
    # parallelism and make the padded MCTS GPU batch needlessly larger.
    NUM_WORKERS = max(1, min(8, os.cpu_count() or 1))
    _deck_errors = 0

    # ── Spawn persistent worker processes pool ───────────────────────────
    import multiprocessing as mp
    from training.worker_bootstrap import run as _worker_bootstrap
    ctx = mp.get_context("spawn")
    pipes = []
    processes = []
    logger.info("[train] Spawning %d persistent worker processes...", NUM_WORKERS)
    for idx in range(NUM_WORKERS):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_worker_bootstrap, args=(child_conn, idx, cfg), daemon=True)
        p.start()
        pipes.append(parent_conn)
        processes.append(p)

    while len(buffer) < cfg.train.min_replay_size:
        tracker.update()
        rng, rng_sp = jax.random.split(rng)
        
        t_sp_start = time.time()
        hists, _, errs = run_parallel_self_play(
            num_games_to_play=8,
            num_workers=NUM_WORKERS,
            deck_net=deck_net,
            deck_params=deck_params,
            network=network,
            state_params=_params_cpu,
            cfg=cfg,
            rng=rng_sp,
            num_card_ids=num_card_ids,
            energy_ids=energy_ids,
            ace_spec_ids=ace_spec_ids,
            pipes=pipes,
            processes=processes,
            is_seeding=True,
            card_data=card_data,
        )
        tracker.self_play_times.append(time.time() - t_sp_start)
        
        _deck_errors += errs
        tracker.update(deck_errors=_deck_errors)
        for hist in hists:
            t_added = _add_history_to_buffer(hist, buffer, cfg)
            tracker.transitions_per_game_list.append(t_added)
        tracker.update(buffer_size=len(buffer))

    logger.info("Buffer seeded: %d entries in %.1fs", len(buffer), time.time() - _start_seeding)

    # ── 8. Main training loop ─────────────────────────────────────────────
    import collections
    global_step = 0
    t0 = time.time()
    total_transitions_since_last_push = 0
    rep_frozen = getattr(cfg.train, "freeze_representation_initially", True)
    loss_window = collections.deque(maxlen=getattr(cfg.train, "unfreeze_w", 500))

    tracker.update(start_step=start_step, rep_frozen=rep_frozen)

    if rep_frozen:
        logger.info(
            "[train] h(s) (RepresentationNetwork) gelé initialement. Dégel automatique activé "
            "(W=%d, epsilon=%.1f%%, S_min=%d steps nouveaux).",
            getattr(cfg.train, "unfreeze_w", 500),
            getattr(cfg.train, "unfreeze_epsilon", 0.01) * 100.0,
            getattr(cfg.train, "unfreeze_s_min", 2000),
        )

    for step in range(start_step, cfg.train.num_total_steps):
        global_step = step
        new_step = step - start_step

        # ── a. Self-play ──────────────────────────────────────────────────
        if step % cfg.train.self_play_interval == 0:
            _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
            
            # Enregistrer immédiatement le modèle entraîné courant pour la suite
            _save_checkpoint(state, deck_params, cfg, step, force_latest_only=True, deck_opt_state=deck_opt_state)
            
            rng, rng_sp = jax.random.split(rng)
            
            tracker.update(phase=f"Self-Play Step {step} (nouveau: {new_step})")

            n_games = cfg.train.games_per_self_play
            t_sp_start = time.time()
            hists, deck_updates, _ = run_parallel_self_play(
                num_games_to_play=n_games,
                num_workers=NUM_WORKERS,
                deck_net=deck_net,
                deck_params=deck_params,
                network=network,
                state_params=_params_cpu,
                cfg=cfg,
                rng=rng_sp,
                num_card_ids=num_card_ids,
                energy_ids=energy_ids,
                ace_spec_ids=ace_spec_ids,
                pipes=pipes,
                processes=processes,
                is_seeding=False,
                card_data=card_data,
            )
            tracker.self_play_times.append(time.time() - t_sp_start)

            added_transitions = 0
            for hist in hists:
                t_added = _add_history_to_buffer(hist, buffer, cfg)
                added_transitions += t_added
                tracker.transitions_per_game_list.append(t_added)
            
            total_transitions_since_last_push += added_transitions

            for deck_ids, rew in deck_updates:
                deck_params, deck_opt_state, deck_baseline, d_loss = \
                    deck_reinforce_update(
                        deck_net, deck_params, deck_opt_state, deck_optimizer,
                        deck_ids, rew, deck_baseline,
                        entropy_coef=cfg.train.deck_entropy_coef,
                        baseline_ema=cfg.train.deck_baseline_ema,
                    )

        # ── b. Sample + train ─────────────────────────────────────────────
        entries, indices, is_w = buffer.sample(cfg.train.batch_size)
        batch_cpu = collate_batch(
            entries, is_w, cfg.train.num_unroll_steps, cfg.model.max_actions
        )
        batch_sharded = shard_batch(batch_cpu, cfg.infra.num_devices)

        # Clé PRNG pour Reanalyze — répliquée puis différenciée par device via fold_in
        rng, rng_step = jax.random.split(rng)
        rng_step_replicated = jnp.broadcast_to(
            rng_step[None], (cfg.infra.num_devices,) + rng_step.shape
        )
        rep_frozen_replicated = jnp.broadcast_to(
            jnp.array(rep_frozen, dtype=jnp.bool_), (cfg.infra.num_devices,)
        )

        t_step_start = time.time()
        state, metrics, td_errs = train_step_fn(
            state, batch_sharded, rng_step_replicated, rep_frozen_replicated
        )
        tracker.train_step_times.append(time.time() - t_step_start)

        if float(metrics["update_is_finite"][0]) == 0.0:
            logger.error(
                "[train] Non-finite loss or gradient at step %d; update and priority refresh skipped.",
                step,
            )
            continue

        # ── Contrôleur de dégel automatique (Split-Window Plateau Detection) ──
        loss_val = float(metrics["loss_total"][0])
        loss_window.append(loss_val)

        w_size = getattr(cfg.train, "unfreeze_w", 500)
        s_min = getattr(cfg.train, "unfreeze_s_min", 2000)
        eps = getattr(cfg.train, "unfreeze_epsilon", 0.01)

        current_gain = None
        if rep_frozen and len(loss_window) == w_size:
            w_list = list(loss_window)
            half_w = w_size // 2
            m_ancien = float(np.mean(w_list[:half_w]))
            m_recent = float(np.mean(w_list[half_w:]))
            current_gain = (m_ancien - m_recent) / (m_ancien + 1e-8)

            if new_step >= s_min and current_gain < eps:
                rep_frozen = False
                logger.info("=========================================================================")
                logger.info(
                    "[train] Plateau de loss détecté (M_ancien=%.4f, M_récent=%.4f, Gain=%.2f%% < %.1f%%) à l'étape %d (nouveau: %d).",
                    m_ancien, m_recent, current_gain * 100.0, eps * 100.0, step, new_step
                )
                logger.info("[train] Dégel automatique de h(s) (RepresentationNetwork) ! Début du fine-tuning global.")
                logger.info("=========================================================================")

        h_status = format_h_status(
            step=step,
            start_step=start_step,
            rep_frozen=rep_frozen,
            loss_window_len=len(loss_window),
            w_size=w_size,
            s_min=s_min,
            eps=eps,
            current_gain=current_gain,
            avg_step_time=tracker.avg_train_step_time,
        )

        tracker.update(step=step, rep_frozen=rep_frozen, h_status=h_status)

        # Update priorities (TD error basé sur les fresh targets Reanalyze)
        td_err_flat = np.array(unshard(td_errs))
        buffer.update_priorities(indices, td_err_flat)

        # ── Priority Refresh allégé (tous les N steps) ────────────────────
        if step % cfg.train.priority_refresh_every == 0 and len(buffer) > 0:
            _params_cpu_mz = jax.tree_util.tree_map(lambda x: x[0], state.params)["muzero"]
            _refresh_buffer_priorities(buffer, network, _params_cpu_mz, cfg)

        # ── c. Logging ────────────────────────────────────────────────────
        if step % 100 == 0:
            m = {k: float(v[0]) for k, v in metrics.items()
                 if not hasattr(v, '__len__') or v.ndim == 0 or
                 (hasattr(v, 'shape') and v.shape == (cfg.infra.num_devices,))}
            elapsed = time.time() - t0
            prb_acc_mean = float(np.mean(metrics.get("probe_acc_per_task", [np.zeros(11)])[0])) * 100.0
            logger.info(
                "step=%d (nouveau:%d)  loss=%.4f  pol=%.4f  val=%.4f  rew=%.4f  "
                "prb=%.4f (acc=%.1f%%)  buf=%d  h(s)=%s  %.1fs",
                step,
                new_step,
                float(metrics["loss_total"][0]),
                float(metrics["loss_policy"][0]),
                float(metrics["loss_value"][0]),
                float(metrics["loss_reward"][0]),
                float(metrics["loss_probes"][0]),
                prb_acc_mean,
                len(buffer),
                h_status,
                elapsed,
            )
            # Detailed per-probe loss and accuracy breakdown
            _log_probe_metrics(metrics, cfg)

            _log_wandb_full_metrics(
                metrics, buffer, batch_cpu, td_err_flat, step, elapsed, rep_frozen, cfg
            )

        # ── d. Checkpoint & HF push ───────────────────────────────────────
        if step % cfg.train.checkpoint_every == 0 and step > 0:
            _save_checkpoint(state, deck_params, cfg, step, deck_opt_state=deck_opt_state)
            
            if cfg.hf.enabled:
                _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
                logger.info("[hf-push] Publication synchrone du snapshot %d...", step)
                if not push_to_hub(_params_cpu, deck_params, cfg, step):
                    logger.error("[hf-push] Snapshot %d non confirmé sur HF.", step)

    # Final checkpoint
    _save_checkpoint(state, deck_params, cfg, global_step, deck_opt_state=deck_opt_state)
    if cfg.hf.enabled:
        _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
        logger.info("[hf-push] Lancement du push final synchrone vers HuggingFace...")
        push_to_hub(_params_cpu, deck_params, cfg, global_step)

    # Clean up persistent workers pool
    logger.info("[train] Nettoyage du pool de workers persistants...")
    for pipe in pipes:
        try:
            pipe.send(None)
        except Exception:
            pass
    for p in processes:
        p.join(timeout=1.0)
        if p.is_alive():
            p.terminate()

    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _refresh_buffer_priorities(
    buffer: PrioritizedReplayBuffer,
    network,
    mz_params: dict,
    cfg: Config,
) -> None:
    """
    Recalcule les priorités PER sur un échantillon aléatoire du buffer.

    Utilise uniquement h+f (représentation + prédiction), sans MCTS.  Le
    Transformer a un coût mémoire quadratique dans la taille du batch : le
    calcul est donc plafonné et envoyé par micro-batches pour ne pas provoquer
    d'OOM quand le replay buffer grandit.
    La priorité mise à jour = |v_pred - v_target_stockée|.
    """
    fraction = cfg.train.priority_refresh_fraction
    n = min(
        max(1, int(len(buffer) * fraction)),
        cfg.train.priority_refresh_max_entries,
    )
    indices = np.random.choice(len(buffer), size=n, replace=False)

    indexed_entries = [
        (int(i), buffer._entries[int(i)])
        for i in indices
        if buffer._entries[int(i)] is not None
    ]
    if not indexed_entries:
        return

    batch_size = cfg.train.priority_refresh_batch_size
    for start in range(0, len(indexed_entries), batch_size):
        chunk = indexed_entries[start:start + batch_size]
        chunk_indices = np.array([i for i, _ in chunk], dtype=np.int32)
        entries = [entry for _, entry in chunk]
        obs_keys = entries[0].obs_seq[0].keys()
        obs_batch = {
            k: jnp.array(np.stack([entry.obs_seq[0][k] for entry in entries]))
            for k in obs_keys
        }

        z = network.apply(mz_params, obs_batch, method=network.represent)
        _, v_pred = network.apply(mz_params, z, method=network.predict)
        v_target = np.array([entry.target_val[0] for entry in entries], dtype=np.float32)
        buffer.update_priorities(chunk_indices, np.abs(np.array(v_pred) - v_target))


def _add_history_to_buffer(hist: GameHistory, buffer: PrioritizedReplayBuffer, cfg: Config) -> int:

    if len(hist) == 0 or hist.returns is None:
        return 0
    probe_tgts = np.stack([
        extract_probe_targets(raw, hist.player_idx)
        for raw in hist.raw_states
    ] + [np.full(11, -1, dtype=np.int32)])   # +1 for final state

    buffer.add_game(
        obs_list    = hist.observations,
        actions     = hist.actions,
        rewards     = hist.rewards,
        search_pols = hist.search_pols,
        returns     = hist.returns,
        probe_tgts  = probe_tgts[:len(hist.observations) + 1],
    )
    return len(hist.actions)


def _make_dummy_obs(cfg) -> dict:
    """Crée un obs numpy à zéros pour l'init des paramètres."""
    import numpy as np
    from env.encoding import GLOBAL_FEAT_DIM, POKEMON_FEAT_DIM, OPTION_FEAT_DIM
    return {
        "global_feat":      np.zeros(GLOBAL_FEAT_DIM,  dtype=np.float32),
        "my_active_id":     np.zeros(1,                 dtype=np.int32),
        "my_active_feat":   np.zeros((1, POKEMON_FEAT_DIM), dtype=np.float32),
        "my_bench_ids":     np.zeros(cfg.max_bench_size, dtype=np.int32),
        "my_bench_feat":    np.zeros((cfg.max_bench_size, POKEMON_FEAT_DIM), dtype=np.float32),
        "my_bench_mask":    np.zeros(cfg.max_bench_size, dtype=bool),
        "my_hand_ids":      np.zeros(cfg.max_hand_size,  dtype=np.int32),
        "my_hand_mask":     np.zeros(cfg.max_hand_size,  dtype=bool),
        "my_discard_ids":   np.zeros(cfg.max_discard_size, dtype=np.int32),
        "my_discard_mask":  np.zeros(cfg.max_discard_size, dtype=bool),
        "my_prize_ids":     np.zeros(cfg.max_prize_size, dtype=np.int32),
        "opp_active_id":    np.zeros(1,                  dtype=np.int32),
        "opp_active_feat":  np.zeros((1, POKEMON_FEAT_DIM), dtype=np.float32),
        "opp_bench_ids":    np.zeros(cfg.max_bench_size, dtype=np.int32),
        "opp_bench_feat":   np.zeros((cfg.max_bench_size, POKEMON_FEAT_DIM), dtype=np.float32),
        "opp_bench_mask":   np.zeros(cfg.max_bench_size, dtype=bool),
        "opp_discard_ids":  np.zeros(cfg.max_discard_size, dtype=np.int32),
        "opp_discard_mask": np.zeros(cfg.max_discard_size, dtype=bool),
        "opp_prize_ids":    np.zeros(cfg.max_prize_size, dtype=np.int32),
        "opp_hand_ids":     np.zeros(cfg.max_hand_size,  dtype=np.int32),
        "opp_hand_mask":    np.zeros(cfg.max_hand_size,  dtype=bool),
        "option_ids":       np.zeros(cfg.max_actions,    dtype=np.int32),
        "option_feat":      np.zeros((cfg.max_actions, OPTION_FEAT_DIM), dtype=np.float32),
        "option_mask":      np.zeros(cfg.max_actions,    dtype=bool),
    }


def _save_checkpoint(state, deck_params, cfg: Config, step: int, force_latest_only: bool = False, deck_opt_state=None):
    import pickle
    params_cpu = jax.tree_util.tree_map(lambda x: np.array(x[0]), state.params)
    opt_state_cpu = jax.tree_util.tree_map(
        lambda x: np.array(x[0]) if hasattr(x, "__getitem__") and hasattr(x, "shape") and len(x.shape) > 0 else x,
        state.opt_state
    )
    deck_opt_state_cpu = None
    if deck_opt_state is not None:
        deck_opt_state_cpu = jax.tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, (np.ndarray, jax.Array)) else x,
            deck_opt_state
        )

    ckpt_data = {
        "step": step,
        "params": params_cpu,
        "opt_state": opt_state_cpu,
        "deck": deck_params,
        "deck_opt_state": deck_opt_state_cpu,
    }

    if not force_latest_only:
        path = Path(cfg.infra.checkpoint_dir) / f"ckpt_{step:07d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(ckpt_data, f)
        logger.info("Checkpoint saved: %s", path)
        
    latest_path = Path(cfg.infra.checkpoint_dir) / "ckpt_latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump(ckpt_data, f)
    logger.info("Latest checkpoint updated: %s", latest_path)


def _log_probe_metrics(metrics: dict, cfg: Config):
    try:
        per_probe = np.array(metrics["probe_per_task"][0])
        acc_per_probe = np.array(metrics.get("probe_acc_per_task", [np.zeros_like(per_probe)])[0])
        from interpretability.probes import PROBE_DEFS
        logger.info("  ── Probe Accuracy Breakdown ──")
        for i, val in enumerate(per_probe):
            acc = float(acc_per_probe[i]) if i < len(acc_per_probe) else 0.0
            name = PROBE_DEFS[i]["name"] if i < len(PROBE_DEFS) else f"probe_{i}"
            logger.info("    [%-30s] loss=%.4f  accuracy=%5.1f%%", name, float(val), acc * 100.0)
    except Exception as e:
        logger.debug("Failed to log probe metrics: %s", e)
