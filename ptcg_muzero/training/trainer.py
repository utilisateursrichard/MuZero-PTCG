"""
ptcg_muzero/training/trainer.py
=================================
Boucle d'entraînement principale.

Architecture dual-GPU
---------------------
* Tous les paramètres sont répliqués sur les 2 devices via
  ``jax.device_put_replicated``.
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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
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
    set_energy_ids,
)
from models.networks import MuZeroNetwork
from search.ismcts import add_exploration_noise, ismcts_action, ismcts_action_batched
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
    """

    def _step(
        state: train_state.TrainState,
        batch: dict,
    ) -> Tuple[train_state.TrainState, dict, jnp.ndarray]:
        def loss_fn(params):
            loss, metrics, td_err = muzero_loss(
                params, network, probe_heads, batch,
                cfg.train, cfg.model,
            )
            return loss, (metrics, td_err)

        (loss, (metrics, td_err)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        # Synchronise gradients across devices
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss  = jax.lax.pmean(loss,  axis_name="devices")

        new_state = state.apply_gradients(grads=grads)
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
    is_seeding: bool = False,
    min_batch_threshold: int = 4,   # attendre au moins N workers avant d'envoyer au GPU
    card_data = None,
):
    if is_seeding:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.search.num_simulations = 15
        cfg.search.num_belief_samples = 1
        logger.info("[self-play] Mode seeding active: acceleration MCTS (sims=15, belief_samples=1)")

    import multiprocessing as mp
    from training.worker_bootstrap import run as _worker_bootstrap
    import numpy as np
    from training.activity import tracker

    # Pré-transférer les params sur les GPUs UNE SEULE FOIS (évite CPU→GPU à chaque inférence)
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        gpu_devices = devices  # fallback CPU
    # On répartit les inférences alternativement sur tous les GPUs disponibles
    logger.info("[self-play] GPU devices pour MCTS : %s", gpu_devices)
    _gpu_params = [
        jax.device_put(state_params, dev) for dev in gpu_devices
    ]
    _inference_device_idx = [0]  # compteur pour alterner les GPUs

    ctx = mp.get_context("spawn")
    pipes = []
    processes = []
    
    for idx in range(num_workers):
        parent_conn, child_conn = ctx.Pipe()
        # bootstrap : pose CUDA_VISIBLE_DEVICES="" AVANT tout import JAX dans le fils
        p = ctx.Process(target=_worker_bootstrap, args=(child_conn, idx, cfg), daemon=True)
        p.start()
        pipes.append(parent_conn)
        processes.append(p)

    games_started = 0
    games_completed = 0
    completed_histories = []
    deck_builder_updates = []
    deck_errors_count = 0

    from models.deck_builder import sample_deck
    deck_logits, _ = deck_net.apply(deck_params)

    pipe_meta = [{} for _ in range(num_workers)]
    worker_steps = [0] * num_workers

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
        try:
            pipes[pipe_idx].send({
                "cmd": "start",
                "deck0": list(deck0),
                "deck1": list(deck1)
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

    while games_completed < num_games_to_play:
        need_action_indices = []
        batched_encs_list = []
        option_masks_list = []
        
        active_pipes_list = [pipes[i] for i in range(num_workers) if pipe_meta[i].get("game_active")]
        if not active_pipes_list:
            break

        # Attendre de façon non-bloquante/bloquante efficace qu'au moins un worker soit prêt
        ready_pipes = list(mp.connection.wait(active_pipes_list))

        # Ajouter immédiatement tous les autres workers qui ont également fini leur étape (sans attendre)
        for p in active_pipes_list:
            if p not in ready_pipes and p.poll():
                ready_pipes.append(p)

        for pipe in ready_pipes:
            pipe_idx = pipes.index(pipe)
            if not pipe.poll():
                continue

            try:
                msg = pipe.recv()
            except Exception as e:
                logger.error(f"Failed to read from worker {pipe_idx}: {e}")
                pipe_meta[pipe_idx]["game_active"] = False
                games_completed += 1
                continue

            status = msg.get("status")
            if status == "need_action":
                worker_steps[pipe_idx] = msg.get("step_count", 0)
                tracker.update(current_game_steps=int(np.max(worker_steps)))
            else:
                tracker.update()


            if status == "need_action":
                need_action_indices.append(pipe_idx)
                batched_encs_list.append(msg["batched_enc"])
                option_masks_list.append(msg["option_mask"])

            elif status == "deck_error":
                deck_errors_count += 1
                logger.warning(f"[coordinateur] Erreur de deck sur le worker {pipe_idx} : {msg.get('error')}")
                
                # Récupération et analyse des cartes sélectionnées pour ce worker (diagnostic)
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

                # Relancer simplement une nouvelle partie sur ce worker
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

        if need_action_indices:
            # Alterner entre les GPUs disponibles pour utiliser tous les GPUs
            dev_idx = _inference_device_idx[0] % len(_gpu_params)
            _inference_device_idx[0] += 1
            gpu_params = _gpu_params[dev_idx]

            # Rembourrage (padding) à taille statique `num_workers` pour éviter les recompilations JAX JIT
            n_actual = len(need_action_indices)
            padded_encs = list(batched_encs_list)
            padded_masks = list(option_masks_list)
            while len(padded_encs) < num_workers:
                padded_encs.append(batched_encs_list[-1])
                padded_masks.append(option_masks_list[-1])

            keys = padded_encs[0].keys()
            batched_enc = {}
            for k in keys:
                arr = np.stack([x[k] for x in padded_encs], axis=0)
                # Transférer les observations directement sur le GPU cible
                batched_enc[k] = jax.device_put(arr, gpu_devices[dev_idx])

            option_masks = np.stack(padded_masks, axis=0)

            rng, rng_act = jax.random.split(rng)
            try:
                best_actions, avg_policies, avg_values = ismcts_action_batched(
                    network, gpu_params, batched_enc, option_masks, rng_act, cfg
                )
            except Exception as e:
                logger.error("[MCTS] Erreur ismcts_action_batched (batch=%d) : %s", n_actual, e, exc_info=True)
                # Fallback : action aléatoire parmi les actions valides (en utilisant les masques réels)
                best_actions = np.array([
                    int(np.random.choice(np.where(option_masks_list[i])[0]))
                    for i in range(n_actual)
                ])
                avg_policies = np.zeros((n_actual, int(cfg.model.max_actions)), dtype=np.float32)
                avg_values   = np.zeros(n_actual, dtype=np.float32)

            for idx_in_batch, pipe_idx in enumerate(need_action_indices):
                pipes[pipe_idx].send({
                    "action_indices": [int(best_actions[idx_in_batch])],
                    "search_pol": np.array(avg_policies[idx_in_batch]),
                    "search_val": float(avg_values[idx_in_batch])
                })

    for pipe in pipes:
        try:
            pipe.send(None)
        except Exception:
            pass
    for p in processes:
        p.join(timeout=1.0)
        if p.is_alive():
            p.terminate()

    return completed_histories, deck_builder_updates, deck_errors_count


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: Config) -> None:
    """Point d'entrée principal d'entraînement."""
    logger.info("=== PTCG MuZero training start ===")
    logger.info("Devices: %s", jax.devices())

    Path(cfg.infra.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.infra.log_dir).mkdir(parents=True, exist_ok=True)

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

    # ── Détection des cartes Ace Spec via deck.csv de référence ──────────
    # Détection exhaustive des cartes Ace Spec en combinant toutes les sources (CSV complet, deck de référence, moteur)
    ace_spec_set = set(card_data.ace_spec_ids)
    logger.info("[ace-spec] %d Ace Spec card(s) détecté(s) via cards.csv : %s", len(ace_spec_set), list(ace_spec_set))

    ref_path = cfg.infra.reference_deck_csv
    try:
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                ref_ids = [int(line.strip()) for line in f if line.strip().isdigit()]

            from collections import Counter
            counts = Counter(ref_ids)
            energy_set = set(energy_ids)

            # Ace Spec = apparait exactement 1 fois ET n'est pas une énergie de base
            ref_ace = [
                cid for cid, cnt in counts.items()
                if cnt == 1 and cid not in energy_set
            ]
            ace_spec_set.update(ref_ace)
            logger.info("[ace-spec] Après fusion avec reference deck.csv : %d Ace Spec cards", len(ace_spec_set))
        else:
            logger.warning("[ace-spec] reference_deck_csv introuvable : %s", ref_path)
    except Exception as e:
        logger.warning("[ace-spec] Lecture du reference deck échouée : %s", e)

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
    logger.info("Card pool: %d ids, %d energy ids, %d ace spec ids",
                num_card_ids, len(energy_ids), len(ace_spec_ids))

    # ── 2. Build modules ──────────────────────────────────────────────────
    network    = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    probes     = ProbeHeads(cfg=cfg.model)
    deck_net   = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)

    # ── 3. Dummy obs for init ─────────────────────────────────────────────
    dummy_obs = _make_dummy_obs(cfg.model)

    # ── 4. Init train states ──────────────────────────────────────────────
    rng      = jax.random.PRNGKey(cfg.infra.seed)
    rng, r1, r2, r3 = jax.random.split(rng, 4)

    state = create_muzero_train_state(network, probes, cfg, r1, dummy_obs)

    deck_params, deck_opt_state, deck_baseline = create_deck_train_state(
        deck_net, cfg, r2
    )
    deck_optimizer = optax.adam(cfg.train.deck_lr)

    # ── 5. Replicate onto 2 devices ───────────────────────────────────────
    devices = jax.devices()[:cfg.infra.num_devices]
    state   = jax.device_put_replicated(state, devices)

    train_step_fn = make_train_step(network, probes, cfg)

    # ── 6. Replay buffer ──────────────────────────────────────────────────
    buffer = PrioritizedReplayBuffer(cfg.train, cfg.model)

    # ── 7. Parallel Self-play seed games ──────────────────────────────────
    logger.info("Seeding replay buffer (min=%d)…", cfg.train.min_replay_size)
    rng, rng_selfplay = jax.random.split(rng)
    _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)

    from training.activity import tracker
    tracker.update(phase="Seeding Replay Buffer", buffer_size=len(buffer), deck_errors=0)

    _start_seeding = time.time()
    NUM_WORKERS = 8   # 8 workers CPU-only (bootstrap garantit pas de GPU dans les fils)
    _deck_errors = 0

    while len(buffer) < cfg.train.min_replay_size:
        tracker.update()
        rng, rng_sp = jax.random.split(rng)
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
            is_seeding=True,
            card_data=card_data,
        )
        _deck_errors += errs
        tracker.update(deck_errors=_deck_errors)
        for hist in hists:
            _add_history_to_buffer(hist, buffer, cfg)
        tracker.update(buffer_size=len(buffer))

    logger.info("Buffer seeded: %d entries in %.1fs", len(buffer), time.time() - _start_seeding)

    # ── 8. Main training loop ─────────────────────────────────────────────
    global_step = 0
    t0 = time.time()
    total_transitions_since_last_push = 0

    for step in range(cfg.train.num_total_steps):
        global_step = step

        # ── a. Self-play ──────────────────────────────────────────────────
        if step % cfg.train.self_play_interval == 0:
            _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
            rng, rng_sp = jax.random.split(rng)
            
            tracker.update(phase=f"Self-Play Step {step}")

            n_games = cfg.train.games_per_self_play
            hists, deck_updates, _ = run_parallel_self_play(
                num_games_to_play=n_games,
                num_workers=min(NUM_WORKERS, max(n_games, 16)),
                deck_net=deck_net,
                deck_params=deck_params,
                network=network,
                state_params=_params_cpu,
                cfg=cfg,
                rng=rng_sp,
                num_card_ids=num_card_ids,
                energy_ids=energy_ids,
                ace_spec_ids=ace_spec_ids,
                is_seeding=False,
                card_data=card_data,
            )

            added_transitions = 0
            for hist in hists:
                added_transitions += _add_history_to_buffer(hist, buffer, cfg)
            
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

        state, metrics, td_errs = train_step_fn(state, batch_sharded)

        # Update priorities
        td_err_flat = np.array(unshard(td_errs))
        buffer.update_priorities(indices, td_err_flat)

        # ── c. Logging ────────────────────────────────────────────────────
        if step % 100 == 0:
            m = {k: float(v[0]) for k, v in metrics.items()
                 if not hasattr(v, '__len__') or v.ndim == 0 or
                 (hasattr(v, 'shape') and v.shape == (cfg.infra.num_devices,))}
            elapsed = time.time() - t0
            logger.info(
                "step=%d  loss=%.4f  pol=%.4f  val=%.4f  rew=%.4f  "
                "prb=%.4f  buf=%d  %.1fs",
                step,
                float(metrics["loss_total"][0]),
                float(metrics["loss_policy"][0]),
                float(metrics["loss_value"][0]),
                float(metrics["loss_reward"][0]),
                float(metrics["loss_probes"][0]),
                len(buffer),
                elapsed,
            )
            # Probe accuracy
            _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
            _log_probe_metrics(metrics, cfg)

        # ── d. Checkpoint ─────────────────────────────────────────────────
        if step % cfg.train.checkpoint_every == 0 and step > 0:
            _save_checkpoint(state, deck_params, cfg, step)

        # ── e. HF push ────────────────────────────────────────────────────
        if (cfg.hf.enabled
                and total_transitions_since_last_push >= cfg.hf.push_every_n_transitions
                and step > 0):
            total_transitions_since_last_push -= cfg.hf.push_every_n_transitions
            _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
            
            global _active_push_thread
            if _active_push_thread is not None and _active_push_thread.is_alive():
                logger.warning(
                    "[hf-push] Un push HuggingFace précédent est encore en cours. "
                    "Saut du push pour l'étape %d pour éviter de bloquer.", step
                )
            else:
                logger.info("[hf-push] Lancement du push asynchrone vers HuggingFace pour l'étape %d...", step)
                _active_push_thread = threading.Thread(
                    target=push_to_hub,
                    args=(_params_cpu, deck_params, cfg, step),
                    daemon=True
                )
                _active_push_thread.start()

    # Final checkpoint
    _save_checkpoint(state, deck_params, cfg, global_step)
    if cfg.hf.enabled:
        _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
        logger.info("[hf-push] Lancement du push final synchrone vers HuggingFace...")
        push_to_hub(_params_cpu, deck_params, cfg, global_step)
    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _add_history_to_buffer(hist: GameHistory, buffer: PrioritizedReplayBuffer, cfg: Config) -> int:
    if len(hist) == 0 or hist.returns is None:
        return 0
    probe_tgts = np.stack([
        extract_probe_targets(raw, hist.player_idx)
        for raw in hist.raw_states
    ] + [np.full(5, -1, dtype=np.int32)])   # +1 for final state

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


def _save_checkpoint(state, deck_params, cfg: Config, step: int):
    import pickle
    path = Path(cfg.infra.checkpoint_dir) / f"ckpt_{step:07d}.pkl"
    params_cpu = jax.tree_util.tree_map(lambda x: np.array(x[0]), state.params)
    with open(path, "wb") as f:
        pickle.dump({"step": step, "params": params_cpu, "deck": deck_params}, f)
    logger.info("Checkpoint saved: %s", path)


def _log_probe_metrics(metrics: dict, cfg: Config):
    try:
        per_probe = np.array(metrics["probe_per_task"][0])
        for i, val in enumerate(per_probe):
            from interpretability.probes import PROBE_DEFS
            logger.info("  probe[%s]=%.4f", PROBE_DEFS[i]["name"], float(val))
    except Exception:
        pass
