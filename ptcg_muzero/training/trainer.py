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
from search.ismcts import add_exploration_noise, ismcts_action
from training.loss import collate_batch, muzero_loss
from training.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


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
    # NOTE : deck.csv est un exemple officiel fourni par la compétition.
    # Il est utilisé ICI UNIQUEMENT pour identifier quels card IDs sont
    # des cartes Ace Spec (ceux qui n'apparaissent qu'une seule fois).
    # Il n'est JAMAIS utilisé comme deck de jeu en self-play.
    ace_spec_ids: list = []

    ref_path = cfg.infra.reference_deck_csv
    try:
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                ref_ids = [int(line.strip()) for line in f if line.strip().isdigit()]

            from collections import Counter
            counts = Counter(ref_ids)
            energy_set = set(energy_ids)

            # Ace Spec = apparait exactement 1 fois ET n'est pas une énergie de base
            ace_spec_ids = [
                cid for cid, cnt in counts.items()
                if cnt == 1 and cid not in energy_set
            ]
            logger.info("[ace-spec] %d Ace Spec card(s) détecté(s) via reference deck.csv : %s",
                        len(ace_spec_ids), ace_spec_ids)
        else:
            logger.warning("[ace-spec] reference_deck_csv introuvable : %s", ref_path)
    except Exception as e:
        logger.warning("[ace-spec] Lecture du reference deck échouée : %s", e)

    # Fallback: engine all_card_data() inspection
    if not ace_spec_ids:
        try:
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

            ace_spec_ids = [_card_id(i, c) for i, c in items if _card_is_ace(c)]
            logger.info("[ace-spec] Engine fallback: %d Ace Spec cards", len(ace_spec_ids))
        except Exception as e:
            logger.warning("[ace-spec] all_card_data() failed: %s", e)

    # Fallback: CSV Rule column
    if not ace_spec_ids:
        ace_spec_ids = card_data.ace_spec_ids
        logger.info("[ace-spec] CSV fallback: %d Ace Spec cards", len(ace_spec_ids))

    if not ace_spec_ids:
        logger.warning("[ace-spec] No Ace Spec cards detected — DeckErrors will be retried.")

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

    from concurrent.futures import ThreadPoolExecutor
    _deck_errors = 0
    _start_seeding = time.time()
    
    # Nombre de workers concurrents (4 est un bon équilibre sur Kaggle)
    NUM_WORKERS = 4

    def _play_single_game_worker(worker_id):
        nonlocal _deck_errors, rng
        # Split local keys safely
        rng, rng_sp, rng_d0, rng_d1 = jax.random.split(rng, 4)
        deck_logits_np, _ = deck_net.apply(deck_params)
        deck0, _ = sample_deck(deck_logits_np[0], rng_d0, num_card_ids, energy_ids,
                               ace_spec_ids=ace_spec_ids)
        deck1, _ = sample_deck(deck_logits_np[0], rng_d1, num_card_ids, energy_ids,
                               ace_spec_ids=ace_spec_ids)

        agent = make_agent_fn(network, _params_cpu, cfg, rng_sp)
        
        logger.info(f"[seeding-worker-{worker_id}] Lancement d'une partie de self-play...")
        game_start_t = time.time()
        try:
            h0, h1 = run_self_play_game(agent, deck0, deck1, cfg,
                                        np.random.default_rng())
            logger.info(
                f"[seeding-worker-{worker_id}] Partie terminée en {time.time() - game_start_t:.1f}s. "
                f"J0 steps={len(h0.action_seq)}, J1 steps={len(h1.action_seq)}"
            )
            return h0, h1
        except DeckError as e:
            _deck_errors += 1
            tracker.update(deck_errors=_deck_errors)
            logger.warning(f"[seeding-worker-{worker_id}] DeckError #{_deck_errors}: {e} — resampling deck")
            return None

    # Lancement parallèle pour le seeding
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while len(buffer) < cfg.train.min_replay_size:
            tracker.update()
            
            # Soumettre un lot de tâches concurrentes
            futures = [executor.submit(_play_single_game_worker, idx) for idx in range(NUM_WORKERS)]
            
            for future in futures:
                res = future.result()
                if res is not None:
                    h0, h1 = res
                    for hist in (h0, h1):
                        _add_history_to_buffer(hist, buffer, cfg)
            
            tracker.update(buffer_size=len(buffer))

    logger.info("Buffer seeded: %d entries in %.1fs", len(buffer), time.time() - _start_seeding)

    # ── 8. Main training loop ─────────────────────────────────────────────
    global_step = 0
    t0 = time.time()

    for step in range(cfg.train.num_total_steps):
        global_step = step

        # ── a. Self-play ──────────────────────────────────────────────────
        if step % cfg.train.self_play_interval == 0:
            _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
            rng, rng_sp = jax.random.split(rng)
            deck_logits_np, _ = deck_net.apply(deck_params)
            
            tracker.update(phase=f"Self-Play Step {step}")

            # Worker pour la boucle d'entraînement principale
            def _play_train_worker(worker_idx):
                nonlocal rng
                rng, rng_d0, rng_d1, rng_ag = jax.random.split(rng, 4)
                deck0, ids0 = sample_deck(
                    deck_logits_np[0], rng_d0, num_card_ids, energy_ids,
                    ace_spec_ids=ace_spec_ids,
                )
                deck1, ids1 = sample_deck(
                    deck_logits_np[0], rng_d1, num_card_ids, energy_ids,
                    ace_spec_ids=ace_spec_ids,
                )
                agent = make_agent_fn(network, _params_cpu, cfg, rng_ag)
                try:
                    h0, h1 = run_self_play_game(
                        agent, deck0, deck1, cfg, np.random.default_rng()
                    )
                    return h0, h1, ids0, ids1
                except DeckError as e:
                    logger.warning("[train] DeckError: %s — skipping game", e)
                    return None

            # Exécuter les games_per_self_play en parallèle
            n_games = cfg.train.games_per_self_play
            with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, n_games)) as executor:
                futures = [executor.submit(_play_train_worker, idx) for idx in range(n_games)]
                
                for future in futures:
                    res = future.result()
                    if res is not None:
                        h0, h1, ids0, ids1 = res
                        for hist in (h0, h1):
                            _add_history_to_buffer(hist, buffer, cfg)

                        # Deck builder REINFORCE update
                        reward_0 = float(h0.game_won or False) * 2 - 1
                        reward_1 = float(h1.game_won or False) * 2 - 1
                        for deck_ids, rew in [(ids0, reward_0), (ids1, reward_1)]:
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
                and step % cfg.hf.push_every_n_steps == 0
                and step > 0):
            _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
            push_to_hub(_params_cpu, deck_params, cfg, step)

    # Final checkpoint
    _save_checkpoint(state, deck_params, cfg, global_step)
    if cfg.hf.enabled:
        _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
        push_to_hub(_params_cpu, deck_params, cfg, global_step)
    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _add_history_to_buffer(hist: GameHistory, buffer: PrioritizedReplayBuffer, cfg: Config):
    if len(hist) == 0 or hist.returns is None:
        return
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
