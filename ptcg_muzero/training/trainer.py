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
from flax import struct
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
    set_card_names,
    set_energy_ids,
)
from models.networks import MuZeroNetwork
from search.ismcts import ismcts_action, ismcts_action_batched, reanalyze_root
from training.activity import tracker
from training.loss import collate_batch, muzero_loss
from training.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)

_active_push_thread = None


# ─────────────────────────────────────────────────────────────────────────────
# TrainState étendu (MuZero + Target Network EMA)
# ─────────────────────────────────────────────────────────────────────────────
@struct.dataclass
class MuZeroTrainState(train_state.TrainState):
    """Étend TrainState pour stocker les paramètres du Target Network (EMA)."""
    target_params: Any = None
    step_metrics: Dict = None


# ─────────────────────────────────────────────────────────────────────────────
# pmap'd train step
# ─────────────────────────────────────────────────────────────────────────────
# AUDIT §3.5 — le stub `_train_step` (décoré `jax.pmap` puis levant
# NotImplementedError) a été supprimé : seul `make_train_step()` construit la
# fonction réellement utilisée.


def make_train_step(
    network: MuZeroNetwork,
    probe_heads: ProbeHeads,
    cfg: Config,
    start_step: int = 0,
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
    K_unroll           = int(cfg.train.num_unroll_steps)
    gamma              = float(cfg.train.gamma)
    max_actions        = int(cfg.model.max_actions)

    def _step(
        state: train_state.TrainState,
        batch: dict,
        rng: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, dict, jnp.ndarray]:

        target_params = state.target_params if hasattr(state, "target_params") and state.target_params is not None else state.params
        mz_target_params = target_params["muzero"]

        # ── Reanalyze In-Pipeline GPU (utilisant le Target Network) ───────────
        rng_device = jax.random.fold_in(rng, jax.lax.axis_index("devices"))

        B = batch["action_seq"].shape[0]
        valid_seq = batch["valid_seq"].astype(jnp.float32)        # [B, K+1]

        # AUDIT §1.3 — Reanalyze n-step correct.
        # Ancien comportement : `target_val[:,0]` était écrasé par la valeur
        # racine du MCTS, produite intégralement par le réseau cible, sans
        # AUCUNE récompense réelle.  La tête de valeur n'était donc jamais
        # ancrée dans le résultat des parties (récompenses sparses ±1 au
        # terminal) → point fixe arbitraire, dérive puis effondrement, et une
        # priorité PER qui ne mesurait que le désaccord online/target.
        #
        # Nouveau comportement (MuZero Reanalyze standard) :
        #     V(s_0) = Σ_{k<K} γ^k · r_k · valid_k  +  γ^K · v_MCTS(s_K) · valid_K
        # NB : `reward_seq[k]` est la récompense perçue AU pas k (cf. add_game :
        # rew_window = rewards[t:end], obs_window = obs_list[t:end+1]).  Sa
        # validité est donc `valid_seq[:, k]`.  Le masque `valid_seq[:, 1:]`
        # utilisé initialement décalait d'un cran et annulait la récompense
        # terminale ±1 de toute fenêtre tronquée par la fin de partie : comme
        # c'est la seule récompense non nulle du jeu, `partial_ret` valait
        # toujours 0 et la cible redevenait un pur auto-bootstrap.
        # Les deux MCTS (racine et état de bootstrap) sont exécutés en UN SEUL
        # appel mctx sur un batch concaténé : le coût supplémentaire est un
        # forward de h() sur B exemples, l'arbre lui-même restant en latent.
        obs0 = {k: v[:, 0]        for k, v in batch["obs_seq"].items()}   # [B, ...]
        obsK = {k: v[:, K_unroll] for k, v in batch["obs_seq"].items()}   # [B, ...]
        obs_cat = {k: jnp.concatenate([obs0[k], obsK[k]], axis=0) for k in obs0}

        z_cat = network.apply(mz_target_params, obs_cat, method=network.represent)
        pi_cat, v_cat = network.apply(mz_target_params, z_cat, method=network.predict)

        mask_cat = obs_cat["option_mask"].astype(jnp.bool_)               # [2B, A]
        # Les lignes paddées (fin de partie) ont un masque entièrement faux :
        # mctx produirait des NaN.  On leur donne une action légale factice ;
        # leur contribution est de toute façon annulée par `valid_seq[:, K]`.
        any_legal = jnp.any(mask_cat, axis=-1, keepdims=True)
        fallback  = (jnp.arange(max_actions)[None, :] == 0)
        mask_cat  = jnp.where(any_legal, mask_cat, fallback)

        fresh_pol_cat, fresh_val_cat = reanalyze_root(
            mz_target_params, network, z_cat, pi_cat, v_cat, mask_cat,
            rng_device, reanalyze_sims, reanalyze_consider,
        )

        fresh_pol = fresh_pol_cat[:B]                                     # [B, A]
        v_mcts_K  = fresh_val_cat[B:]                                     # [B]

        discounts   = gamma ** jnp.arange(K_unroll, dtype=jnp.float32)    # [K]
        rewards_ok  = batch["reward_seq"] * valid_seq[:, :K_unroll]       # [B, K]
        partial_ret = jnp.sum(rewards_ok * discounts[None, :], axis=-1)   # [B]
        bootstrap   = (gamma ** K_unroll) * v_mcts_K * valid_seq[:, K_unroll]
        fresh_val   = partial_ret + bootstrap                             # [B]

        # MuZero Reanalyze v2 standard : combiner le retour empirique stocké
        # (portant le vrai signal ±1 de victoire/défaite calculé avec td_steps=20)
        # avec la valeur fraîche MCTS (fresh_val).
        # Évite le découplage de la valeur en récompense sparse et harmonise k=0
        # avec k=1..K.
        stored_val  = batch["target_val"][:, 0]
        blended_val = 0.5 * stored_val + 0.5 * fresh_val

        batch = {
            **batch,
            "target_pol": batch["target_pol"].at[:, 0].set(jax.lax.stop_gradient(fresh_pol)),
            "target_val": batch["target_val"].at[:, 0].set(jax.lax.stop_gradient(blended_val)),
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

        # Synchronise gradients across devices
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss  = jax.lax.pmean(loss,  axis_name="devices")

        # ── Dégel progressif de h(s) (Actif UNIQUEMENT en mode hot_fix) ──────
        is_hot_fix = bool(getattr(cfg.train, "hot_fix", False))
        step_val = jnp.maximum(state.step - start_step, 0)
        freeze_steps = getattr(cfg.train, "freeze_representation_steps", 0)
        ramp_steps = getattr(cfg.train, "unfreeze_ramp_steps", 1)
        h_scale = jnp.where(
            is_hot_fix & ((freeze_steps > 0) | (ramp_steps > 0)),
            jnp.clip((step_val - freeze_steps) / jnp.maximum(ramp_steps, 1), 0.0, 1.0),
            1.0
        )
        if "muzero" in grads:
            mz_g = grads["muzero"]["params"] if "params" in grads["muzero"] else grads["muzero"]
            if "h" in mz_g:
                mz_g["h"] = jax.tree_util.tree_map(lambda g: g * h_scale, mz_g["h"])

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

        metrics = {**metrics, "h_grad_scale": h_scale}

        # ── Target Network Polyak EMA Update (theta^- = tau * theta^- + (1-tau) * theta) ──
        tau = getattr(cfg.train, "target_network_tau", 0.995)
        new_target_params = jax.tree_util.tree_map(
            lambda tp, p: tau * tp + (1.0 - tau) * p,
            target_params,
            new_state.params,
        )
        new_state = new_state.replace(target_params=new_target_params)

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
    mz_params = network.init(rng_mz, batch_obs, method=network.init_all)

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

    return MuZeroTrainState.create(
        apply_fn=network.apply,
        params=params,
        target_params=params,
        tx=tx,
    )


def _merge_params(defaults, loaded):
    """Conserve les poids/états restaurés et réinitialise les parties absentes ou incompatibles (forme / structure).

    AUDIT §1.1/§3.5 — les clés présentes dans le checkpoint mais ABSENTES de
    l'architecture courante sont désormais écartées (ex. `pi_q`, `pi_k`,
    `a_feat_emb` supprimés).  Auparavant elles étaient réinjectées dans l'arbre
    de paramètres : Adam maintenait des moments pour des poids morts et le
    checkpoint suivant les propageait indéfiniment.
    """
    is_defaults_map = isinstance(defaults, Mapping)
    is_loaded_map = isinstance(loaded, Mapping)

    if is_defaults_map and is_loaded_map:
        res = {}
        for k, v in defaults.items():
            if k in loaded:
                res[k] = _merge_params(v, loaded[k])
            else:
                res[k] = v
        dropped = [k for k in loaded if k not in defaults]
        if dropped:
            logger.info(
                "[checkpoint] Clés obsolètes ignorées (absentes de l'architecture courante) : %s",
                ", ".join(str(k) for k in dropped),
            )
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
                "Shape mismatch detected for layer (model %s vs checkpoint %s); adapting/resetting this layer.",
                defaults.shape, loaded.shape
            )
            # Adapt 2D kernels if input dimension expanded (e.g. option_proj: 157 -> 160)
            if defaults.ndim == 2 and loaded.ndim == 2 and defaults.shape[1] == loaded.shape[1] and defaults.shape[0] > loaded.shape[0]:
                logger.info("Partial adaptation of weight matrix %s -> %s (preserving existing weights).", loaded.shape, defaults.shape)
                new_param = np.array(defaults)
                new_param[:loaded.shape[0], :] = np.array(loaded)
                return jnp.array(new_param)
            return defaults
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Reset chirurgical de la tête de valeur / récompense
# ─────────────────────────────────────────────────────────────────────────────
# NOTE : cette fonction avait été classée « code mort » en §3.5 puis supprimée.
# C'était une erreur d'appréciation : elle est exactement l'outil nécessaire pour
# repartir après les correctifs §1.3 / §2.2 / §2.3, qui n'ont abîmé QUE la tête
# de valeur (cible auto-référentielle + padding tiré vers 0 + saturation ±2) et
# la couche de sortie de la tête de récompense.  Le tronc Transformer h, la tête
# de politique et la fonction de transition de g restent valides et coûtent
# beaucoup plus cher à réapprendre.
DEFAULT_VALUE_HEAD_FRAGMENTS = ("v_dense", "rdet_fc2")


def _matches_fragment(path, fragments) -> bool:
    keys = [getattr(p, "key", p) for p in path]
    path_str = "/".join(str(k) for k in keys)
    return any(f in path_str for f in fragments)


def _reset_value_head_params(params: dict, fresh_params: dict, fragments=None) -> dict:
    """Réinitialise les couches dont le chemin contient l'un de ``fragments``.

    Par défaut : ``v_dense`` (tête de valeur catégorielle) et ``rdet_fc2``
    (couche de sortie de la tête de récompense).  Tout le reste — h, pi_dense,
    det_fc1/det_fc2, projecteurs, embeddings de cartes, sondes — est conservé
    bit pour bit.
    """
    frags = tuple(fragments or DEFAULT_VALUE_HEAD_FRAGMENTS)
    reset_paths: list[str] = []

    def _reset_leaf(path, param_val):
        if not _matches_fragment(path, frags):
            return param_val
        keys = [getattr(p, "key", p) for p in path]
        reset_paths.append("/".join(str(k) for k in keys))
        val = fresh_params
        for k in keys:
            val = val[k]
        return val

    out = jax.tree_util.tree_map_with_path(_reset_leaf, params)
    if reset_paths:
        logger.info(
            "[reset-value-head] %d couche(s) réinitialisée(s) : %s",
            len(reset_paths), ", ".join(reset_paths),
        )
    else:
        logger.warning(
            "[reset-value-head] Aucune couche ne correspond à %s — rien réinitialisé !", frags
        )
    return out


def _zero_adam_moments(opt_state, params, fragments=None):
    """Remet à zéro les moments Adam (mu / nu) des seules couches réinitialisées.

    Indispensable : conserver des moments accumulés sur d'anciens poids en face
    de poids fraîchement initialisés produit un premier pas de gradient énorme,
    qui détruirait la couche neuve.  Les moments des couches CONSERVÉES (h, g,
    politique) restent intacts — on ne repaie pas la reconstruction de l'inertie
    d'Adam pour tout le réseau.
    """
    frags = tuple(fragments or DEFAULT_VALUE_HEAD_FRAGMENTS)
    mask = jax.tree_util.tree_map_with_path(
        lambda path, _leaf: _matches_fragment(path, frags), params
    )

    def _zero(tree):
        return jax.tree_util.tree_map(
            lambda hit, arr: jnp.zeros_like(arr) if hit else arr, mask, tree
        )

    def _walk(node):
        if hasattr(node, "mu") and hasattr(node, "nu"):
            return node._replace(mu=_zero(node.mu), nu=_zero(node.nu))
        if isinstance(node, tuple):
            rebuilt = tuple(_walk(x) for x in node)
            return type(node)(*rebuilt) if hasattr(node, "_fields") else rebuilt
        return node

    return _walk(opt_state)


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

        dir_eps = float(cfg.search.dirichlet_epsilon) if train_mode else 0.0
        dir_alp = float(cfg.search.dirichlet_alpha) if train_mode else 0.3

        best_action, avg_policy, avg_value = ismcts_action(
            network, params_cpu["muzero"], enc_obs, option_mask, rng_act, cfg,
            dirichlet_epsilon=dir_eps, dirichlet_alpha=dir_alp,
        )

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
    actor_devices: Optional[list] = None,
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

    # Déterminer les accélérateurs dédiés au Self-Play
    if actor_devices:
        accel_devices = actor_devices
    else:
        devices = jax.devices()
        accel_devices = [d for d in devices if d.platform in ("gpu", "tpu")]
        if not accel_devices:
            accel_devices = devices  # fallback CPU
    logger.info(
        "[self-play] Launching parallel self-play: %d workers, sims=%d, belief_samples=%d, dirichlet_eps=%.3f, dirichlet_alpha=%.3f",
        num_workers,
        cfg.search.num_simulations,
        cfg.search.num_belief_samples,
        cfg.search.dirichlet_epsilon,
        cfg.search.dirichlet_alpha,
    )
    logger.info("[self-play] Actor TPU/GPU accelerators (%s) for MCTS: %s", accel_devices[0].platform, accel_devices)
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

    from models.deck_builder import sample_deck, DEFAULT_COMPETITIVE_DECK
    deck_logits, _ = deck_net.apply(deck_params)

    pipe_meta = [{} for _ in range(num_workers)]
    worker_steps = [0] * num_workers
    last_msg_time = [time.time()] * num_workers

    # AUDIT §3.1 — Les threads d'inférence écrivent dans `pipes[i]` pendant que le
    # thread principal peut fermer et remplacer ce même pipe (worker gelé ou
    # relancé).  Un verrou par pipe sérialise ces accès, et un compteur d'« époque »
    # permet de jeter les résultats calculés pour une instance de worker désormais
    # morte (sinon une action est envoyée à un worker qui attend un `start`).
    pipe_locks = [threading.Lock() for _ in range(num_workers)]
    pipe_epoch = [0] * num_workers

    def start_game_on_worker(pipe_idx, local_rng):
        nonlocal games_started
        worker_steps[pipe_idx] = 0
        local_rng, r_d0, r_d1 = jax.random.split(local_rng, 3)
        # Utilisation du deck compétitif de référence pour le self-play
        deck0 = list(DEFAULT_COMPETITIVE_DECK)
        deck1 = list(DEFAULT_COMPETITIVE_DECK)
        ids0 = np.array(deck0, dtype=np.int32)
        ids1 = np.array(deck1, dtype=np.int32)
        
        # Réanimation robuste du worker s'il s'est arrêté (ex. suite à une exception)
        p = processes[pipe_idx]
        if not p.is_alive():
            logger.info(f"[coordinator] Restarting stopped worker {pipe_idx}...")
            with pipe_locks[pipe_idx]:
                try:
                    pipes[pipe_idx].close()
                except Exception:
                    pass
                parent_conn, child_conn = ctx.Pipe()
                new_p = ctx.Process(target=_worker_bootstrap, args=(child_conn, pipe_idx, cfg), daemon=True)
                new_p.start()
                pipes[pipe_idx] = parent_conn
                processes[pipe_idx] = new_p
                pipe_epoch[pipe_idx] += 1

        pipe_meta[pipe_idx] = {
            "ids0": ids0,
            "ids1": ids1,
            "game_active": True
        }
        last_msg_time[pipe_idx] = time.time()
        try:
            with pipe_locks[pipe_idx]:
                pipes[pipe_idx].send({
                    "cmd": "start",
                    "deck0": list(deck0),
                    "deck1": list(deck1),
                    "num_belief_samples": int(cfg.search.num_belief_samples)
                })
            games_started += 1
        except Exception as e:
            logger.error(f"[coordinator] Failed to send to restarted worker {pipe_idx}: {e}")
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
    # NB : ne PAS refaire `import threading` ici.  Le module est déjà importé en
    # tête de fichier ; un import local en fait une variable locale pour TOUTE la
    # fonction, y compris la ligne `pipe_locks = [threading.Lock() ...]` qui la
    # précède → UnboundLocalError au premier appel.
    import queue as _pyqueue

    work_queue: "_pyqueue.Queue" = _pyqueue.Queue()
    stop_event = threading.Event()

    # RNG séparé par GPU thread (évite tout partage de state JAX entre threads)
    _gpu_thread_rngs = [jax.random.fold_in(rng, gidx) for gidx in range(len(accel_devices))]

    def _gpu_inference_worker(gpu_idx: int):
        _rng = _gpu_thread_rngs[gpu_idx]
        while not stop_event.is_set():
            # ── Attendre le premier item disponible (timeout court pour re-vérifier stop_event) ──
            try:
                first = work_queue.get(timeout=0.050)
            except _pyqueue.Empty:
                continue

            items = [first]


            # ── Collecte : agréger les requêtes assignées à cet Actor TPU ───
            actor_batch_size = max(1, (num_workers + len(accel_devices) - 1) // len(accel_devices))
            _t0 = time.time()
            while len(items) < actor_batch_size and time.time() - _t0 < 0.015:
                try:
                    items.append(work_queue.get(timeout=0.002))
                except _pyqueue.Empty:
                    continue


            # ── Build batch padded à actor_batch_size (taille fixe → 0 recompilation JAX) ──
            na_indices  = [it[0] for it in items]
            na_encs     = [it[1] for it in items]
            na_masks    = [it[2] for it in items]
            na_epochs   = [it[3] for it in items]

            _pe = list(na_encs)
            _pm = list(na_masks)
            while len(_pe) < actor_batch_size:
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
                logger.error("[Acc%d] Inference error (batch=%d): %s", gpu_idx, _n, _e, exc_info=True)
                ba_np = np.array([int(np.random.choice(np.where(na_masks[i])[0])) for i in range(_n)])
                ap_np = np.zeros((_n, int(cfg.model.max_actions)), dtype=np.float32)
                av_np = np.zeros(_n, dtype=np.float32)

            # ── Envoyer les résultats aux workers ────────────────────────────
            # AUDIT §3.1 : verrou par pipe + contrôle d'époque (un worker relancé
            # entre-temps ne doit pas recevoir la réponse de l'ancienne instance).
            for _i, _pidx in enumerate(na_indices):
                try:
                    with pipe_locks[_pidx]:
                        if pipe_epoch[_pidx] != na_epochs[_i]:
                            logger.debug(
                                "[Acc%d] Stale result ignored for worker %d (epoch %d != %d).",
                                gpu_idx, _pidx, na_epochs[_i], pipe_epoch[_pidx],
                            )
                            continue
                        pipes[_pidx].send({
                            "action_indices": [int(ba_np[_i])],
                            "search_pol":     ap_np[_i],
                            "search_val":     float(av_np[_i]),
                        })
                except Exception as _e:
                    logger.error("[Acc%d] Error sending action to worker %d: %s", gpu_idx, _pidx, _e)

    # Lancer autant de threads GPU/TPU qu'il y a d'accélérateurs
    gpu_threads = []
    for _gidx in range(len(_gpu_params)):
        _t = threading.Thread(target=_gpu_inference_worker, args=(_gidx,), daemon=True,
                              name=f"accel-inference-{_gidx}")
        _t.start()
        gpu_threads.append(_t)
        logger.info("[coordinator] Accelerator inference thread %d started (device: %s)", _gidx, accel_devices[_gidx])

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
                work_queue.put((pipe_idx, msg["batched_enc"], msg["option_mask"], pipe_epoch[pipe_idx]))

            else:
                tracker.update()

            if status == "deck_error":
                deck_errors_count += 1
                logger.warning(f"[coordinator] Deck error on worker {pipe_idx}: {msg.get('error')}")
                meta = pipe_meta[pipe_idx]
                for p_idx, ids in [("player 0", meta.get("ids0")), ("player 1", meta.get("ids1"))]:
                    if ids is not None:
                        from collections import Counter
                        counts = Counter(list(np.array(ids)))
                        deck_summary = []
                        for cid, count in counts.items():
                            name = card_data.card_name(int(cid)) if card_data else f"ID {cid}"
                            is_ace = int(cid) in ace_spec_ids
                            deck_summary.append(f"{name} (ID: {cid}, count: {count}{', ACE' if is_ace else ''})")
                        logger.info(f"  Involved deck for {p_idx}: {', '.join(deck_summary)}")
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
                logger.info(f"[coordinator] Game completed on worker {pipe_idx}! ({games_completed}/{num_games_to_play} completed, {games_started}/{num_games_to_play} started)")
                if games_started < num_games_to_play:
                    rng, rng_next = jax.random.split(rng)
                    start_game_on_worker(pipe_idx, rng_next)
                else:
                    pipe_meta[pipe_idx]["game_active"] = False

            elif status == "error":
                logger.error(f"[coordinator] Worker {pipe_idx} sent fatal error: {msg.get('error')}")
                pipe_meta[pipe_idx]["game_active"] = False
                games_completed += 1
                if games_started < num_games_to_play:
                    logger.info(f"[coordinator] Attempting to replace failed worker {pipe_idx}...")
                    rng, rng_restart = jax.random.split(rng)
                    start_game_on_worker(pipe_idx, rng_restart)

        # Détecter et relancer les workers réellement gelés (inactivité > 240s)
        now = time.time()
        for idx in range(num_workers):
            if pipe_meta[idx].get("game_active") and (now - last_msg_time[idx] > 240.0):
                logger.warning(
                    f"[coordinator] Worker {idx} appears frozen (idle for {now - last_msg_time[idx]:.1f}s, step {worker_steps[idx]}). Force restarting..."
                )

                try:
                    processes[idx].terminate()
                    processes[idx].join(timeout=2.0)
                    if processes[idx].is_alive():
                        processes[idx].kill()
                except Exception as e:
                    logger.error(f"[coordinator] Error terminating worker {idx}: {e}")
                with pipe_locks[idx]:
                    try:
                        pipes[idx].close()
                    except Exception:
                        pass
                    parent_conn, child_conn = ctx.Pipe()
                    new_p = ctx.Process(target=_worker_bootstrap, args=(child_conn, idx, cfg), daemon=True)
                    new_p.start()
                    pipes[idx] = parent_conn
                    processes[idx] = new_p
                    pipe_epoch[idx] += 1
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
            logger.info("[wandb] API key retrieved from Kaggle secret '%s'", cfg.wandb.kaggle_secret_name)
    except Exception:
        pass

    # 2. Fallback si WANDB est défini comme variable d'environnement au lieu de WANDB_API_KEY
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB"):
        os.environ["WANDB_API_KEY"] = os.environ.get("WANDB")

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("[wandb] No WANDB_API_KEY or Kaggle secret '%s' found. WandB disabled.", cfg.wandb.kaggle_secret_name)
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
        logger.info("[wandb] wandb.init() successful! (project=%s, run_id=%s, url=%s)", cfg.wandb.project, run.id, url)
        return run
    except Exception as e:
        logger.warning("[wandb] Could not initialize WandB: %s", e)
        return None


def _log_wandb_full_metrics(
    metrics: dict,
    buffer,
    batch_cpu: dict,
    td_err_flat: np.ndarray,
    step: int,
    elapsed: float,
    cfg: Config,
):
    if not getattr(cfg, "wandb", None) or not cfg.wandb.enabled:
        return
    try:
        import wandb
        if wandb.run is None:
            return

        def _get_scalar(k, default=0.0):
            if k not in metrics:
                return default
            val = metrics[k]
            try:
                if hasattr(val, "ndim") and val.ndim > 0:
                    return float(val[0])
                return float(val)
            except Exception:
                return default

        w_dict = {}

        # ── 1. Losses ──────────────────────────────────────────────────────────
        w_dict["loss/total"]       = _get_scalar("loss_total")
        w_dict["loss/policy"]      = _get_scalar("loss_policy")
        w_dict["loss/value"]       = _get_scalar("loss_value")
        w_dict["loss/reward"]      = _get_scalar("loss_reward")
        w_dict["loss/probes"]      = _get_scalar("loss_probes")
        if "loss_consistency" in metrics:
            w_dict["loss/consistency"] = _get_scalar("loss_consistency")

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
            p_losses = np.array(metrics["probe_per_task"])
            if p_losses.ndim > 1:
                p_losses = p_losses[0]
            for i, name in enumerate(PROBE_NAMES):
                if i < len(p_losses):
                    w_dict[f"probes_loss/{name}"] = float(p_losses[i])

        if "probe_acc_per_task" in metrics:
            p_accs = np.array(metrics["probe_acc_per_task"]) * 100.0
            if p_accs.ndim > 1:
                p_accs = p_accs[0]
            w_dict["probes/accuracy_mean"] = float(np.mean(p_accs))
            for i, name in enumerate(PROBE_NAMES):
                if i < len(p_accs):
                    w_dict[f"probes_acc/{name}"] = float(p_accs[i])

        # ── 5. Policy & Value Telemetry ───────────────────────────────────────
        if "policy_entropy_norm" in metrics:
            w_dict["policy/entropy_norm"] = _get_scalar("policy_entropy_norm")
        if "policy_entropy_net" in metrics:
            w_dict["policy/entropy_net"]  = _get_scalar("policy_entropy_net")
        if "policy_entropy_mcts" in metrics:
            w_dict["policy/entropy_mcts"] = _get_scalar("policy_entropy_mcts")
        if "policy_p_max_mean" in metrics:
            w_dict["policy/p_max_mean"]   = _get_scalar("policy_p_max_mean")
        if "h_grad_scale" in metrics:
            w_dict["policy/h_grad_scale"] = _get_scalar("h_grad_scale")
        if "value_mean" in metrics:
            w_dict["value/mean"]          = _get_scalar("value_mean")
        if "value_abs_mean" in metrics:
            w_dict["value/abs_mean"]      = _get_scalar("value_abs_mean")
        if "value_saturation_pct" in metrics:
            w_dict["value/saturation_pct"] = _get_scalar("value_saturation_pct")

        # ── 6. Action Distribution & Game Flow ────────────────────────────────
        w_dict["game/avg_episode_length"] = float(tracker.avg_transitions_per_game)
        act_dist = tracker.get_action_percentages()
        for k, v in act_dist.items():
            w_dict[f"action_dist/{k}_pct"] = float(v)

        # ── 7. Replay Buffer Stats ─────────────────────────────────────────────
        w_dict["buffer/size"] = len(buffer)
        if hasattr(buffer, "_sum_tree"):
            total_prio = float(buffer._sum_tree.query_all())
            w_dict["buffer/total_priority"] = total_prio
            w_dict["buffer/avg_priority"]   = total_prio / max(len(buffer), 1)
        if hasattr(buffer, "get_max_priority"):
            w_dict["buffer/max_priority"]   = float(buffer.get_max_priority())

        # ── 8. Performance & System Metrics ────────────────────────────────────
        w_dict["perf/step_time_sec"] = elapsed
        w_dict["perf/steps_per_sec"] = 1.0 / (elapsed + 1e-8)

        wandb.log(w_dict, step=step)

    except Exception as e:
        logger.warning("[wandb] Error during extended log: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: Config) -> None:
    """Point d'entrée principal d'entraînement."""
    logger.info("=== PTCG MuZero training start ===")
    logger.info("Devices: %s", jax.devices())
    logger.info(
        "[search-config] MCTS self-play: sims=%d, belief_samples=%d, max_considered=%d, dirichlet_eps=%.3f, dirichlet_alpha=%.3f, temp_init=%.2f, temp_min=%.2f",
        cfg.search.num_simulations,
        cfg.search.num_belief_samples,
        cfg.search.max_num_considered_actions,
        cfg.search.dirichlet_epsilon,
        cfg.search.dirichlet_alpha,
        cfg.search.temperature_init,
        cfg.search.temperature_min,
    )

    Path(cfg.infra.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.infra.log_dir).mkdir(parents=True, exist_ok=True)

    _init_wandb(cfg)

    # ── 1. Card data ──────────────────────────────────────────────────────
    card_data    = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids

    static_np    = card_data.feature_matrix(num_card_ids)
    static_jax   = jnp.array(static_np)   # frozen, never in params

    # Mark basic energy IDs (only Basic Energy cards are unlimited, Special Energies have max 4 copies)
    energy_ids = [
        cid for cid in card_data.card_ids
        if card_data._cards[cid].get("stage", "").strip().lower() == "basic energy"
    ]
    set_energy_ids(energy_ids)
    basic_pokemon_ids = [
        cid for cid in card_data.card_ids
        if card_data._cards[cid].get("stage", "").strip().lower()
        in ("basic pokémon", "basic pokemon")
    ]
    set_basic_pokemon_ids(basic_pokemon_ids)
    id_to_name = {cid: card_data.card_name(cid) for cid in card_data.card_ids}
    set_card_names(id_to_name)

    # ── Détection des cartes Ace Spec ─────────────────────────────────────
    # Only explicit card metadata is trustworthy here.  A card appearing once
    # in a reference deck is not evidence that it is an Ace Spec.
    ace_spec_set = set(card_data.ace_spec_ids)
    logger.info("[ace-spec] %d Ace Spec card(s) detected via cards.csv: %s", len(ace_spec_set), list(ace_spec_set))

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
        logger.info("[ace-spec] After merge with all_card_data(): %d Ace Spec cards", len(ace_spec_set))
    except Exception as e:
        logger.warning("[ace-spec] all_card_data() failed: %s", e)

    ace_spec_ids = sorted(list(ace_spec_set))
    if not ace_spec_ids:
        logger.warning("[ace-spec] No Ace Spec cards detected!")

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

    target_step = getattr(cfg.train, "resume_step", None)
    target_ckpt = getattr(cfg.train, "resume_ckpt", None)

    # 1. Tenter de charger depuis un chemin explicite si spécifié (--ckpt ou -s chemin.pkl)
    if target_ckpt and Path(target_ckpt).exists():
        try:
            import pickle
            with open(target_ckpt, "rb") as f:
                ckpt_data = pickle.load(f)
            loaded_params = ckpt_data["params"]
            loaded_opt_state = ckpt_data.get("opt_state", None)
            loaded_deck_params = ckpt_data.get("deck", {})
            loaded_deck_opt_state = ckpt_data.get("deck_opt_state", None)
            start_step = ckpt_data.get("step", 0)
            logger.info("=== Loaded explicit local checkpoint: %s (step %d) ===", target_ckpt, start_step)
        except Exception as e:
            logger.warning("Failed to load explicit checkpoint %s: %s", target_ckpt, e)

    # 2. Tenter de charger depuis HF Hub si activé (avec target_step si spécifié)
    if loaded_params is None and cfg.hf.enabled and cfg.hf.repo_id:
        try:
            from export.hub import load_from_hub
            logger.info("Checking model on HuggingFace Hub (%s, step=%s)...", cfg.hf.repo_id, target_step)
            hf_mz, hf_dk, hf_cfg, hf_step = load_from_hub(cfg.hf.repo_id, step=target_step, cfg=cfg)
            if hf_mz:
                loaded_params = hf_mz
                loaded_deck_params = hf_dk
                start_step = hf_step
                logger.info("=== Model loaded from HF Hub: %s (step %d) ===", cfg.hf.repo_id, start_step)
        except Exception as e:
            logger.info("No HuggingFace Hub model downloaded or check failed (%s): %s", cfg.hf.repo_id, e)

    # 3. Vérifier le checkpoint local UNIQUEMENT si aucun step/ckpt explicite n'a été demandé
    if target_step is None and target_ckpt is None and latest_path.exists():
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
                logger.info("=== Resuming from latest local checkpoint: %s (step %d) ===", latest_path, start_step)
            elif loaded_params is not None:
                logger.info("=== Local checkpoint (step %d) <= HF Hub (step %d). Keeping HF Hub weights. ===", local_step, start_step)
        except Exception as e:
            logger.warning("Failed to read existing local checkpoint: %s.", e)

    if loaded_params is not None:
        state = create_muzero_train_state(network, probes, cfg, r1, dummy_obs)
        params_jax = jax.tree_util.tree_map(jax.device_put, loaded_params)
        params_jax = _merge_params(state.params, params_jax)

        # ── HOT-FIX SURGERY / RESET PARTIEL ──────────────────────────────────
        is_hot_fix = getattr(cfg.train, "hot_fix", False)
        reset_f = is_hot_fix or getattr(cfg.train, "reset_policy_head", False)
        reset_g = is_hot_fix or getattr(cfg.train, "reset_dynamics_head", False)

        def _get_mz_dict(p_dict):
            if "muzero" in p_dict:
                if isinstance(p_dict["muzero"], dict) and "params" in p_dict["muzero"]:
                    return p_dict["muzero"]["params"]
                return p_dict["muzero"]
            return p_dict

        target_mz = _get_mz_dict(params_jax)
        fresh_mz = _get_mz_dict(state.params)

        if reset_f:
            logger.info("=== [HOT-FIX] Surgical reset of Prediction Network f (policy π & value V) ===")
            if "f" in fresh_mz:
                target_mz["f"] = fresh_mz["f"]
        if reset_g:
            logger.info("=== [HOT-FIX] Surgical reset of Dynamics Network g (50-dim transitions) and consistency heads ===")
            if "g" in fresh_mz:
                target_mz["g"] = fresh_mz["g"]
            if "project" in fresh_mz:
                target_mz["project"] = fresh_mz["project"]
            if "predict_next" in fresh_mz:
                target_mz["predict_next"] = fresh_mz["predict_next"]

        # ── Reset chirurgical de la tête de valeur (option la moins destructrice) ──
        # Ne touche que `v_dense` et `rdet_fc2`.  Sans effet si `reset_f` est déjà
        # actif (f entier vient d'être réinitialisé).
        reset_v = getattr(cfg.train, "reset_value_head", False) and not reset_f
        if reset_v:
            logger.info(
                "=== [RESET-VALUE-HEAD] Reset of value/reward head only "
                "(keeping h, policy, and transition g) ==="
            )
            params_jax = _reset_value_head_params(params_jax, state.params)

        # Le target network doit repartir des MÊMES poids : sinon l'EMA continue
        # d'alimenter le Reanalyze avec l'ancienne tête de valeur corrompue
        # pendant plusieurs centaines de steps (tau=0.995 → demi-vie ~138 steps).
        state = state.replace(params=params_jax, target_params=params_jax, step=jnp.array(start_step, dtype=jnp.int32))

        fresh_opt_state = state.tx.init(params_jax)
        if (reset_f or reset_g) or loaded_opt_state is None:
            logger.info("=== [HOT-FIX] Clean reset of Adam optimizer ===")
            state = state.replace(opt_state=fresh_opt_state)
        else:
            try:
                opt_state_jax = jax.tree_util.tree_map(jax.device_put, loaded_opt_state)
                opt_state_merged = _merge_params(fresh_opt_state, opt_state_jax)
                # Validation de compatibilité du PyTree avec les gradients du nouveau modèle
                jax.tree.map(lambda a, b: None, fresh_opt_state, opt_state_merged)
                if reset_v:
                    # Moments Adam remis à zéro pour les SEULES couches neuves.
                    opt_state_merged = _zero_adam_moments(opt_state_merged, params_jax)
                    logger.info(
                        "=== [RESET-VALUE-HEAD] Adam moments reset for v_dense / rdet_fc2 "
                        "(momentum kept for h, g, and policy) ==="
                    )
                state = state.replace(opt_state=opt_state_merged)
                logger.info("=== MuZero optimizer state successfully restored and merged at step %d ===", start_step)
            except Exception as e:
                logger.warning(
                    "Checkpoint optimizer state incompatible with new architecture (%s). Cleanly reinitializing optimizer at step %d.",
                    e, start_step
                )
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
                logger.info("=== Deck builder optimizer state successfully restored ===")
            except Exception as e:
                logger.warning("Could not restore deck builder optimizer state: %s", e)
                deck_opt_state = optax.adam(cfg.train.deck_lr).init(deck_params)
        else:
            deck_opt_state = optax.adam(cfg.train.deck_lr).init(deck_params)
        deck_baseline = 0.0
    else:
        deck_params, deck_opt_state, deck_baseline = create_deck_train_state(
            deck_net, cfg, r2
        )
    deck_optimizer = optax.adam(cfg.train.deck_lr)

    # ── 5. Découplage Matériel des Devices (Learner vs Actors) ───────────
    all_devs = jax.devices()[:cfg.infra.num_devices]
    num_total_devs = len(all_devs)

    # Répartition optimale en puissances de 2 (ex: 4 Learners + 4 Actors sur 8 TPUs)
    if num_total_devs >= 4:
        configured_learners = getattr(cfg.infra, "num_learner_devices", 0)
        num_learner_devs = configured_learners if configured_learners > 0 else (num_total_devs // 2)
        learner_devices = all_devs[:num_learner_devs]
        actor_devices   = all_devs[num_learner_devs:]
    elif num_total_devs in (2, 3):
        num_learner_devs = 1
        learner_devices = all_devs[:1]
        actor_devices   = all_devs[1:]
    else:
        num_learner_devs = 1
        learner_devices = all_devs
        actor_devices   = all_devs


    num_devs = len(learner_devices)
    state    = flax.jax_utils.replicate(state, learner_devices)
    logger.info(
        "[train] Decoupled TPU architecture: %d Learner(s) %s | %d Actor(s) %s",
        len(learner_devices), [str(d) for d in learner_devices],
        len(actor_devices), [str(d) for d in actor_devices],
    )

    train_step_fn = make_train_step(network, probes, cfg, start_step=start_step)


    # ── 6. Replay buffer (avec tentative de restauration HF Hub & local) ──
    from training.replay_buffer import PrioritizedReplayBuffer
    buffer = None
    buffer_meta = {}

    skip_buffer_load = getattr(cfg.train, "hot_fix", False) or getattr(cfg.train, "fresh_buffer", False)
    if skip_buffer_load:
        logger.info("=== [HOT-FIX / FRESH BUFFER] Starting with empty Replay Buffer (ignoring older passive games) ===")
    else:
        # 1. Tenter de charger le buffer depuis HF Hub si activé
        if cfg.hf.enabled and cfg.hf.repo_id:
            try:
                from export.hub import load_buffer_from_hub
                logger.info("Checking for existing Replay Buffer on HuggingFace Hub (%s)...", cfg.hf.repo_id)
                hf_buf, hf_meta = load_buffer_from_hub(cfg.hf.repo_id, cfg.train, cfg.model, cfg=cfg)
                if hf_buf is not None:
                    buffer = hf_buf
                    buffer_meta = hf_meta
            except Exception as e:
                logger.warning("[hf-buffer] HF Hub restore failed: %s", e)

    # 2. Vérifier si un buffer local existe (dans local_dir ou checkpoint_dir)
    local_buf_path = Path(cfg.hf.local_dir) / "replay_buffer.pkl"
    local_meta_path = Path(cfg.hf.local_dir) / "buffer_meta.json"
    if not local_buf_path.exists():
        local_buf_path = Path(cfg.infra.checkpoint_dir) / "replay_buffer.pkl"
        local_meta_path = Path(cfg.infra.checkpoint_dir) / "buffer_meta.json"

    if local_buf_path.exists():
        try:
            loc_buf = PrioritizedReplayBuffer.deserialize(str(local_buf_path), cfg.train, cfg.model)
            loc_step = getattr(loc_buf, "loaded_step", 0)
            if local_meta_path.exists():
                try:
                    import json
                    loc_meta = json.loads(local_meta_path.read_text(encoding="utf-8"))
                    loc_step = loc_meta.get("step", loc_step)
                except Exception:
                    pass

            if buffer is None or loc_step >= buffer_meta.get("step", 0):
                buffer = loc_buf
                fill_pct = round(100.0 * len(buffer) / max(buffer._max_size, 1), 2)
                logger.info(
                    "=== Replay Buffer restored from local storage (%s): %d/%d entries (%.1f%% full, step %d) ===",
                    local_buf_path, len(buffer), buffer._max_size, fill_pct, loc_step
                )
        except Exception as e:
            logger.warning("Failed to load local replay buffer (%s): %s", local_buf_path, e)

    if buffer is None:
        buffer = PrioritizedReplayBuffer(cfg.train, cfg.model)
        logger.info("No existing Replay Buffer found. Creating a fresh buffer.")
    else:
        fill_pct = round(100.0 * len(buffer) / max(buffer._max_size, 1), 2)
        logger.info(
            "Replay Buffer ready: %d/%d entries loaded (%.1f%% full, min required for seeding: %d).",
            len(buffer), buffer._max_size, fill_pct, cfg.train.min_replay_size
        )

    # ── 7. Parallel Self-play seed games ──────────────────────────────────
    rng, rng_selfplay = jax.random.split(rng)
    _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)

    from training.activity import tracker, format_h_status
    tracker.attach_buffer(buffer)  # lecture en direct de len(buffer) dans le heartbeat

    is_tpu = any(getattr(d, "platform", "") == "tpu" for d in jax.devices()) or os.environ.get("TPU_NAME") is not None or "TPU" in str(jax.devices())
    if getattr(cfg.train, "num_workers", 0) > 0:
        NUM_WORKERS = cfg.train.num_workers
    elif is_tpu:
        NUM_WORKERS = max(1, min(96, os.cpu_count() or 64))
        if getattr(cfg.train, "games_per_self_play", 8) == 8:
            cfg.train.games_per_self_play = NUM_WORKERS
        # Adapter l'intervalle pour maintenir le Replay Ratio d'or (~8x) avec 96 workers
        if getattr(cfg.train, "self_play_interval", 100) == 100:
            cfg.train.self_play_interval = max(100, int(NUM_WORKERS * 12.5))
    else:
        NUM_WORKERS = max(1, min(8, os.cpu_count() or 1))

    logger.info(
        "[train] Hardware detection: TPU=%s, %d CPU cores → %d workers (games_per_self_play=%d, self_play_interval=%d steps, target Replay Ratio ≈ 8.8x).",
        is_tpu, os.cpu_count() or 1, NUM_WORKERS, cfg.train.games_per_self_play, cfg.train.self_play_interval
    )


    # ── JIT Warmup (TPU / GPU) pour éliminer les latences de première compilation ──
    try:
        actor_batch_size = max(1, (NUM_WORKERS + len(actor_devices) - 1) // len(actor_devices))
        logger.info(
            "[train] TPU JIT-compilation warmup on %d Actor(s) (ismcts_action_batched, actor_batch_size=%d)...",
            len(actor_devices), actor_batch_size
        )
        from search.ismcts import ismcts_action_batched
        from env.encoding import encode_observation
        dummy_enc_single = encode_observation({}, 0, cfg.model)
        N_s = int(cfg.search.num_belief_samples)
        dummy_batch_enc_base = {}
        for k, v in dummy_enc_single.items():
            arr_sample = np.stack([v] * N_s, axis=0)
            dummy_batch_enc_base[k] = np.stack([arr_sample] * actor_batch_size, axis=0)
        dummy_omasks_base = np.ones((actor_batch_size, int(cfg.model.max_actions)), dtype=bool)

        for dev_idx, dev in enumerate(actor_devices):
            logger.info("  → Warmup and XLA compilation on Actor TPU %d (%s)...", dev_idx, dev)
            dev_params = jax.device_put(_params_cpu, dev)
            dev_enc = {k: jax.device_put(v, dev) for k, v in dummy_batch_enc_base.items()}
            dev_omasks = jax.device_put(dummy_omasks_base, dev)
            dev_rng = jax.device_put(jax.random.PRNGKey(dev_idx), dev)
            _ba_warm, _ap_warm, _av_warm = ismcts_action_batched(
                network, dev_params, dev_enc, dev_omasks, dev_rng, cfg
            )
        logger.info("[train] ISMCTS warmup completed successfully on all Actor TPUs (batch=%d, belief=%d).", actor_batch_size, N_s)
    except Exception as exc:
        logger.warning("[train] ISMCTS warmup skipped (%s).", exc)






    if len(buffer) < cfg.train.min_replay_size:
        logger.info(
            "Seeding replay buffer (%d/%d current entries, min required=%d)...",
            len(buffer), buffer._max_size, cfg.train.min_replay_size
        )
        tracker.update(phase="Seeding Replay Buffer", buffer_size=len(buffer), deck_errors=0)

        _start_seeding = time.time()
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
                actor_devices=actor_devices,
            )

            tracker.self_play_times.append(time.time() - t_sp_start)
            
            _deck_errors += errs
            tracker.update(deck_errors=_deck_errors)
            for hist in hists:
                t_added = _add_history_to_buffer(hist, buffer, cfg)
                tracker.transitions_per_game_list.append(t_added)
            tracker.update(buffer_size=len(buffer))

        logger.info("Buffer seeded: %d entries in %.1fs", len(buffer), time.time() - _start_seeding)
    else:
        fill_pct = round(100.0 * len(buffer) / max(buffer._max_size, 1), 2)
        logger.info(
            "Replay Buffer already sufficiently filled (%d entries >= min=%d, %.1f%%). Seeding phase skipped!",
            len(buffer), cfg.train.min_replay_size, fill_pct
        )
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

    # ── 8. Main training loop ─────────────────────────────────────────────
    global_step = 0
    t0 = time.time()
    total_transitions_since_last_push = 0

    tracker.update(start_step=start_step)

    for step in range(start_step, cfg.train.num_total_steps):
        global_step = step
        new_step = step - start_step

        # ── a. Self-play ──────────────────────────────────────────────────
        if step % cfg.train.self_play_interval == 0:
            _tp = getattr(state, "target_params", None)
            if _tp is None:
                _tp = state.params
            _params_cpu = jax.tree_util.tree_map(lambda x: x[0], _tp)
            
            # Enregistrer immédiatement le modèle entraîné courant pour la suite
            _save_checkpoint(state, deck_params, cfg, step, force_latest_only=True, deck_opt_state=deck_opt_state)
            
            rng, rng_sp = jax.random.split(rng)
            
            tracker.update(phase=f"Self-Play Step {step} (new: {new_step})")

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
                actor_devices=actor_devices,
            )

            tracker.self_play_times.append(time.time() - t_sp_start)

            added_transitions = 0
            for hist in hists:
                t_added = _add_history_to_buffer(hist, buffer, cfg)
                added_transitions += t_added
                tracker.transitions_per_game_list.append(t_added)

                # AUDIT §3.6 — la distribution des types d'action est désormais
                # relevée par le worker à partir du champ `type` de l'option
                # réellement jouée, puis rejouée ici.  L'ancienne heuristique
                # (`argmax(option_feat[:17])`) renvoyait 0 pour toute ligne
                # paddée/nulle et gonflait donc la catégorie « other ».
                for opt_type in getattr(hist, "action_types", None) or []:
                    tracker.record_action(int(opt_type))

            total_transitions_since_last_push += added_transitions

            # AUDIT §2.4 — REINFORCE sur un deck constant n'apporte aucune
            # information (les deux joueurs jouent DEFAULT_COMPETITIVE_DECK et
            # reçoivent des récompenses opposées) : les mises à jour sont une
            # marche aléatoire sur les logits.  Activable via
            # `cfg.train.deck_builder_enabled` quand le deck redeviendra dynamique.
            if getattr(cfg.train, "deck_builder_enabled", False):
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
        batch_sharded = shard_batch(batch_cpu, num_devs)

        # Clé PRNG pour Reanalyze — répliquée puis différenciée par device via fold_in
        rng, rng_step = jax.random.split(rng)
        rng_step_replicated = jnp.broadcast_to(
            rng_step[None], (num_devs,) + rng_step.shape
        )

        t_step_start = time.time()
        state, metrics, td_errs = train_step_fn(
            state, batch_sharded, rng_step_replicated
        )
        tracker.train_step_times.append(time.time() - t_step_start)

        if float(metrics["update_is_finite"][0]) == 0.0:
            logger.error(
                "[train] Non-finite loss or gradient at step %d; update and priority refresh skipped.",
                step,
            )
            continue

        tracker.update(step=step)

        # Update priorities (TD error basé sur les fresh targets Reanalyze)
        td_err_flat = np.array(unshard(td_errs))
        buffer.update_priorities(indices, td_err_flat)

        # ── Priority Refresh allégé (tous les N steps) ────────────────────
        if step % cfg.train.priority_refresh_every == 0 and len(buffer) > 0:
            _tp = getattr(state, "target_params", None)
            if _tp is None:
                _tp = state.params
            _params_cpu_mz = jax.tree_util.tree_map(lambda x: x[0], _tp)["muzero"]
            _refresh_buffer_priorities(buffer, network, _params_cpu_mz, cfg)

        # ── c. Logging ────────────────────────────────────────────────────
        if step % 100 == 0:
            m = {k: float(v[0]) for k, v in metrics.items()
                 if not hasattr(v, '__len__') or v.ndim == 0 or
                 (hasattr(v, 'shape') and v.shape == (cfg.infra.num_devices,))}
            elapsed = time.time() - t0
            prb_acc_mean = float(np.mean(metrics.get("probe_acc_per_task", [np.zeros(11)])[0])) * 100.0
            
            h_norm = float(metrics.get("policy_entropy_norm", [0.0])[0])
            p_max = float(metrics.get("policy_p_max_mean", [0.0])[0])
            h_net = float(metrics.get("policy_entropy_net", [0.0])[0])
            h_mcts = float(metrics.get("policy_entropy_mcts", [0.0])[0])
            v_sat = float(metrics.get("value_saturation_pct", [0.0])[0])
            h_scale = float(metrics.get("h_grad_scale", [1.0])[0])
            act_dist = tracker.get_action_percentages()

            tracker.update(
                policy_entropy_norm=h_norm,
                policy_p_max=p_max,
                policy_entropy_net=h_net,
                policy_entropy_mcts=h_mcts,
                value_sat_pct=v_sat,
                h_grad_scale=h_scale,
            )

            logger.info(
                "step=%d (new:%d)  loss=%.4f  pol=%.4f  val=%.4f  rew=%.4f  "
                "prb=%.4f (acc=%.1f%%)  H_norm=%.2f  p_max=%.2f  H(π)=%.2f vs H(p)=%.2f  "
                "ATK=%.1f%% ATT=%.1f%% PLY=%.1f%% END=%.1f%%  h_scale=%.2f  buf=%d  %.1fs",
                step,
                new_step,
                float(metrics["loss_total"][0]),
                float(metrics["loss_policy"][0]),
                float(metrics["loss_value"][0]),
                float(metrics["loss_reward"][0]),
                float(metrics["loss_probes"][0]),
                prb_acc_mean,
                h_norm,
                p_max,
                h_mcts,
                h_net,
                act_dist.get("attack", 0.0),
                act_dist.get("attach", 0.0),
                act_dist.get("play", 0.0),
                act_dist.get("end", 0.0),
                h_scale,
                len(buffer),
                elapsed,
            )
            # Detailed per-probe loss and accuracy breakdown
            _log_probe_metrics(metrics, cfg)

            _log_wandb_full_metrics(
                metrics, buffer, batch_cpu, td_err_flat, step, elapsed, cfg
            )

        # ── d. Checkpoint & HF push ───────────────────────────────────────
        if step % cfg.train.checkpoint_every == 0 and step > 0:
            _save_checkpoint(state, deck_params, cfg, step, deck_opt_state=deck_opt_state)
            
            if cfg.hf.enabled:
                _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
                logger.info("[hf-push] Synchronous push of snapshot %d...", step)
                if not push_to_hub(_params_cpu, deck_params, cfg, step):
                    logger.error("[hf-push] Snapshot %d not confirmed on HF.", step)

        # ── e. Async replay buffer push to HF ─────────────────────────────
        if step % cfg.train.buffer_push_every == 0 and step > 0 and cfg.hf.enabled:
            from export.hub import push_buffer_to_hub_async
            push_buffer_to_hub_async(buffer, cfg, step)

    # Final checkpoint
    _save_checkpoint(state, deck_params, cfg, global_step, deck_opt_state=deck_opt_state)
    if cfg.hf.enabled:
        _params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
        logger.info("[hf-push] Launching final synchronous push to HuggingFace...")
        push_to_hub(_params_cpu, deck_params, cfg, global_step)
        # Final buffer push (sync — wait for completion before exit)
        from export.hub import push_buffer_to_hub_async
        push_buffer_to_hub_async(buffer, cfg, global_step)

    # Clean up persistent workers pool
    logger.info("[train] Cleaning up persistent worker pool...")
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
        actual_len = len(entries)
        
        # Padding à batch_size fixe pour garantir 0 recompilation XLA
        if actual_len < batch_size:
            entries = entries + [entries[-1]] * (batch_size - actual_len)

        obs_keys = entries[0].obs_seq[0].keys()
        obs_batch = {
            k: jnp.array(np.stack([entry.obs_seq[0][k] for entry in entries]))
            for k in obs_keys
        }

        z = network.apply(mz_params, obs_batch, method=network.represent)
        _, v_pred = network.apply(mz_params, z, method=network.predict)
        v_pred_arr = np.array(v_pred)[:actual_len]
        v_target = np.array([entry.target_val[0] for _, entry in chunk], dtype=np.float32)
        buffer.update_priorities(chunk_indices, np.abs(v_pred_arr - v_target))



def _add_history_to_buffer(hist: GameHistory, buffer: PrioritizedReplayBuffer, cfg: Config) -> int:

    if len(hist) == 0 or hist.returns is None:
        return 0

    # AUDIT §3.8 — les cibles de sonde sont désormais extraites côté worker et
    # `raw_states` (l'observation brute complète de chaque pas : main, défausse
    # de 60 cartes, prizes, logs…) n'est plus transmis par le pipe.  On garde le
    # chemin historique en secours pour les anciennes trajectoires.
    precomputed = getattr(hist, "probe_targets", None)
    if precomputed:
        rows = [np.asarray(p, dtype=np.int32) for p in precomputed]
    else:
        rows = [extract_probe_targets(raw, hist.player_idx) for raw in hist.raw_states]
    if not rows:
        rows = [np.full(11, -1, dtype=np.int32)] * len(hist.observations)
    probe_tgts = np.stack(rows + [np.full(11, -1, dtype=np.int32)])   # +1 for final state

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
