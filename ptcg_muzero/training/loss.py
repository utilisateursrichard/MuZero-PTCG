"""
ptcg_muzero/training/loss.py
==============================
Calcul de la loss MuZero complète, JIT-able et compatible pmap.

La loss totale est :
    L = w_pol · L_policy
      + w_val · L_value
      + w_rew · L_reward    (unrolled K steps)
      + w_prb · L_probes
      + reg                  (weight decay via optax)

Pour chaque step de déroulement k ∈ [0, K] :
  - L_policy  : KL(target_policy ‖ predicted_policy)  via cross-entropie
  - L_value   : MSE(target_return, predicted_value)
  - L_reward  : MSE(actual_reward, predicted_reward)   k ≥ 1

Toutes les quantités sont calculées en batch, batchées sur chaque device.
La moyenne des gradients entre devices est faite dans trainer.py via lax.pmean.
"""
from __future__ import annotations

from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax

from config import ModelConfig, TrainConfig
from interpretability.probes import ProbeHeads, probe_accuracy, probe_loss


# ─────────────────────────────────────────────────────────────────────────────
# Batch structure (tout ce qui arrive du replay buffer, déjà stacké en JAX)
# ─────────────────────────────────────────────────────────────────────────────
# Chaque champ a une dimension de batch B en tête.
# obs_seq : dict   — chaque valeur [B, K+1, ...]  (K = num_unroll_steps)
# action_seq       : [B, K, A]    float32 multi-hot
# reward_seq       : [B, K]       float32
# target_pol       : [B, K+1, A]  float32
# target_val       : [B, K+1]     float32
# probe_tgts       : [B, K+1, 5]  int32
# is_weights       : [B]          float32  (IS correction)


# ─────────────────────────────────────────────────────────────────────────────
# Loss principale
# ─────────────────────────────────────────────────────────────────────────────
def muzero_loss(
    params: dict,
    network,          # MuZeroNetwork Flax module (bound)
    probe_heads,      # ProbeHeads Flax module (bound)
    batch: dict,
    cfg_train: TrainConfig,
    cfg_model: ModelConfig,
) -> Tuple[jnp.ndarray, dict]:
    """
    Calcule la loss MuZero sur un batch.

    Paramètres
    ----------
    params      : dict contenant {muzero: ..., probes: ...}
    network     : MuZeroNetwork (apply via params["muzero"])
    probe_heads : ProbeHeads    (apply via params["probes"])
    batch       : dict issu du collate_batch()
    cfg_train   : TrainConfig
    cfg_model   : ModelConfig

    Retourne
    --------
    total_loss : scalar
    metrics    : dict de métriques pour le logging
    """
    K = cfg_train.num_unroll_steps
    B = batch["action_seq"].shape[0]

    mz_params  = params["muzero"]
    prb_params = params["probes"]

    # ── 0. Pré-encoder TOUTES les K+1 observations en une seule passe ────────
    # Shape batch["obs_seq"][key] : [B, K+1, ...]
    # On aplatit en [B*(K+1), ...], encode, puis reshape → [B, K+1, D]
    # Cela élimine K appels séquentiels à h() dans le scan (gain ~20-30%).
    obs_flat = {k: v.reshape((B * (K + 1),) + v.shape[2:])
                for k, v in batch["obs_seq"].items()}

    # h : [B*(K+1), D]
    z_all_flat = network.apply(mz_params, obs_flat, method=network.represent)
    z_all = z_all_flat.reshape(B, K + 1, -1)   # [B, K+1, D]

    # Projections pour consistency loss cible (stop_gradient côté cible uniquement)
    # project_state : [B*(K+1), D]
    proj_all_flat = network.apply(mz_params, z_all_flat, method=network.project_state)
    # stop_gradient : la cible de consistency ne doit pas être différenciée
    proj_all = jax.lax.stop_gradient(proj_all_flat.reshape(B, K + 1, -1))  # [B, K+1, D]

    # ── 1. Encode root (step k=0) ─────────────────────────────────────────
    # z[:,0,:] est déjà dans z_all — pas de second encodage
    z    = z_all[:, 0, :]                                          # [B, D]
    pi_logits, v_scalar_0, v_logits_0 = network.apply(mz_params, z, method=network.predict_full)

    total_pol_ex = jnp.zeros((B,), dtype=jnp.float32)
    total_val_ex = jnp.zeros((B,), dtype=jnp.float32)
    total_rew_ex = jnp.zeros((B,), dtype=jnp.float32)

    # k=0 : policy + value (no reward at root)
    pol_loss_0 = _policy_loss_per_example(pi_logits, batch["target_pol"][:, 0, :])
    val_loss_0 = _categorical_loss_per_example(
        v_logits_0, batch["target_val"][:, 0],
        cfg_model.num_value_bins, cfg_model.value_min, cfg_model.value_max,
    )
    total_pol_ex += pol_loss_0
    total_val_ex += val_loss_0

    probe_logits_0 = probe_heads.apply(prb_params, z)
    prb_loss_0, per_probe_0 = probe_loss(probe_logits_0, batch["probe_tgts"][:, 0, :])
    acc_per_probe_0 = probe_accuracy(probe_logits_0, batch["probe_tgts"][:, 0, :])

    # ── 2. Unroll K steps ─────────────────────────────────────────────────
    z_cur = z

    def unroll_step(z_in, k):
        action_onehot  = batch["action_seq"][:, k, :]          # [B, A]

        reward_pred, reward_logits, z_next = network.apply(
            mz_params, z_in, action_onehot,
            method=network.dynamics_full,
        )
        pi_k, v_k_scalar, v_k_logits = network.apply(mz_params, z_next, method=network.predict_full)

        pol_loss_k = _policy_loss_per_example(
            pi_k, batch["target_pol"][:, k + 1, :]
        )
        val_loss_k = _categorical_loss_per_example(
            v_k_logits, batch["target_val"][:, k + 1],
            cfg_model.num_value_bins, cfg_model.value_min, cfg_model.value_max,
        )
        rew_loss_k = _categorical_loss_per_example(
            reward_logits, batch["reward_seq"][:, k],
            cfg_model.num_value_bins, cfg_model.value_min, cfg_model.value_max,
        )

        # EfficientZero Consistency Loss (vectorisée) :
        # La cible est proj_all[:,k+1] — pré-calculée et stop_gradient avant le scan.
        # Plus de h() ni project_state() ici → économie de K forward passes.
        target_proj = proj_all[:, k + 1]       # [B, D]  — gratuit, déjà stop_gradient

        # Projection et prédiction du futur latent virtuel (côté prédiction, gradients actifs)
        pred_proj = network.apply(mz_params, z_next, method=network.project_state)
        pred_pred = network.apply(mz_params, pred_proj, method=network.predict_state)

        # Similarité cosinus négative
        eps = 1e-8
        pred_norm   = pred_pred   / (jnp.linalg.norm(pred_pred,   axis=-1, keepdims=True) + eps)
        target_norm = target_proj / (jnp.linalg.norm(target_proj, axis=-1, keepdims=True) + eps)
        consistency_loss_k = - jnp.sum(pred_norm * target_norm, axis=-1)

        return z_next, (pol_loss_k, val_loss_k, rew_loss_k, consistency_loss_k, pi_k, v_k_scalar, z_next)

    # Use lax.scan for unrolling — avoids Python-level loop in JIT
    _, (pol_losses, val_losses, rew_losses, consistency_losses, _, _, zs) = jax.lax.scan(
        lambda carry, k: unroll_step(carry, k),
        z_cur,
        jnp.arange(K),
    )
    # pol_losses / val_losses / rew_losses / consistency_losses : [K, B]

    total_pol_ex += jnp.mean(pol_losses, axis=0)
    total_val_ex += jnp.mean(val_losses, axis=0)
    total_rew_ex  = jnp.mean(rew_losses, axis=0)
    total_consistency_ex = jnp.mean(consistency_losses, axis=0)

    # ── 3. Probe losses (computed on root latent only to save compute) ────
    total_prb = prb_loss_0

    # ── 4. Aggregate with IS weights ─────────────────────────────────────
    # (weights already normalised in the buffer)
    w = batch["is_weights"]   # [B]

    pol_loss = _weighted_mean(total_pol_ex, w)
    val_loss = _weighted_mean(total_val_ex, w)
    rew_loss = _weighted_mean(total_rew_ex, w)
    consistency_loss = _weighted_mean(total_consistency_ex, w)

    total_loss = (
        cfg_train.policy_loss_weight * pol_loss
        + cfg_train.value_loss_weight  * val_loss
        + cfg_train.reward_loss_weight * rew_loss
        + cfg_train.probe_loss_weight  * total_prb
        + cfg_train.consistency_loss_weight * consistency_loss
    )

    # TD-error for priority update (absolute value error on value at root)
    td_error = jnp.abs(v_scalar_0 - batch["target_val"][:, 0])

    metrics = {
        "loss_total":  total_loss,
        "loss_policy": pol_loss,
        "loss_value":  val_loss,
        "loss_reward": rew_loss,
        "loss_probes": total_prb,
        "loss_consistency": consistency_loss,
        "td_error_mean": jnp.mean(td_error),
        "probe_per_task": per_probe_0,   # [NUM_PROBES]
        "probe_acc_per_task": acc_per_probe_0,  # [NUM_PROBES]
    }
    return total_loss, metrics, td_error



# ─────────────────────────────────────────────────────────────────────────────
# Per-term losses
# ─────────────────────────────────────────────────────────────────────────────
def _policy_loss(
    logits: jnp.ndarray,       # [B, A]
    target: jnp.ndarray,       # [B, A]  soft policy (MCTS visit counts)
) -> jnp.ndarray:
    """
    Cross-entropy between MCTS policy (target) and network policy (logits).
    KL(target ‖ network) ≡ -Σ target · log_softmax(logits)  + const.
    """
    return jnp.mean(_policy_loss_per_example(logits, target))


def _policy_loss_per_example(
    logits: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    # Keeping finite logits in this loss avoids the indeterminate 0 * -inf
    # when an action has zero target probability.  A genuinely bad (NaN)
    # activation is still caught by the train-step finite guard.
    log_pi = jax.nn.log_softmax(jnp.clip(logits, -1e4, 1e4), axis=-1)
    target_sum = jnp.sum(target, axis=-1, keepdims=True)
    safe_target_sum = jnp.where(target_sum > 0, target_sum, 1.0)
    target_norm = jnp.where(target_sum > 0, target / safe_target_sum, target)
    terms = jnp.where(target_norm != 0, target_norm * log_pi, 0.0)
    return -jnp.sum(terms, axis=-1)


def _scalar_to_categorical(
    target: jnp.ndarray,       # [...]
    num_bins: int = 51,
    v_min: float = -2.5,
    v_max: float = 2.5,
) -> jnp.ndarray:
    """
    Encodes continuous scalar targets into 2-hot soft probability distributions
    over `num_bins` discrete bins via linear interpolation (HL-Gauss style).
    """
    v_clipped = jnp.clip(target, v_min, v_max)
    step = (v_max - v_min) / (num_bins - 1)

    idx_float = (v_clipped - v_min) / step
    lower_idx = jnp.floor(idx_float).astype(jnp.int32)
    lower_idx = jnp.clip(lower_idx, 0, num_bins - 2)
    upper_idx = lower_idx + 1

    weight_upper = jnp.clip(idx_float - lower_idx.astype(jnp.float32), 0.0, 1.0)
    weight_lower = 1.0 - weight_upper

    lower_onehot = jax.nn.one_hot(lower_idx, num_bins)
    upper_onehot = jax.nn.one_hot(upper_idx, num_bins)

    return lower_onehot * weight_lower[..., None] + upper_onehot * weight_upper[..., None]


def _categorical_loss_per_example(
    logits: jnp.ndarray,       # [..., num_bins]
    target: jnp.ndarray,       # [...] scalar
    num_bins: int = 51,
    v_min: float = -2.5,
    v_max: float = 2.5,
) -> jnp.ndarray:
    """
    Cross-entropy loss between predicted categorical logits and 2-hot encoded targets.
    """
    labels = _scalar_to_categorical(target, num_bins, v_min, v_max)
    log_probs = jax.nn.log_softmax(jnp.clip(logits, -1e4, 1e4), axis=-1)
    return -jnp.sum(labels * log_probs, axis=-1)



def _value_loss(
    pred_logits: jnp.ndarray,   # [B, num_bins]
    target: jnp.ndarray,        # [B]
    num_bins: int = 31,
    v_min: float = -1.2,
    v_max: float = 1.2,
) -> jnp.ndarray:
    return jnp.mean(_categorical_loss_per_example(pred_logits, target, num_bins, v_min, v_max))


def _value_loss_per_example(
    pred_logits: jnp.ndarray,
    target: jnp.ndarray,
    num_bins: int = 31,
    v_min: float = -1.2,
    v_max: float = 1.2,
) -> jnp.ndarray:
    return _categorical_loss_per_example(pred_logits, target, num_bins, v_min, v_max)


def _reward_loss(
    pred_logits: jnp.ndarray,   # [B, num_bins]
    target: jnp.ndarray,        # [B]
    num_bins: int = 31,
    v_min: float = -1.2,
    v_max: float = 1.2,
) -> jnp.ndarray:
    return jnp.mean(_categorical_loss_per_example(pred_logits, target, num_bins, v_min, v_max))


def _reward_loss_per_example(
    pred_logits: jnp.ndarray,
    target: jnp.ndarray,
    num_bins: int = 31,
    v_min: float = -1.2,
    v_max: float = 1.2,
) -> jnp.ndarray:
    return _categorical_loss_per_example(pred_logits, target, num_bins, v_min, v_max)



def _weighted_mean(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    weights = weights.astype(values.dtype)
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Batch collation (numpy → JAX)
# ─────────────────────────────────────────────────────────────────────────────
def collate_batch(
    entries,        # List[ReplayEntry]
    is_weights,     # np.ndarray [B]
    num_unroll: int,
    max_actions: int,
) -> dict:
    """
    Transforme une liste de ReplayEntry en un batch JAX prêt pour la loss.
    Tous les champs sont stackés avec une dimension B en tête.
    Les obs sont aussi stackées → dict de [B, K+1, ...].
    """
    import numpy as np

    B = len(entries)
    K = num_unroll

    # ── Actions / rewards / targets ──────────────────────────────────────
    action_seq = np.stack([e.action_seq for e in entries])   # [B, K, A]
    reward_seq = np.stack([e.reward_seq for e in entries])   # [B, K]

    # target_pol / val : pad or truncate to K+1
    target_pol = np.zeros((B, K + 1, max_actions), dtype=np.float32)
    target_val = np.zeros((B, K + 1), dtype=np.float32)
    probe_tgts = np.full((B, K + 1, 11), -1, dtype=np.int32)

    for i, e in enumerate(entries):
        n = min(len(e.target_pol), K + 1)
        target_pol[i, :n] = e.target_pol[:n]
        n = min(len(e.target_val), K + 1)
        target_val[i, :n] = e.target_val[:n]
        n = min(len(e.probe_tgts), K + 1)
        probe_tgts[i, :n] = e.probe_tgts[:n]

    # ── Observations : stack chaque champ ─────────────────────────────────
    # Chaque entry.obs_seq est une list de K+1 dicts
    obs_keys = entries[0].obs_seq[0].keys() if entries[0].obs_seq else []
    obs_seq_batch = {}
    for key in obs_keys:
        # [B, K+1, *obs_shape]
        stacked = np.stack([
            np.stack([
                e.obs_seq[k][key] if k < len(e.obs_seq)
                else np.zeros_like(e.obs_seq[-1][key])
                for k in range(K + 1)
            ])
            for e in entries
        ])
        obs_seq_batch[key] = jnp.array(stacked)

    return {
        "obs_seq":    obs_seq_batch,
        "action_seq": jnp.array(action_seq),
        "reward_seq": jnp.array(reward_seq),
        "target_pol": jnp.array(target_pol),
        "target_val": jnp.array(target_val),
        "probe_tgts": jnp.array(probe_tgts),
        "is_weights": jnp.array(is_weights),
    }


def _slice_obs(obs_seq_batch: dict, k: int) -> dict:
    """Extrait le step k d'un obs batché [B, K+1, ...] → [B, ...]."""
    return {key: val[:, k] for key, val in obs_seq_batch.items()}
