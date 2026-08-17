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

    # ── AUDIT §2.2 : masque de validité par pas de déroulement ───────────────
    # `collate_batch` complète les fenêtres tronquées (fins de partie) avec des
    # zéros.  Sans masque, la tête de valeur est entraînée à prédire 0 sur les
    # états terminaux (où la valeur vaut ±1) et le bonus d'entropie est gonflé
    # par des `option_mask` entièrement faux (softmax uniforme sur 128 actions).
    valid_seq = batch.get("valid_seq")
    if valid_seq is None:   # rétro-compatibilité avec d'anciens batches
        valid_seq = jnp.ones((B, K + 1), dtype=jnp.float32)
    valid_seq = valid_seq.astype(jnp.float32)

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

    mask_root = batch["obs_seq"]["option_mask"][:, 0] if "option_mask" in batch["obs_seq"] else jnp.ones_like(pi_logits, dtype=jnp.bool_)

    total_pol_ex = jnp.zeros((B,), dtype=jnp.float32)
    total_val_ex = jnp.zeros((B,), dtype=jnp.float32)
    total_rew_ex = jnp.zeros((B,), dtype=jnp.float32)

    # k=0 : policy + value (no reward at root) avec masquage strict des actions légales
    pol_loss_0 = _policy_loss_per_example(pi_logits, batch["target_pol"][:, 0, :], mask=mask_root)
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
        mask_k = batch["obs_seq"]["option_mask"][:, k + 1] if "option_mask" in batch["obs_seq"] else None
        # Validité du pas k+1 : concerne obs / value / policy / consistency,
        # qui portent tous sur l'ÉTAT s_{k+1}.
        valid_k = valid_seq[:, k + 1]                          # [B]
        # Validité de la récompense d'indice k.  `add_game` construit
        # `rew_window = rewards[t:end]` et `obs_window = obs_list[t:end+1]` :
        # reward_seq[k] est donc la récompense perçue AU pas de décision k,
        # pas au pas k+1.  Sa validité est celle de l'état s_k.
        #
        # Le masque `valid_seq[:, k+1]` utilisé auparavant décalait d'un cran et
        # supprimait systématiquement la DERNIÈRE récompense réelle de toute
        # fenêtre tronquée par la fin de partie — c'est-à-dire précisément la
        # récompense terminale ±1, seule récompense non nulle du jeu.
        # Conséquences : la tête de récompense n'apprenait que « 0 » et le terme
        # `partial_ret` du Reanalyze (trainer.py) était toujours nul, ce qui
        # ramenait la cible de valeur à un pur auto-bootstrap (cf. AUDIT §1.3).
        valid_r_k = valid_seq[:, k]                            # [B]

        reward_pred, reward_logits, z_next = network.apply(
            mz_params, z_in, action_onehot,
            method=network.dynamics_full,
        )
        pi_k, v_k_scalar, v_k_logits = network.apply(mz_params, z_next, method=network.predict_full)

        pol_loss_k = _policy_loss_per_example(
            pi_k, batch["target_pol"][:, k + 1, :], mask=mask_k
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

        # Entropie du réseau au pas k
        pi_k_masked = jnp.where(mask_k, pi_k, -1e9) if mask_k is not None else pi_k
        pi_k_probs = jax.nn.softmax(pi_k_masked, axis=-1)
        h_k_net = -jnp.sum(jnp.where(pi_k_probs > 0, pi_k_probs * jnp.log(jnp.maximum(pi_k_probs, 1e-12)), 0.0), axis=-1)

        # ── AUDIT §2.2 : neutraliser intégralement les pas paddés ────────────
        pol_loss_k         = pol_loss_k         * valid_k
        val_loss_k         = val_loss_k         * valid_k
        rew_loss_k         = rew_loss_k         * valid_r_k     # ← s_k, pas s_{k+1}
        consistency_loss_k = consistency_loss_k * valid_k
        h_k_net            = h_k_net            * valid_k

        return z_next, (pol_loss_k, val_loss_k, rew_loss_k, consistency_loss_k, h_k_net, pi_k, v_k_scalar, z_next)

    # Use lax.scan for unrolling — avoids Python-level loop in JIT
    _, (pol_losses, val_losses, rew_losses, consistency_losses, h_unroll_losses, _, _, zs) = jax.lax.scan(
        lambda carry, k: unroll_step(carry, k),
        z_cur,
        jnp.arange(K),
    )
    # pol_losses / val_losses / rew_losses / consistency_losses : [K, B]

    # Moyenne pondérée par la validité (et non moyenne brute sur K) : une fenêtre
    # dont seuls 2 pas sur 5 sont réels ne doit pas être diluée par 3 zéros.
    valid_unroll = valid_seq[:, 1:]                     # [B, K]  états s_1..s_K
    n_valid_ex   = jnp.maximum(jnp.sum(valid_unroll, axis=-1), 1.0)   # [B]
    # Les récompenses portent sur s_0..s_{K-1} : leur dénominateur diffère.
    valid_rew    = valid_seq[:, :K]                     # [B, K]
    n_valid_rew  = jnp.maximum(jnp.sum(valid_rew, axis=-1), 1.0)      # [B]

    total_pol_ex += jnp.sum(pol_losses, axis=0) / n_valid_ex
    total_val_ex += jnp.sum(val_losses, axis=0) / n_valid_ex
    total_rew_ex  = jnp.sum(rew_losses, axis=0) / n_valid_rew
    total_consistency_ex = jnp.sum(consistency_losses, axis=0) / n_valid_ex

    # ── 3. Probe losses (computed on root latent only to save compute) ────
    total_prb = prb_loss_0

    # ── 4. Aggregate with IS weights ─────────────────────────────────────
    # (weights already normalised in the buffer)
    w = batch["is_weights"]   # [B]

    pol_loss = _weighted_mean(total_pol_ex, w)
    val_loss = _weighted_mean(total_val_ex, w)
    rew_loss = _weighted_mean(total_rew_ex, w)
    consistency_loss = _weighted_mean(total_consistency_ex, w)

    # ── Métriques avancées de télémétrie (Entropie, Max Prob, Saturation Valeur) ──
    pi_masked_logits = jnp.where(mask_root, pi_logits, -1e9)
    pi_legal_probs = jax.nn.softmax(pi_masked_logits, axis=-1)

    # Entropie du réseau H(p) à la racine et étendue sur le déroulement
    h_network_per_ex = -jnp.sum(jnp.where(pi_legal_probs > 0, pi_legal_probs * jnp.log(jnp.maximum(pi_legal_probs, 1e-12)), 0.0), axis=-1)
    num_legal = jnp.maximum(jnp.sum(mask_root.astype(jnp.float32), axis=-1), 1.0)
    max_h = jnp.log(jnp.maximum(num_legal, 2.0))
    h_norm_per_ex = jnp.where(num_legal > 1.0, h_network_per_ex / max_h, 0.0)
    h_norm_per_ex = jnp.clip(h_norm_per_ex, 0.0, 1.0)

    # Entropie globale moyenne (root + unrolled steps) pour régularisation globale.
    # AUDIT §2.2 : h_unroll_losses est déjà masqué ; on renormalise par le nombre
    # de pas réellement valides au lieu de K.
    h_unroll_mean = jnp.sum(h_unroll_losses, axis=0) / n_valid_ex        # [B]
    h_net_extended = (jnp.mean(h_network_per_ex) + jnp.mean(h_unroll_mean)) / 2.0

    total_loss = (
        cfg_train.policy_loss_weight * pol_loss
        + cfg_train.value_loss_weight  * val_loss
        + cfg_train.reward_loss_weight * rew_loss
        + cfg_train.probe_loss_weight  * total_prb
        + cfg_train.consistency_loss_weight * consistency_loss
        - getattr(cfg_train, "policy_entropy_weight", 0.0) * h_net_extended
    )

    # TD-error for priority update (absolute value error on value at root)
    td_error = jnp.abs(v_scalar_0 - batch["target_val"][:, 0])

    # Entropie de la cible MCTS H(π)
    tgt_pol_0 = batch["target_pol"][:, 0, :]
    tgt_sum = jnp.maximum(jnp.sum(tgt_pol_0, axis=-1, keepdims=True), 1e-8)
    tgt_norm = tgt_pol_0 / tgt_sum
    h_mcts_per_ex = -jnp.sum(jnp.where(tgt_norm > 0, tgt_norm * jnp.log(jnp.maximum(tgt_norm, 1e-12)), 0.0), axis=-1)

    # Probabilité maximale moyenne (détection de collapse précoce)
    p_max_per_ex = jnp.max(pi_legal_probs, axis=-1)

    # Saturation de la valeur (|v| > 0.8)
    v_sat_per_ex = (jnp.abs(v_scalar_0) > 0.8).astype(jnp.float32)

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
        # ── Télémétrie Politique & Entropie ──
        "policy_entropy_norm": jnp.mean(h_norm_per_ex),
        "policy_entropy_net":  jnp.mean(h_network_per_ex),
        "policy_entropy_mcts": jnp.mean(h_mcts_per_ex),
        "policy_p_max_mean":   jnp.mean(p_max_per_ex),
        # ── Télémétrie Valeur ──
        "value_mean":          jnp.mean(v_scalar_0),
        "value_abs_mean":      jnp.mean(jnp.abs(v_scalar_0)),
        "value_saturation_pct": jnp.mean(v_sat_per_ex) * 100.0,
    }
    return total_loss, metrics, td_error



# ─────────────────────────────────────────────────────────────────────────────
# Per-term losses
# ─────────────────────────────────────────────────────────────────────────────
def _policy_loss_per_example(
    logits: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    # Keeping finite logits in this loss avoids the indeterminate 0 * -inf
    # when an action has zero target probability.  A genuinely bad (NaN)
    # activation is still caught by the train-step finite guard.
    if mask is not None:
        logits = jnp.where(mask, logits, -1e9)
    log_pi = jax.nn.log_softmax(jnp.clip(logits, -1e4, 1e4), axis=-1)
    target_sum = jnp.sum(target, axis=-1, keepdims=True)
    safe_target_sum = jnp.where(target_sum > 0, target_sum, 1.0)
    target_norm = jnp.where(target_sum > 0, target / safe_target_sum, target)
    terms = jnp.where(target_norm != 0, target_norm * log_pi, 0.0)
    return -jnp.sum(terms, axis=-1)


def _scalar_to_categorical(
    target: jnp.ndarray,       # [...]
    num_bins: int = 51,
    v_min: float = -1.8,
    v_max: float = 1.8,
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
    v_min: float = -1.8,
    v_max: float = 1.8,
) -> jnp.ndarray:
    """
    Cross-entropy loss between predicted categorical logits and 2-hot encoded targets.
    """
    labels = _scalar_to_categorical(target, num_bins, v_min, v_max)
    log_probs = jax.nn.log_softmax(jnp.clip(logits, -1e4, 1e4), axis=-1)
    return -jnp.sum(labels * log_probs, axis=-1)


# AUDIT §3.5 — les helpers morts `_policy_loss`, `_value_loss(_per_example)` et
# `_reward_loss(_per_example)` ont été supprimés : ils portaient des valeurs par
# défaut (num_bins=31, v_min=-2.5) incohérentes avec ModelConfig (51, ±1.2) et
# constituaient un piège pour toute réutilisation future.


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
    # AUDIT §2.2 : masque de validité — 1.0 tant que le pas correspond à une
    # vraie observation de la partie, 0.0 dès qu'il s'agit de padding.
    valid_seq  = np.zeros((B, K + 1), dtype=np.float32)

    for i, e in enumerate(entries):
        n = min(len(e.target_pol), K + 1)
        target_pol[i, :n] = e.target_pol[:n]
        n = min(len(e.target_val), K + 1)
        target_val[i, :n] = e.target_val[:n]
        n = min(len(e.probe_tgts), K + 1)
        probe_tgts[i, :n] = e.probe_tgts[:n]
        # La longueur qui fait foi est celle de la séquence d'observations :
        # `obs_seq` / `target_val` / `target_pol` sont découpés sur la même fenêtre.
        n_obs = min(len(e.obs_seq), len(e.target_val), K + 1)
        valid_seq[i, :max(n_obs, 1)] = 1.0   # la racine est toujours valide

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
        "valid_seq":  jnp.array(valid_seq),
    }
