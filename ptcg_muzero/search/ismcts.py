"""
ptcg_muzero/search/ismcts.py
==============================
ISMCTS hybride pour le PTCG Pokémon.

Stratégie en deux couches
--------------------------
1. **Belief sampling (info imparfaite)**
   Avant chaque appel à mctx on génère `num_belief_samples` hypothèses
   sur la main/deck adverses (déterminisation).  On lance un arbre MCTS
   séparé sur chaque monde, puis on agrège les politiques par moyenne.

2. **Chance nodes (stochasticité)**
   Le réseau de dynamique ``g`` gère déjà les transitions stochastiques
   par l'approximation collapsed-expectation (cf. networks.py).  Côté
   mctx on utilise ``gumbel_muzero_policy`` qui est compatible avec des
   valeurs continues — aucune modification nécessaire.

Point d'entrée principal : ``ismcts_action``
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import mctx

from config import Config, ModelConfig, SearchConfig

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

class MuZeroRecurrentFn(NamedTuple):
    """Wraps g+f into the signature expected by mctx."""
    dynamics_fn:   Callable   # (params, rng, z, action) → (reward, z_next)
    prediction_fn: Callable   # (params, z)               → (pi_logits, value)


# ─────────────────────────────────────────────────────────────────────────────
# mctx recurrent function
# ─────────────────────────────────────────────────────────────────────────────
def make_recurrent_fn(
    dynamics_fn: Callable,   # network.dynamics bound method
    predict_fn: Callable,    # network.predict bound method
    params: dict,
    cfg: ModelConfig,
) -> Callable:
    """
    Return a ``recurrent_fn`` in the mctx format:

        (params, rng_key, action, embedding) → RecurrentFnOutput, new_embedding

    ``embedding`` here is the latent state z of shape [B, latent_dim].
    ``action`` is an int32 scalar per sample in [0, max_actions).
    """
    num_actions = cfg.max_actions

    def recurrent_fn(
        params_unused,   # mctx passes params through; we use closure
        rng_key: jnp.ndarray,
        action: jnp.ndarray,          # [B] int32
        embedding: jnp.ndarray,        # [B, latent_dim]
    ) -> Tuple[mctx.RecurrentFnOutput, jnp.ndarray]:
        B = embedding.shape[0]

        # One-hot encode action
        action_onehot = jax.nn.one_hot(action, num_actions)   # [B, A]

        # g: dynamics (using dynamics_fn directly with apply syntax wrapper)
        reward, z_next = dynamics_fn(
            params, embedding, action_onehot
        )

        # f: prediction from next state
        pi_logits, value = predict_fn(
            params, z_next
        )

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,          # [B]
            discount=jnp.full((B,), 0.997),
            prior_logits=pi_logits, # [B, A]
            value=value,            # [B]
        )
        return recurrent_output, z_next

    return recurrent_fn


# ─────────────────────────────────────────────────────────────────────────────
# Belief model (déterminisation de la main adverse)
# ─────────────────────────────────────────────────────────────────────────────
def sample_belief(
    obs: dict,
    rng: jnp.ndarray,
    cfg: ModelConfig,
) -> dict:
    """
    Produce one determinised world by sampling a plausible opponent hand.

    Strategy:
      - We know ``opp_discard_ids`` (face-up) and our own ``my_hand_ids``.
      - The remaining unseen cards (opponent hand + deck) are sampled
        uniformly from the complement.
      - ``opp_hand_count`` tells us how many cards to assign to the hand.
      - This is a simple uniform prior; a learned belief model can replace
        this function later without changing the rest of the pipeline.

    Returns a copy of ``obs`` with ``opp_bench_ids`` / ``opp_hand_ids``
    fields filled in from the sampled world.
    """
    # We work in numpy-land here (called outside JAX traced code)
    import numpy as np

    obs_out = {k: v.copy() if hasattr(v, "copy") else v for k, v in obs.items()}

    # Cards already accounted for (visible)
    known_ids = set()
    for field in ("my_hand_ids", "my_active_id", "my_bench_ids",
                  "my_discard_ids", "my_prize_ids",
                  "opp_active_id", "opp_bench_ids", "opp_discard_ids"):
        arr = obs.get(field)
        if arr is not None:
            known_ids.update(int(x) for x in arr.ravel() if x > 0)

    # Universe: all card IDs 1..num_card_ids-1
    all_ids = np.arange(1, cfg.num_card_ids, dtype=np.int32)
    unseen  = all_ids[~np.isin(all_ids, list(known_ids))]

    opp_hand_count = int(obs.get("global_feat", np.zeros(12))[11] * 20)
    opp_hand_count = min(max(opp_hand_count, 0), cfg.max_hand_size)

    if len(unseen) == 0 or opp_hand_count == 0:
        return obs_out

    rng_np = np.random.default_rng(int(rng[0]) % (2**31))
    sample_count = min(opp_hand_count, len(unseen))
    sampled = rng_np.choice(unseen, size=sample_count, replace=False)

    opp_hand = np.zeros(cfg.max_hand_size, dtype=np.int32)
    opp_hand_mask = np.zeros(cfg.max_hand_size, dtype=bool)
    opp_hand[:len(sampled)] = sampled
    opp_hand_mask[:len(sampled)] = True
    # Inject into obs (used by the encoder to build the opponent hand token)
    obs_out["opp_hand_ids"] = opp_hand
    obs_out["opp_hand_mask"] = opp_hand_mask
    return obs_out


# ─────────────────────────────────────────────────────────────────────────────
# Main ISMCTS entry point
# ─────────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnames=("dynamics_fn", "predict_fn", "max_actions", "num_simulations", "max_num_considered_actions"))
def _run_single_mcts(
    params: dict,
    root_embedding: jnp.ndarray,   # [1, D]
    prior_logits: jnp.ndarray,     # [1, A]
    root_value: jnp.ndarray,       # [1]
    option_mask: jnp.ndarray,      # [1, A] bool
    rng: jnp.ndarray,
    *,
    dynamics_fn: Callable,
    predict_fn: Callable,
    max_actions: int,
    num_simulations: int,
    max_num_considered_actions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Gumbel MuZero search for one determinised world.
    Returns (policy [1,A], value [1]).
    """
    # Create a dummy container for max_actions to satisfy make_recurrent_fn
    class _DummyCfg:
        def __init__(self, ma):
            self.max_actions = ma
    
    recurrent_fn = make_recurrent_fn(dynamics_fn, predict_fn, params, _DummyCfg(max_actions))

    # Mask illegal actions: set logits of invalid options to -1e9
    invalid_mask = ~option_mask           # True where action is INVALID
    masked_logits = jnp.where(invalid_mask, -1e9, prior_logits)

    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=root_value,
        embedding=root_embedding,
    )

    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        invalid_actions=invalid_mask,
    )

    return policy_output.action_weights, policy_output.search_tree.node_values[:, 0:1].T


def ismcts_action(
    network,
    params: dict,
    obs: dict,          # single (unbatched) obs dict of numpy arrays
    option_mask_np,     # numpy [max_actions] bool
    rng: jnp.ndarray,
    cfg: Config,
) -> Tuple[int, jnp.ndarray, float]:
    """
    Full ISMCTS decision:
      1. For each belief sample → determinise obs → encode → run Gumbel MuZero
      2. Average action_weights across samples
      3. Return (best_action_index, avg_policy, avg_value)

    Returns
    -------
    best_action : int
    avg_policy  : jnp.ndarray [max_actions]
    avg_value   : float
    """
    import numpy as np

    sc  = cfg.search
    mc  = cfg.model
    N_samples = int(sc.num_belief_samples)

    # 1. Générer toutes les déterminisations (belief samples) sur CPU
    rng, *rng_beliefs = jax.random.split(rng, N_samples + 1)
    rng_mcts_keys = jax.random.split(rng, N_samples)
    
    det_list = []
    for s in range(N_samples):
        det_list.append(sample_belief(obs, rng_beliefs[s], mc))

    # Assembler le batch d'observations JAX de taille [N_samples, ...]
    jax_obs_batched = {}
    for k in det_list[0].keys():
        jax_obs_batched[k] = jnp.stack([d[k] for d in det_list], axis=0)

    # 2. Encoder l'état racine pour toutes les hypothèses en parallèle
    # z: [N_samples, D], pi_logits: [N_samples, A], v: [N_samples, 1]
    z, pi_logits, v = network.apply(params, jax_obs_batched)

    # Préparer les masques d'option de taille [N_samples, A]
    mask_jax_batched = jnp.stack([jnp.array(option_mask_np)] * N_samples, axis=0)

    # 3. Définir les fonctions lambda stables
    dyn_fn = lambda p, z, a: network.apply(p, z, a, method=network.dynamics)
    prd_fn = lambda p, z: network.apply(p, z, method=network.predict)

    # 4. Vectoriser _run_single_mcts sur la dimension batch (axis 0)
    # JAX va exécuter toutes les simulations MCTS en parallèle sur le GPU
    vmapped_mcts = jax.vmap(
        lambda z_item, logits_item, v_item, mask_item, rng_item: _run_single_mcts(
            params,
            z_item[None],
            logits_item[None],
            v_item,
            mask_item[None],
            rng_item,
            dynamics_fn=dyn_fn,
            predict_fn=prd_fn,
            max_actions=int(mc.max_actions),
            num_simulations=int(sc.num_simulations),
            max_num_considered_actions=int(sc.max_num_considered_actions),
        ),
        in_axes=(0, 0, 0, 0, 0)
    )

    # Exécution vectorisée
    action_weights_batched, node_val_batched = vmapped_mcts(
        z, pi_logits, v, mask_jax_batched, rng_mcts_keys
    )

    # 5. Moyenne sur l'axe des belief samples
    avg_policy = jnp.mean(action_weights_batched[:, 0, :], axis=0)  # [A]
    avg_value  = float(jnp.mean(node_val_batched[:, 0, 0]))

    # Appliquer le masque d'options légales
    avg_policy = jnp.where(
        jnp.array(option_mask_np), avg_policy, -1e9
    )
    best_action = int(jnp.argmax(avg_policy))

    return best_action, avg_policy, avg_value


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet noise injection at root (self-play only)
# ─────────────────────────────────────────────────────────────────────────────
def add_exploration_noise(
    policy_logits: jnp.ndarray,   # [A]
    option_mask: jnp.ndarray,     # [A] bool
    rng: jnp.ndarray,
    alpha: float = 0.3,
    epsilon: float = 0.25,
) -> jnp.ndarray:
    """
    Inject Dirichlet noise à la AlphaZero over legal actions only.
    """
    num_legal = jnp.sum(option_mask).astype(jnp.int32)
    noise = jax.random.dirichlet(rng, alpha * jnp.ones(policy_logits.shape))
    noisy = (1 - epsilon) * policy_logits + epsilon * noise
    return jnp.where(option_mask, noisy, policy_logits)


def ismcts_action_batched(
    network,
    params: dict,
    batched_enc_obs: dict,
    option_masks_np,
    rng: jnp.ndarray,
    cfg: Config,
) -> Tuple[Any, jnp.ndarray, Any]:
    """
    Runs MCTS on a batch of B games, each with S belief samples.
    - batched_enc_obs: dict of JAX arrays, shape [B, S, ...]
    - option_masks_np: array of shape [B, A]
    """
    import numpy as np
    sc = cfg.search
    mc = cfg.model
    B = option_masks_np.shape[0]
    S = int(sc.num_belief_samples)

    flat_obs = {}
    for k, v in batched_enc_obs.items():
        shape = v.shape
        flat_obs[k] = jnp.reshape(v, (B * S,) + shape[2:])

    # GPU encoder forward pass
    z, pi_logits, v = network.apply(params, flat_obs)

    option_masks_jnp = jnp.array(option_masks_np)
    mask_jax_batched = jnp.repeat(option_masks_jnp, S, axis=0)

    rng, rng_mcts = jax.random.split(rng)
    rng_mcts_keys = jax.random.split(rng_mcts, B * S)

    dyn_fn = lambda p, z, a: network.apply(p, z, a, method=network.dynamics)
    prd_fn = lambda p, z: network.apply(p, z, method=network.predict)

    vmapped_mcts = jax.vmap(
        lambda z_item, logits_item, v_item, mask_item, rng_item: _run_single_mcts(
            params,
            z_item[None],
            logits_item[None],
            v_item,
            mask_item[None],
            rng_item,
            dynamics_fn=dyn_fn,
            predict_fn=prd_fn,
            max_actions=int(mc.max_actions),
            num_simulations=int(sc.num_simulations),
            max_num_considered_actions=int(sc.max_num_considered_actions),
        ),
        in_axes=(0, 0, 0, 0, 0)
    )

    action_weights_batched, node_val_batched = vmapped_mcts(
        z, pi_logits, v, mask_jax_batched, rng_mcts_keys
    )

    policies = jnp.reshape(action_weights_batched[:, 0, :], (B, S, int(mc.max_actions)))
    values = jnp.reshape(node_val_batched[:, 0, 0], (B, S))

    avg_policies = jnp.mean(policies, axis=1)
    avg_values = jnp.mean(values, axis=1)

    avg_policies_masked = jnp.where(
        option_masks_jnp, avg_policies, -1e9
    )
    best_actions = np.array(jnp.argmax(avg_policies_masked, axis=-1))

    return best_actions, avg_policies, np.array(avg_values)

