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
    network,          # MuZeroNetwork Flax module (bound)
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

        # g: dynamics
        reward, z_next = network.apply(
            params, embedding, action_onehot,
            method=network.dynamics,
        )

        # f: prediction from next state
        pi_logits, value = network.apply(
            params, z_next,
            method=network.predict,
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
@partial(jax.jit, static_argnames=("network", "cfg_model", "cfg_search"))
def _run_single_mcts(
    params: dict,
    root_embedding: jnp.ndarray,   # [1, D]
    prior_logits: jnp.ndarray,     # [1, A]
    root_value: jnp.ndarray,       # [1]
    option_mask: jnp.ndarray,      # [1, A] bool
    rng: jnp.ndarray,
    *,
    network,
    cfg_model: ModelConfig,
    cfg_search: SearchConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Gumbel MuZero search for one determinised world.
    Returns (policy [1,A], value [1]).
    """
    recurrent_fn = make_recurrent_fn(network, params, cfg_model)

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
        num_simulations=cfg_search.num_simulations,
        max_num_considered_actions=cfg_search.max_num_considered_actions,
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
    B   = 1  # single game, not batched here

    all_policies = []
    all_values   = []

    # Batch observation into shape [1, ...] for JAX
    def _to_jax_batch(d: dict) -> dict:
        return {k: jnp.array(v[None]) for k, v in d.items()}

    for s in range(sc.num_belief_samples):
        rng, rng_belief, rng_mcts = jax.random.split(rng, 3)

        # 1. Determinise
        det_obs = sample_belief(obs, rng_belief, mc)
        jax_obs = _to_jax_batch(det_obs)

        # 2. Encode root state
        z, pi_logits, v = network.apply(params, jax_obs)  # [1,D], [1,A], [1]

        # 3. Mask and run MCTS
        mask_jax = jnp.array(option_mask_np[None])   # [1, A]

        action_weights, node_val = _run_single_mcts(
            params,
            z,
            pi_logits,
            v,
            mask_jax,
            rng_mcts,
            network=network,
            cfg_model=mc,
            cfg_search=sc,
        )

        all_policies.append(action_weights[0])    # [A]
        all_values.append(float(node_val[0, 0]))

    # 4. Average across belief samples
    avg_policy = jnp.mean(jnp.stack(all_policies), axis=0)    # [A]
    avg_value  = float(np.mean(all_values))

    # Mask then argmax
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
