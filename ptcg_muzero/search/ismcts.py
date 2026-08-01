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
from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import mctx

from config import Config, ModelConfig, SearchConfig
from dataclasses import asdict

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
    network,
    params: dict,
) -> Callable:
    """
    Return a ``recurrent_fn`` in the mctx format:

        (params, rng_key, action, embedding) → RecurrentFnOutput, new_embedding

    ``embedding`` here is the latent state z of shape [B, latent_dim].
    ``action`` is an int32 scalar per sample in [0, max_actions).
    """
    num_actions = network.cfg.max_actions

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
        reward, z_next = network.apply(
            params, embedding, action_onehot, method=network.dynamics
        )

        # f: prediction from next state
        pi_logits, value = network.apply(
            params, z_next, method=network.predict
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

    if hasattr(rng, "__len__") or hasattr(rng, "shape"):
        seed = int(rng[0])
    else:
        seed = int(rng)
    rng_np = np.random.default_rng(seed % (2**31))
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
@partial(jax.jit, static_argnames=("cfg_tuple", "max_actions", "num_simulations", "max_num_considered_actions"))
def _run_single_mcts(
    params: dict,
    root_embedding: jnp.ndarray,   # [1, D]
    prior_logits: jnp.ndarray,     # [1, A]
    root_value: jnp.ndarray,       # [1]
    option_mask: jnp.ndarray,      # [1, A] bool
    rng: jnp.ndarray,
    static_features: jnp.ndarray,
    *,
    cfg_tuple: tuple,
    max_actions: int,
    num_simulations: int,
    max_num_considered_actions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Gumbel MuZero search for one determinised world.
    Returns (policy [1,A], value [1]).
    """
    from models.networks import MuZeroNetwork
    model_cfg = ModelConfig(**dict(cfg_tuple))
    network = MuZeroNetwork(cfg=model_cfg, static_features=static_features)
    recurrent_fn = make_recurrent_fn(network, params)

    # Mask illegal actions: set logits of invalid options to -1e9
    invalid_mask = ~option_mask           # True where action is INVALID
    masked_logits = jnp.where(invalid_mask, -1e9, prior_logits)

    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=jnp.atleast_1d(jnp.squeeze(root_value)),   # guarantee shape [B] for mctx
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

    # Extract only MuZero parameters (the full parameter dict also contains "probes")
    mz_params = params["muzero"] if isinstance(params, dict) and "muzero" in params else params

    # 2. Encoder l'état racine pour toutes les hypothèses en parallèle
    # z: [N_samples, D], pi_logits: [N_samples, A], v: [N_samples, 1]
    z, pi_logits, v = network.apply(mz_params, jax_obs_batched)

    option_mask_jnp = jnp.array(option_mask_np)
    mask_jax_batched = jnp.stack([option_mask_jnp] * N_samples, axis=0)

    # 3. Exécution MCTS nativement sur le batch de N_samples (sans vmap)
    invalid_mask = ~mask_jax_batched
    masked_logits = jnp.where(invalid_mask, -1e9, pi_logits)

    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=v,           # [N_samples]
        embedding=z,       # [N_samples, D]
    )

    recurrent_fn = make_recurrent_fn(network, mz_params)

    policy_output = mctx.gumbel_muzero_policy(
        params=mz_params,
        rng_key=rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=int(sc.num_simulations),
        max_num_considered_actions=int(sc.max_num_considered_actions),
        invalid_actions=invalid_mask,
    )

    # 4. Moyenne sur l'axe des belief samples
    avg_policy = jnp.mean(policy_output.action_weights, axis=0)  # [A]
    avg_value  = float(jnp.mean(policy_output.search_tree.node_values[:, 0]))

    # Appliquer le masque d'options légales
    avg_policy_masked = jnp.where(
        option_mask_jnp, avg_policy, -1e9
    )
    best_action = int(jnp.argmax(avg_policy_masked))

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


def reanalyze_root(
    params: dict,
    network,
    z: jnp.ndarray,                    # [B, D]   latent states déjà encodés
    pi_logits: jnp.ndarray,            # [B, A]
    v: jnp.ndarray,                    # [B]
    option_mask: jnp.ndarray,          # [B, A]   bool
    rng: jnp.ndarray,                  # PRNGKey
    num_simulations: int,
    max_num_considered_actions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reanalyze In-Pipeline GPU : relance un MCTS Gumbel allégé sur le batch
    root entier en utilisant les paramètres courants du réseau.

    Contrairement à ismcts_action_batched (self-play) qui utilise vmap parce
    qu'il gère B×S mondes indépendants (belief samples), ici il y a exactement
    un monde par exemple → mctx.gumbel_muzero_policy accepte [B] nativement,
    sans vmap.  Cela s'intègre directement dans la closure pmap de train_step.

    Paramètres
    ----------
    params  : uniquement les params MuZero (pas "probes")
    z       : latent states racine [B, D], déjà calculés par network.represent
    pi_logits : logits de politique [B, A], déjà calculés par network.predict
    v       : valeurs racine [B], déjà calculées par network.predict
    option_mask : masque des actions légales [B, A] bool
    rng     : clé PRNG (doit être unique par device dans pmap)

    Retourne
    --------
    fresh_target_pol : [B, A]  nouvelles politiques cibles (action_weights MCTS)
    fresh_target_val : [B]     nouvelles valeurs cibles (valeur de la racine MCTS)
    """
    recurrent_fn = make_recurrent_fn(network, params)

    invalid_mask = ~option_mask
    masked_logits = jnp.where(invalid_mask, -1e9, pi_logits)

    # mctx attend value: [B], embedding: [B, D], prior_logits: [B, A] — batché nativement
    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=v,           # [B]   — pas de [None] nécessaire ici
        embedding=z,       # [B, D]
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

    fresh_pol = policy_output.action_weights              # [B, A]
    fresh_val = policy_output.search_tree.node_values[:, 0]  # [B]  valeur à la racine
    return fresh_pol, fresh_val



def _ismcts_action_batched_impl(
    params: dict,
    batched_enc_obs: dict,
    option_masks: jnp.ndarray,
    rng: jnp.ndarray,
    network: Any,
    num_simulations: int,
    max_num_considered_actions: int,
    num_belief_samples: int,
    max_actions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B = option_masks.shape[0]
    S = num_belief_samples

    mz_params = params["muzero"] if isinstance(params, dict) and "muzero" in params else params

    flat_obs = {}
    for k, v in batched_enc_obs.items():
        shape = v.shape
        flat_obs[k] = jnp.reshape(v, (B * S,) + shape[2:])

    # GPU encoder forward pass
    z, pi_logits, v = network.apply(mz_params, flat_obs)

    mask_jax_batched = jnp.repeat(option_masks, S, axis=0)
    invalid_mask = ~mask_jax_batched
    masked_logits = jnp.where(invalid_mask, -1e9, pi_logits)

    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=v,
        embedding=z,
    )

    recurrent_fn = make_recurrent_fn(network, mz_params)
    rng_mcts, _ = jax.random.split(rng)

    policy_output = mctx.gumbel_muzero_policy(
        params=mz_params,
        rng_key=rng_mcts,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        invalid_actions=invalid_mask,
    )

    action_weights_batched = policy_output.action_weights
    node_val_batched = policy_output.search_tree.node_values[:, 0]

    policies = jnp.reshape(action_weights_batched, (B, S, max_actions))
    values = jnp.reshape(node_val_batched, (B, S))

    avg_policies = jnp.mean(policies, axis=1)
    avg_values = jnp.mean(values, axis=1)

    avg_policies_masked = jnp.where(option_masks, avg_policies, -1e9)
    best_actions = jnp.argmax(avg_policies_masked, axis=-1)

    return best_actions, avg_policies, avg_values


# A Flax module can hold JAX arrays (``static_features`` in MuZeroNetwork).
# Consequently it is not hashable and cannot be a ``static_argname`` of a
# jitted function.  Keep the module in the closure instead, and cache the
# resulting compiled callable for the lifetime of the worker.
_ISMCTS_BATCHED_JIT_CACHE = {}


def _get_ismcts_action_batched_jit(
    network: Any,
    num_simulations: int,
    max_num_considered_actions: int,
    num_belief_samples: int,
    max_actions: int,
):
    key = (
        id(network),
        num_simulations,
        max_num_considered_actions,
        num_belief_samples,
        max_actions,
    )
    compiled = _ISMCTS_BATCHED_JIT_CACHE.get(key)
    if compiled is None:
        @jax.jit
        def compiled(params, batched_enc_obs, option_masks, rng):
            return _ismcts_action_batched_impl(
                params,
                batched_enc_obs,
                option_masks,
                rng,
                network=network,
                num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
                num_belief_samples=num_belief_samples,
                max_actions=max_actions,
            )

        _ISMCTS_BATCHED_JIT_CACHE[key] = compiled
    return compiled


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
    JIT-compiled for GPU efficiency.
    """
    import numpy as np
    sc = cfg.search
    mc = cfg.model

    option_masks_jnp = jnp.array(option_masks_np)

    compiled = _get_ismcts_action_batched_jit(
        network,
        int(sc.num_simulations),
        int(sc.max_num_considered_actions),
        int(sc.num_belief_samples),
        int(mc.max_actions),
    )
    best_actions, avg_policies, avg_values = compiled(
        params,
        batched_enc_obs,
        option_masks_jnp,
        rng,
    )

    return np.array(best_actions), np.array(avg_policies), np.array(avg_values)

