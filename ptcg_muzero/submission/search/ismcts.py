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

import logging
from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import mctx

from config import Config, ModelConfig, SearchConfig

logger = logging.getLogger("ptcg_muzero.ismcts")

# AUDIT §3.5 — le NamedTuple `MuZeroRecurrentFn` et l'import `asdict` étaient
# inutilisés ; supprimés.


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
        z_next, reward = network.apply(
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
# ── AUDIT §2.1 : pool de cartes utilisé pour la déterminisation ───────────────
# L'ancienne version tirait la main adverse dans `arange(1, num_card_ids)`, soit
# les 1268 IDs du jeu, alors que les deux joueurs utilisent un deck FIXE de 60
# cartes.  La main adverse simulée était donc composée à ~100 % de cartes
# impossibles : le MCTS raisonnait sur des mondes irréalisables et — surtout —
# l'observation stockée dans le replay buffer (déterminisation n°0) entraînait
# h(s) sur ~5-8 tokens de bruit pur par pas, avec des features statiques (HP,
# type, dégâts) totalement hors distribution.
_BELIEF_DECK: list[int] | None = None      # None = non résolu, [] = désactivé
_BELIEF_DECK_EXPLICIT: bool = False


def set_belief_deck(card_ids) -> None:
    """Déclare le deck réellement joué par l'adversaire (multiset de card_ids).

    Passer ``None`` ou une liste vide rétablit l'ancien comportement (tirage
    dans tout le pool de cartes), à n'utiliser que si le deck adverse est
    réellement inconnu.
    """
    global _BELIEF_DECK, _BELIEF_DECK_EXPLICIT
    _BELIEF_DECK_EXPLICIT = True
    _BELIEF_DECK = list(card_ids) if card_ids else []


def _default_belief_deck() -> list[int] | None:
    global _BELIEF_DECK
    if _BELIEF_DECK is None and not _BELIEF_DECK_EXPLICIT:
        try:
            from models.deck_builder import DEFAULT_COMPETITIVE_DECK
            _BELIEF_DECK = list(DEFAULT_COMPETITIVE_DECK)
        except Exception:
            _BELIEF_DECK = []
    return _BELIEF_DECK or None


def _card_id_of(card) -> int:
    """ID d'une carte, quelle que soit sa forme renvoyée par le moteur.

    Le moteur peut exposer une carte comme un dict ``{"id": 96}``, comme un
    objet portant un attribut ``id``, ou directement comme l'entier ``96``
    (cf. le même traitement dans ``env/encoding.py``).  Sans ce dernier cas les
    cartes visibles n'étaient pas reconnues : elles n'étaient pas retirées du
    pool de déterminisation et pouvaient être « re-tirées » dans la main
    adverse simulée.
    """
    import numpy as _np

    if card is None:
        return 0
    if isinstance(card, bool):          # bool est un int en Python
        return 0
    if isinstance(card, (int, _np.integer)):
        return int(card)
    raw = card.get("id") if isinstance(card, dict) else getattr(card, "id", None)
    if raw is None:
        return 0
    if hasattr(raw, "value"):
        raw = raw.value
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _candidate_pool(known_opp_counts: dict, cfg: ModelConfig):
    """Cartes plausiblement dans la main adverse, en respectant les multiplicités.

    Retourne un ``np.ndarray`` d'IDs (avec répétitions) ou ``None`` si aucun deck
    de référence n'est connu (→ repli sur le pool complet).
    """
    import numpy as np
    from collections import Counter

    deck = _default_belief_deck()
    if not deck:
        return None
    remaining = Counter(int(c) for c in deck)
    remaining.subtract(known_opp_counts)
    pool = [cid for cid, n in remaining.items() if n > 0 for _ in range(n)]
    return np.array(pool, dtype=np.int32) if pool else None


def sample_belief(
    obs: dict,
    rng: jnp.ndarray,
    cfg: ModelConfig,
) -> dict:
    """
    Produce one determinised world by sampling a plausible opponent hand.

    Supports both raw Python environment dicts (from self-play workers)
    and pre-encoded numpy array dicts (from direct agent inference).
    """
    import numpy as np
    from collections import Counter

    is_raw_dict = isinstance(obs, dict) and ("current" in obs or "select" in obs)
    known_opp_counts: Counter = Counter()

    if is_raw_dict:
        # AUDIT §3.7 — `copy.deepcopy(obs)` était appelé num_belief_samples (4) ×
        # par décision × 8 workers sur des observations contenant défausse (60),
        # main, prizes et logs.  Seul `players[opp]["hand"]` est modifié : une
        # copie superficielle ciblée suffit.
        current = dict(obs.get("current") or {})
        your_idx = current.get("yourIndex", 0)
        players_src = current.get("players") or [{}, {}]
        players = [dict(p) if isinstance(p, dict) else p for p in players_src]
        current["players"] = players
        obs_out = dict(obs)
        obs_out["current"] = current

        me = players[your_idx] if (isinstance(players, list) and your_idx < len(players)) else {}
        opp = players[1 - your_idx] if (isinstance(players, list) and (1 - your_idx) < len(players)) else {}

        known_ids = set()
        for area_key in ("active", "bench", "hand", "discard", "prize"):
            items = me.get(area_key) or []
            if not isinstance(items, list):
                items = [items]
            for item in items:
                cid = _card_id_of(item)
                if cid > 0:
                    known_ids.add(cid)

        for area_key in ("active", "bench", "discard"):
            items = opp.get(area_key) or []
            if not isinstance(items, list):
                items = [items]
            for item in items:
                cid = _card_id_of(item)
                if cid > 0:
                    known_ids.add(cid)
                    known_opp_counts[cid] += 1

        opp_hand_count = int(opp.get("handCount", 0) or 0)
    else:
        obs_out = {k: v.copy() if hasattr(v, "copy") else v for k, v in obs.items()}
        known_ids = set()
        for field in ("my_hand_ids", "my_active_id", "my_bench_ids",
                      "my_discard_ids", "my_prize_ids",
                      "opp_active_id", "opp_bench_ids", "opp_discard_ids"):
            arr = obs.get(field)
            if arr is not None:
                known_ids.update(int(x) for x in arr.ravel() if x > 0)
        for field in ("opp_active_id", "opp_bench_ids", "opp_discard_ids"):
            arr = obs.get(field)
            if arr is not None:
                for x in arr.ravel():
                    if x > 0:
                        known_opp_counts[int(x)] += 1

        global_feat = obs.get("global_feat")
        if global_feat is not None:
            opp_hand_count = int(round(float(global_feat[11]) * 20.0))
        else:
            opp_hand_count = 0

    # AUDIT §2.1 : tirer dans le deck adverse connu, avec ses multiplicités.
    unseen = _candidate_pool(known_opp_counts, cfg)
    if unseen is None:
        all_ids = np.arange(1, cfg.num_card_ids, dtype=np.int32)
        unseen = all_ids[~np.isin(all_ids, list(known_ids))]

    if len(unseen) == 0 or opp_hand_count == 0:
        return obs_out

    try:
        seed = int(np.asarray(rng).ravel()[0])
    except (IndexError, TypeError, ValueError):
        seed = int(rng)
    rng_np = np.random.default_rng(seed % (2**31))
    sample_count = min(opp_hand_count, len(unseen), cfg.max_hand_size)
    # `replace=False` sur un pool contenant déjà les répétitions du deck :
    # on peut donc tirer 4 exemplaires d'une carte présente en 4 exemplaires.
    sampled = rng_np.choice(unseen, size=sample_count, replace=False)

    if is_raw_dict:
        players = obs_out.get("current", {}).get("players", [{}, {}])
        opp = players[1 - your_idx]
        opp["hand"] = [{"id": int(cid)} for cid in sampled]
    else:
        opp_hand = np.zeros(cfg.max_hand_size, dtype=np.int32)
        opp_hand_mask = np.zeros(cfg.max_hand_size, dtype=bool)
        n = min(len(sampled), cfg.max_hand_size)
        if n > 0:
            opp_hand[:n] = sampled[:n]
            opp_hand_mask[:n] = True
        obs_out["opp_hand_ids"] = opp_hand
        obs_out["opp_hand_mask"] = opp_hand_mask

    return obs_out


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet noise helper via exact Gamma sampling (JAX vectorisé)
# ─────────────────────────────────────────────────────────────────────────────
def inject_dirichlet_noise(
    prior_logits: jnp.ndarray,
    invalid_mask: jnp.ndarray,
    rng: jnp.ndarray,
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.3,
) -> jnp.ndarray:
    """
    Injecte un bruit de Dirichlet exact sur les actions légales via échantillonnage Gamma.
    - Exploite la propriété Gamma : Dirichlet(alpha) ~ Gamma(alpha, 1) / sum(Gamma sur actions légales)
    - Alpha adaptatif calibré sur le nombre d'actions légales : alpha = clip(10.0 / N_legal, 0.3, 5.0)
    - Garantit un epsilon effectif rigoureusement stable sur le sous-ensemble légal.
    """
    if dirichlet_epsilon <= 0.0:
        return jnp.where(invalid_mask, -1e9, prior_logits)

    legal_mask = ~invalid_mask
    num_legal = jnp.maximum(jnp.sum(legal_mask.astype(jnp.float32), axis=-1, keepdims=True), 1.0)
    alpha_adapt = jnp.clip(10.0 / num_legal, 0.3, 5.0)

    # Gamma sampling
    gamma_noise = jax.random.gamma(rng, alpha_adapt, shape=prior_logits.shape)
    gamma_masked = jnp.where(legal_mask, gamma_noise, 0.0)
    gamma_sum = jnp.maximum(jnp.sum(gamma_masked, axis=-1, keepdims=True), 1e-8)
    dirichlet_noise = gamma_masked / gamma_sum

    # Prior probabilities on legal actions
    legal_logits = jnp.where(invalid_mask, -1e9, prior_logits)
    prior_probs = jax.nn.softmax(legal_logits, axis=-1)
    noisy_probs = (1.0 - dirichlet_epsilon) * prior_probs + dirichlet_epsilon * dirichlet_noise

    # Return masked logits
    return jnp.where(invalid_mask, -1e9, jnp.log(jnp.maximum(noisy_probs, 1e-12)))


# ─────────────────────────────────────────────────────────────────────────────
# Main ISMCTS entry point
# ─────────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnames=("cfg_tuple", "num_simulations", "max_num_considered_actions", "dirichlet_epsilon", "dirichlet_alpha"))
def _run_batched_mcts_jit(
    params: dict,
    z: jnp.ndarray,
    pi_logits: jnp.ndarray,
    v: jnp.ndarray,
    invalid_mask: jnp.ndarray,
    rng: jnp.ndarray,
    static_features: jnp.ndarray,
    *,
    cfg_tuple: tuple,
    num_simulations: int,
    max_num_considered_actions: int,
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Exécute Gumbel MuZero MCTS compilé avec JAX JIT sur un batch de N_samples (belief samples).
    Retourne (action_weights [N_samples, A], node_values [N_samples]).
    """
    from models.networks import MuZeroNetwork
    model_cfg = ModelConfig(**dict(cfg_tuple))
    network = MuZeroNetwork(cfg=model_cfg, static_features=static_features)

    rng_mcts, rng_noise = jax.random.split(rng)
    masked_logits = inject_dirichlet_noise(
        pi_logits, invalid_mask, rng_noise, dirichlet_epsilon, dirichlet_alpha
    )

    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=v,
        embedding=z,
    )
    recurrent_fn = make_recurrent_fn(network, params)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_mcts,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        invalid_actions=invalid_mask,
    )
    return policy_output.action_weights, policy_output.search_tree.node_values[:, 0]


def ismcts_action(
    network,
    params: dict,
    obs: dict,          # single (unbatched) obs dict of numpy arrays
    option_mask_np,     # numpy [max_actions] bool
    rng: jnp.ndarray,
    cfg: Config,
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.3,
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

    # 3. Exécution MCTS JIT-compilée sur le batch de N_samples (ou direct si num_simulations <= 1)
    if int(sc.num_simulations) <= 1:
        masked_pi = jnp.where(mask_jax_batched, pi_logits, -1e9)
        probs_batched = jax.nn.softmax(masked_pi, axis=-1)
        avg_policy = jnp.mean(probs_batched, axis=0)
        avg_value = float(jnp.mean(v))
    else:
        invalid_mask = ~mask_jax_batched
        cfg_tuple = tuple(mc.__dict__.items())

        action_weights, node_values = _run_batched_mcts_jit(
            mz_params,
            z,
            pi_logits,
            v,
            invalid_mask,
            rng,
            network.static_features,
            cfg_tuple=cfg_tuple,
            num_simulations=int(sc.num_simulations),
            max_num_considered_actions=int(sc.max_num_considered_actions),
            dirichlet_epsilon=float(dirichlet_epsilon),
            dirichlet_alpha=float(dirichlet_alpha),
        )

        # 4. Moyenne sur l'axe des belief samples
        avg_policy = jnp.mean(action_weights, axis=0)  # [A]
        avg_value  = float(jnp.mean(node_values))

    # Appliquer le masque d'options légales
    avg_policy_masked = jnp.where(
        option_mask_jnp, avg_policy, -1e9
    )
    best_action = int(jnp.argmax(avg_policy_masked))

    return best_action, avg_policy, avg_value


# AUDIT §3.5 — `add_exploration_noise` (variante non batchée, importée mais
# jamais appelée) supprimée : `inject_dirichlet_noise` ci-dessus est la seule
# implémentation utilisée, et elle est batchée.


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
    Reanalyze In-Pipeline GPU : relance un MCTS Gumbel sur le batch
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
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.3,
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

    rng_mcts, rng_noise = jax.random.split(rng)
    masked_logits = inject_dirichlet_noise(
        pi_logits, invalid_mask, rng_noise, dirichlet_epsilon, dirichlet_alpha
    )

    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=v,
        embedding=z,
    )

    recurrent_fn = make_recurrent_fn(network, mz_params)

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
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.3,
):
    key = (
        id(network),
        num_simulations,
        max_num_considered_actions,
        num_belief_samples,
        max_actions,
        float(dirichlet_epsilon),
        float(dirichlet_alpha),
    )
    # AUDIT §3.2 — `id()` est recyclé après garbage collection : on conserve une
    # référence forte au module dans le cache pour que son identité reste unique
    # tant que l'entrée existe, et on revérifie l'identité à la lecture.
    cached = _ISMCTS_BATCHED_JIT_CACHE.get(key)
    compiled = None
    if cached is not None:
        cached_net, cached_fn = cached
        if cached_net is network:
            compiled = cached_fn
        else:
            _ISMCTS_BATCHED_JIT_CACHE.pop(key, None)
    if compiled is None:
        logger.info(
            "[ismcts_batched] JIT compilation ISMCTS: sims=%d, belief_samples=%d, max_considered=%d, dirichlet_eps=%.3f, dirichlet_alp=%.3f",
            num_simulations, num_belief_samples, max_num_considered_actions, dirichlet_epsilon, dirichlet_alpha
        )
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
                dirichlet_epsilon=dirichlet_epsilon,
                dirichlet_alpha=dirichlet_alpha,
            )

        _ISMCTS_BATCHED_JIT_CACHE[key] = (network, compiled)
    return compiled


def ismcts_action_batched(
    network,
    params: dict,
    batched_enc_obs: dict,
    option_masks_np,
    rng: jnp.ndarray,
    cfg: Config,
    dirichlet_epsilon: float | None = None,
    dirichlet_alpha: float | None = None,
) -> Tuple[Any, jnp.ndarray, Any]:
    """
    Runs MCTS on a batch of B games, each with S belief samples.
    JIT-compiled for GPU efficiency.
    """
    import numpy as np
    sc = cfg.search
    mc = cfg.model

    # Aligner tous les tenseurs sur le même device matériel que `params`
    target_dev = None
    if isinstance(params, dict):
        first_val = next((v for v in params.values() if hasattr(v, "devices")), None)
        if first_val and hasattr(first_val, "devices") and first_val.devices():
            target_dev = list(first_val.devices())[0]

    if target_dev:
        option_masks_jnp = jax.device_put(option_masks_np, target_dev)
        rng = jax.device_put(rng, target_dev)
    else:
        option_masks_jnp = jnp.array(option_masks_np)

    eps = float(sc.dirichlet_epsilon if dirichlet_epsilon is None else dirichlet_epsilon)
    alp = float(sc.dirichlet_alpha if dirichlet_alpha is None else dirichlet_alpha)

    compiled = _get_ismcts_action_batched_jit(
        network,
        int(sc.num_simulations),
        int(sc.max_num_considered_actions),
        int(sc.num_belief_samples),
        int(mc.max_actions),
        eps,
        alp,
    )
    best_actions, avg_policies, avg_values = compiled(
        params,
        batched_enc_obs,
        option_masks_jnp,
        rng,
    )

    return np.array(best_actions), np.array(avg_policies), np.array(avg_values)


