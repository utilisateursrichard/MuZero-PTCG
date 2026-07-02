"""
ptcg_muzero/models/deck_builder.py
=====================================
Réseau de construction de deck, entraîné par REINFORCE (gradient de politique).

Architecture
------------
Le deck builder est un réseau séparé dont les paramètres vivent dans un
``TrainState`` distinct du réseau MuZero principal.

Input  : embedding global de la bibliothèque de cartes disponibles
         → vecteur de contexte (optionnel, ex. stats de victoire récentes)
Output : distribution Categorical sur les 60 positions du deck,
         chaque position choisissant un card_id parmi les N cartes connues.

Contraintes réglementaires PTCG :
  - Exactement 60 cartes par deck.
  - Maximum 4 exemplaires d'une même carte non-Energy.
  - Minimum 1 carte Pokémon de base.
  Ces contraintes sont appliquées par rejection sampling / masquage.

Entraînement
------------
Le signal de récompense du deck est la valeur moyenne des parties jouées
avec ce deck.  On utilise REINFORCE avec une baseline EMA pour réduire la
variance :

    ∇J = (R - b) · ∇ log π(deck)
    b  ← β·b + (1-β)·R            (EMA de baseline)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from config import Config, ModelConfig, TrainConfig

# ── Constantes PTCG ──────────────────────────────────────────────────────────
DECK_SIZE         = 60
MAX_COPIES_NORMAL = 4   # cartes non-énergie de base
MAX_COPIES_ENERGY = 60  # pas de limite sur les énergies de base

# IDs d'énergie de base (rows Category == "Energy" dans le CSV)
# Ces IDs sont remplis au démarrage par ``set_energy_ids``
_BASIC_ENERGY_IDS: List[int] = []

# IDs des cartes Ace Spec (max 1 exemplaire par deck)
# Ces IDs sont remplis au démarrage par ``set_ace_spec_ids``
_ACE_SPEC_IDS: List[int] = []


def set_energy_ids(ids: List[int]) -> None:
    """Appelé une fois après parsing du CSV."""
    global _BASIC_ENERGY_IDS
    _BASIC_ENERGY_IDS = list(ids)


def set_ace_spec_ids(ids: List[int]) -> None:
    """Appelé une fois après parsing du CSV."""
    global _ACE_SPEC_IDS
    _ACE_SPEC_IDS = list(ids)


# ─────────────────────────────────────────────────────────────────────────────
# Réseau
# ─────────────────────────────────────────────────────────────────────────────
class DeckBuilderNetwork(nn.Module):
    """
    Produit un log-prob sur les card_ids pour chaque slot du deck.

    Le réseau génère un vecteur de logits de taille ``num_card_ids``
    partagé pour tous les slots (approche bag-of-cards simplifiée), puis
    applique le masquage de contraintes dans ``sample_deck``.
    """
    cfg: ModelConfig
    static_features: jnp.ndarray   # frozen [num_card_ids, CARD_STATIC_DIM]

    @nn.compact
    def __call__(
        self, context: jnp.ndarray | None = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            context: optional float32 [B, context_dim] conditioning vector
                     (e.g. win-rate history per card).  Pass None to ignore.
        Returns:
            logits: [B, num_card_ids]  unnormalised card-selection scores
            probs:  [B, num_card_ids]  normalised card-selection probabilities
        """
        D   = self.cfg.latent_dim
        N   = self.cfg.num_card_ids
        S   = self.static_features.shape[-1]  # CARD_STATIC_DIM

        # Embed every known card
        card_emb   = nn.Embed(num_embeddings=N, features=D)(
            jnp.arange(N)
        )                                          # [N, D]
        static_emb = nn.Dense(D)(self.static_features)  # [N, D]
        card_repr  = card_emb + static_emb         # [N, D]

        # Optional context injection
        if context is not None:
            ctx_proj = nn.Dense(D)(context)        # [B, D]
            # Broadcast over cards: [B, N, D]
            card_repr_b = card_repr[None] + ctx_proj[:, None, :]
        else:
            # No batch dimension on input — add one
            card_repr_b = card_repr[None]          # [1, N, D]

        # Attention-pool: query = learned deck token
        deck_q = self.param(
            "deck_query",
            nn.initializers.normal(0.02),
            (1, 1, D),
        )
        B = card_repr_b.shape[0]
        q = jnp.broadcast_to(deck_q, (B, 1, D))
        attn = jnp.einsum("bid,bjd->bij", q, card_repr_b)  # [B, 1, N]
        attn = attn / (D ** 0.5)
        logits = attn[:, 0, :]   # [B, N]
        probs = jax.nn.softmax(logits, axis=-1)
        return logits, probs


# ─────────────────────────────────────────────────────────────────────────────
# Sampling avec contraintes réglementaires
# ─────────────────────────────────────────────────────────────────────────────
def sample_deck(
    logits: jnp.ndarray,          # [num_card_ids]
    rng: jnp.ndarray,
    num_card_ids: int,
    energy_ids: List[int],
    ace_spec_ids: List[int] | None = None,
    max_copies_normal: int = MAX_COPIES_NORMAL,
    temperature: float = 1.0,
) -> Tuple[List[int], jnp.ndarray]:
    """
    Sample DECK_SIZE card IDs respecting PTCG rules:
      - Basic Energy cards: unlimited copies
      - Ace Spec cards: max 1 copy
      - All other cards: max ``max_copies_normal`` copies (default 4)
    Uses sequential sampling with count-based masking (numpy, outside JIT).

    Returns:
        deck      : list of 60 int card_ids
        deck_ids  : jnp.ndarray [DECK_SIZE]  — chosen card IDs for REINFORCE
    """
    import numpy as np

    energy_set   = set(energy_ids)
    ace_spec_set = set(ace_spec_ids) if ace_spec_ids else set(_ACE_SPEC_IDS)
    counts  = np.zeros(num_card_ids, dtype=np.int32)
    deck    = []

    scaled_logits = logits / max(temperature, 1e-6)
    logits_np     = np.array(scaled_logits)

    rng_np = np.random.default_rng(int(rng[0]) % (2**31))

    for _ in range(DECK_SIZE):
        # Build mask: 0 = available, -inf = forbidden
        mask = np.zeros(num_card_ids, dtype=np.float32)
        mask[0] = -np.inf   # ID 0 is padding / "no card"
        for cid in range(1, num_card_ids):
            if cid in energy_set:
                lim = num_card_ids          # pas de limite sur les énergies de base
            elif cid in ace_spec_set:
                lim = 1                     # Ace Spec : 1 exemplaire max
            else:
                lim = max_copies_normal     # cartes normales : 4 max
            if counts[cid] >= lim:
                mask[cid] = -np.inf

        probs = _softmax_np(logits_np + mask)
        chosen = rng_np.choice(num_card_ids, p=probs)
        counts[chosen] += 1
        deck.append(int(chosen))

    return deck, jnp.array(deck, dtype=jnp.int32)


def _softmax_np(x):
    import numpy as np
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE update step
# ─────────────────────────────────────────────────────────────────────────────
def deck_reinforce_update(
    network: DeckBuilderNetwork,
    deck_params: dict,
    deck_opt_state,
    optimizer: optax.GradientTransformation,
    deck_ids: jnp.ndarray,     # [DECK_SIZE]  sampled card IDs
    reward: float,             # scalar game outcome
    baseline: float,           # EMA baseline
    entropy_coef: float = 0.01,
    baseline_ema: float = 0.99,
) -> Tuple[dict, any, float, float]:
    """
    One REINFORCE gradient step on the deck builder.

    Loss = -(R - b) · Σ log π(card_i)   +   entropy_coef · H(π)

    Returns updated (params, opt_state, new_baseline, loss_value).
    """
    advantage = reward - baseline

    def loss_fn(p):
        logits, _ = network.apply(p)
        logits = logits[0]
        log_probs = jax.nn.log_softmax(logits)
        selected_log_probs = log_probs[deck_ids]
        entropy = -jnp.sum(jnp.exp(log_probs) * log_probs)
        sum_selected_lp = jnp.sum(selected_log_probs)
        loss = -advantage * sum_selected_lp - entropy_coef * entropy
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(deck_params)
    updates, new_opt_state = optimizer.update(grads, deck_opt_state, deck_params)
    new_params = optax.apply_updates(deck_params, updates)

    # Update EMA baseline
    new_baseline = baseline_ema * baseline + (1 - baseline_ema) * reward

    return new_params, new_opt_state, new_baseline, float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation helper
# ─────────────────────────────────────────────────────────────────────────────
def create_deck_train_state(
    network: DeckBuilderNetwork,
    cfg: Config,
    rng: jnp.ndarray,
) -> Tuple[dict, any, float]:
    """
    Initialise le DeckBuilderNetwork et retourne
    (params, opt_state, baseline=0.0).
    """
    dummy_ctx = jnp.zeros((1, cfg.model.latent_dim))
    params    = network.init(rng, context=dummy_ctx)
    optimizer = optax.adam(cfg.train.deck_lr)
    opt_state = optimizer.init(params)
    return params, opt_state, 0.0
