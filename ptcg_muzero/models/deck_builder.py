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


    OUTDATED DO NOT USE
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

# IDs of Basic Pokémon.  Filled from the card CSV at startup so sampled decks
# always satisfy the engine's mandatory-basic-Pokémon rule.
_BASIC_POKEMON_IDS: List[int] = []

# Map of Card ID to Name string to enforce 4-copy limit per name
_CARD_ID_TO_NAME: Dict[int, str] = {}

# Deck de compétition de référence (Ogerpon Masque Turquoise-ex + Méganium + Arboliva-ex)
DEFAULT_COMPETITIVE_DECK: List[int] = [
    # Pokémon (21)
    96, 96, 96, 96,      # 4x Teal Mask Ogerpon ex
    402, 402,            # 2x Smoliv
    403, 403,            # 2x Dolliv
    404, 404,            # 2x Arboliva ex
    708, 708,            # 2x Chikorita
    709, 709,            # 2x Bayleef
    710, 710,            # 2x Meganium (Wild Growth)
    140,                 # 1x Fezandipiti ex
    1071,                # 1x Meowth ex
    235,                 # 1x Budew
    172,                 # 1x Hoothoot
    173,                 # 1x Noctowl
    # Dresseurs (28)
    1227, 1227, 1227, 1227,  # 4x Lillie's Determination
    1231, 1231,              # 2x Dawn
    1182, 1182,              # 2x Boss's Orders
    1184,                    # 1x Lana's Aid
    1201,                    # 1x Briar
    1094, 1094, 1094, 1094,  # 4x Bug Catching Set
    1121, 1121, 1121, 1121,  # 4x Ultra Ball
    1152, 1152, 1152,        # 3x Poké Pad
    1097,                    # 1x Night Stretcher
    1116,                    # 1x Energy Switch
    1080,                    # 1x Unfair Stamp (ACE SPEC)
    1261, 1261, 1261, 1261,  # 4x Forest of Vitality
    # Énergies (11)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 11x Basic Grass Energy
]


def set_energy_ids(ids: List[int]) -> None:
    """Appelé une fois après parsing du CSV."""
    global _BASIC_ENERGY_IDS
    _BASIC_ENERGY_IDS = list(ids)


def set_ace_spec_ids(ids: List[int]) -> None:
    """Appelé une fois après parsing du CSV."""
    global _ACE_SPEC_IDS
    _ACE_SPEC_IDS = list(ids)


def set_basic_pokemon_ids(ids: List[int]) -> None:
    """Called once after parsing the card CSV."""
    global _BASIC_POKEMON_IDS
    _BASIC_POKEMON_IDS = list(ids)


def set_card_names(id_to_name: Dict[int, str]) -> None:
    """Called once after parsing the card CSV to map IDs to Card Names."""
    global _CARD_ID_TO_NAME
    _CARD_ID_TO_NAME = dict(id_to_name)


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
    basic_pokemon_ids: List[int] | None = None,
    max_copies_normal: int = MAX_COPIES_NORMAL,
    temperature: float = 1.0,
) -> Tuple[List[int], jnp.ndarray]:
    """
    Sample DECK_SIZE card IDs respecting PTCG rules:
      - Basic Energy cards: unlimited copies
      - Ace Spec cards: max 1 copy
      - All other cards: max ``max_copies_normal`` copies (default 4) per CARD NAME
    Uses sequential sampling with count-based masking (numpy, outside JIT).

    Returns:
        deck      : list of 60 int card_ids
        deck_ids  : jnp.ndarray [DECK_SIZE]  — chosen card IDs for REINFORCE
    """
    import numpy as np
    import collections

    energy_set   = set(energy_ids)
    ace_spec_set = set(ace_spec_ids) if ace_spec_ids else set(_ACE_SPEC_IDS)
    basic_set = set(basic_pokemon_ids) if basic_pokemon_ids is not None else set(_BASIC_POKEMON_IDS)
    counts       = collections.defaultdict(int)
    name_counts  = collections.defaultdict(int)
    deck         = []

    scaled_logits = logits / max(temperature, 1e-6)
    logits_np     = np.array(scaled_logits)

    rng_np = np.random.default_rng(int(rng[0]) % (2**31))

    # Track if any Ace Spec card has already been added to the deck
    has_ace_spec = False

    for _ in range(DECK_SIZE):
        # Build mask: 0 = available, -inf = forbidden
        mask = np.zeros(num_card_ids, dtype=np.float32)
        mask[0] = -np.inf   # ID 0 is padding / "no card"
        for cid in range(1, num_card_ids):
            cname = _CARD_ID_TO_NAME.get(cid, str(cid))
            if cid in ace_spec_set:
                lim = 0 if has_ace_spec else 1
                curr = name_counts[cname]
            elif cid in energy_set:
                lim = num_card_ids          # pas de limite sur les énergies de base
                curr = counts[cid]
            else:
                lim = max_copies_normal     # 4 max par NOM DE CARTE
                curr = name_counts[cname]
            if curr >= lim or counts[cid] >= max_copies_normal:
                mask[cid] = -np.inf

        probs = _softmax_np(logits_np + mask)
        chosen = int(rng_np.choice(num_card_ids, p=probs))
        counts[chosen] += 1
        cname = _CARD_ID_TO_NAME.get(chosen, str(chosen))
        name_counts[cname] += 1
        deck.append(chosen)
        
        # Si la carte choisie fait partie du set Ace Spec, on verrouille la contrainte globale
        if chosen in ace_spec_set:
            has_ace_spec = True

    # battle_start rejects a deck without Basic Pokémon, and a deck without
    # Energy cards cannot attach energy or attack.  Enforce strict deck balance:
    #   - At least 8 Basic Pokémon
    #   - At least 14 Energy cards
    min_basic = 8
    current_basic = sum(1 for cid in deck if cid in basic_set)
    if basic_set and current_basic < min_basic:
        for i in range(len(deck) - 1, -1, -1):
            if current_basic >= min_basic:
                break
            if deck[i] not in basic_set and deck[i] not in energy_set:
                eligible_basic = [
                    cid for cid in basic_set
                    if 0 < cid < num_card_ids
                    and counts[cid] < max_copies_normal
                    and name_counts[_CARD_ID_TO_NAME.get(cid, str(cid))] < max_copies_normal
                ]
                if not eligible_basic:
                    break
                chosen = max(eligible_basic, key=lambda cid: logits_np[cid])
                old_cid = deck[i]
                old_name = _CARD_ID_TO_NAME.get(old_cid, str(old_cid))
                new_name = _CARD_ID_TO_NAME.get(chosen, str(chosen))

                counts[old_cid] -= 1
                name_counts[old_name] -= 1
                counts[chosen] += 1
                name_counts[new_name] += 1
                deck[i] = chosen
                current_basic += 1

    min_energy = 14
    current_energy = sum(1 for cid in deck if cid in energy_set)
    if energy_set and current_energy < min_energy:
        for i in range(len(deck) - 1, -1, -1):
            if current_energy >= min_energy:
                break
            if deck[i] not in energy_set and (deck[i] not in basic_set or current_basic > min_basic):
                eligible_energy = [
                    cid for cid in energy_set
                    if 0 < cid < num_card_ids
                    and (counts[cid] < max_copies_normal or cid in energy_ids)
                    and (name_counts[_CARD_ID_TO_NAME.get(cid, str(cid))] < max_copies_normal or cid in energy_ids)
                    and (cid not in ace_spec_set or not has_ace_spec)
                ]
                if not eligible_energy:
                    break
                energy_logits = logits_np[eligible_energy]
                energy_probs = _softmax_np(energy_logits)
                chosen = int(rng_np.choice(eligible_energy, p=energy_probs))
                if chosen in ace_spec_set:
                    has_ace_spec = True

                old_cid = deck[i]
                old_name = _CARD_ID_TO_NAME.get(old_cid, str(old_cid))
                new_name = _CARD_ID_TO_NAME.get(chosen, str(chosen))

                if old_cid in basic_set:
                    current_basic -= 1
                counts[old_cid] -= 1
                name_counts[old_name] -= 1
                counts[chosen] += 1
                name_counts[new_name] += 1
                deck[i] = chosen
                current_energy += 1

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
