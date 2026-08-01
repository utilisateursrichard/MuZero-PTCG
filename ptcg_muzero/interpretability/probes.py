"""
ptcg_muzero/interpretability/probes.py
========================================
Linear Probing Classifiers — interprétabilité mécanique.

Principe
--------
On détache le gradient de l'état latent (``jax.lax.stop_gradient``) et on
entraîne 5 sondes linéaires indépendantes pour prédire des concepts critiques
du jeu directement depuis z.  Si une sonde atteint une précision élevée,
cela prouve que le concept est *linéairement encodé* dans l'espace latent.

Les 5 sondes
------------
  0. ``active_in_ko_range``   — Mon Pokémon actif peut-il être KO en 1 attaque ?
                                 cible : bool  (binaire)
  1. ``type_advantage``        — Mon type bat-il le type actif adverse ?
                                 cible : {-1, 0, +1} → 3 classes
  2. ``prize_lead``            — Suis-je en avance sur les prizes ?
                                 cible : {behind, even, ahead} → 3 classes
  3. ``hand_advantage``        — Ai-je ≥ 2 cartes de plus que l'adversaire ?
                                 cible : bool
  4. ``opp_energy_ready``      — L'actif adverse a-t-il assez d'énergie pour attaquer ?
                                 cible : bool

Toutes les sondes partagent la même structure : une seule Dense(num_classes).
Leur gradient ne remonte PAS dans le réseau principal (stop_gradient).

API principale
--------------
  ProbeHeads  – Flax module contenant les 5 têtes linéaires
  extract_probe_targets  – extrait les cibles depuis l'obs numpy
  probe_loss  – calcule la cross-entropie pour toutes les sondes
  probe_accuracy – métriques de précision (pas de gradient)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import logging
import threading

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from config import ModelConfig
from env.encoding import GLOBAL_FEAT_DIM

_probe_logger = logging.getLogger("ptcg_muzero.probes")

# ─────────────────────────────────────────────────────────────────────────────
# Définition des sondes
# ─────────────────────────────────────────────────────────────────────────────
PROBE_DEFS: List[Dict] = [
    {"name": "active_in_ko_range",  "num_classes": 2, "idx": 0},
    {"name": "type_advantage",      "num_classes": 3, "idx": 1},
    {"name": "prize_lead",          "num_classes": 3, "idx": 2},
    {"name": "hand_advantage",      "num_classes": 2, "idx": 3},
    {"name": "opp_energy_ready",    "num_classes": 2, "idx": 4},
    {"name": "opp_bench_attacker_ready", "num_classes": 2, "idx": 5},
    {"name": "gust_ko_opportunity",      "num_classes": 2, "idx": 6},
    {"name": "deck_out_risk",            "num_classes": 2, "idx": 7},
    {"name": "evolution_in_hand",        "num_classes": 2, "idx": 8},
    {"name": "ko_next_turn_probable",    "num_classes": 2, "idx": 9},
    {"name": "energy_attachment_available", "num_classes": 2, "idx": 10},
]
NUM_PROBES      = len(PROBE_DEFS)
MAX_PROBE_CLASS = max(d["num_classes"] for d in PROBE_DEFS)


# ─────────────────────────────────────────────────────────────────────────────
# Flax module
# ─────────────────────────────────────────────────────────────────────────────
class ProbeHeads(nn.Module):
    """
    Cinq têtes linéaires indépendantes branchées sur stop_gradient(z).

    Les poids de ces sondes font PARTIE du paramètre tree du MuZeroNetwork
    (dans un sous-dict ``probes``) mais sont aussi disponibles en standalone.
    """
    cfg: ModelConfig

    @nn.compact
    def __call__(
        self, z: jnp.ndarray
    ) -> List[jnp.ndarray]:
        """
        Args:
            z: [B, latent_dim]  — latent state (gradient stopped internally)
        Returns:
            list of 5 logit tensors, each [B, num_classes_i]
        """
        z_sg = jax.lax.stop_gradient(z)   # ← gradient barrier

        logits_list = []
        for pdef in PROBE_DEFS:
            h = nn.Dense(
                pdef["num_classes"],
                name=f"probe_{pdef['name']}",
                use_bias=True,
            )(z_sg)
            logits_list.append(h)
        return logits_list


# ─────────────────────────────────────────────────────────────────────────────
# Cache thread-safe pour la base de données des cartes du moteur
_card_db: dict | None = None
_card_db_lock = threading.Lock()

def _get_card_db() -> dict:
    """Charge (au premier appel) et met en cache la base de données du moteur.
    Thread-safe grâce au verrou. Fallback sur {} si le moteur n'est pas disponible
    (tous les steps de type_advantage seront alors marqués -1, ignorés par probe_loss).
    """
    global _card_db
    if _card_db is None:
        with _card_db_lock:
            if _card_db is None:   # double-checked locking
                try:
                    from env.cabt_api import all_card_data
                    _card_db = {c.cardId: c for c in all_card_data()}
                    _probe_logger.info(
                        "[probes] card_db chargé : %d cartes", len(_card_db)
                    )
                except Exception as exc:
                    _probe_logger.warning(
                        "[probes] Impossible de charger card_db (%s). "
                        "Les targets type_advantage seront ignorées (target=-1).",
                        exc,
                    )
                    _card_db = {}
    return _card_db


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des cibles depuis l'observation
# ─────────────────────────────────────────────────────────────────────────────
def extract_probe_targets(
    obs: dict,
    my_idx: int,
) -> np.ndarray:
    """
    Extrait les 5 cibles entières depuis un obs non-batché (numpy).

    Returns:
        int32 array [NUM_PROBES] — classe cible pour chaque sonde.
        -1  signifie « pas de cible disponible ce step ».
    """
    targets = np.full(NUM_PROBES, -1, dtype=np.int32)

    current = obs.get("current") or {}
    players = current.get("players", [{}, {}])
    opp_idx = 1 - my_idx
    me  = players[my_idx] if my_idx < len(players) else {}
    opp = players[opp_idx] if opp_idx < len(players) else {}

    # ── Sonde 0 : active_in_ko_range ─────────────────────────────────────
    # Mon actif peut-il être KO en 1 attaque adverse ?
    my_active_list = me.get("active") or []
    opp_active_list = opp.get("active") or []
    if my_active_list and opp_active_list:
        my_poke = my_active_list[0]
        opp_poke = opp_active_list[0]
        if my_poke is not None and opp_poke is not None:
            my_hp    = _safe_int(my_poke, "hp", 100)
            max_dmg  = _estimate_max_damage(opp_poke)
            targets[0] = int(max_dmg >= my_hp)

    # ── Sonde 1 : type_advantage ─────────────────────────────────────────
    # 0 = désavantage, 1 = neutre, 2 = avantage
    if my_active_list and opp_active_list:
        my_poke  = my_active_list[0]
        opp_poke = opp_active_list[0]
        if my_poke is not None and opp_poke is not None:
            my_card_id = _safe_int(my_poke, "id", -1)
            opp_card_id = _safe_int(opp_poke, "id", -1)
            db = _get_card_db()
            my_card = db.get(my_card_id)
            opp_card = db.get(opp_card_id)
            if my_card is not None and opp_card is not None:
                my_type = getattr(my_card.energyType, "value", int(my_card.energyType))
                opp_weak_enum = opp_card.weakness
                opp_res_enum = opp_card.resistance
                
                opp_weak = getattr(opp_weak_enum, "value", int(opp_weak_enum)) if opp_weak_enum is not None else -1
                opp_res = getattr(opp_res_enum, "value", int(opp_res_enum)) if opp_res_enum is not None else -1
                
                if my_type >= 0:
                    if my_type == opp_weak:
                        targets[1] = 2    # avantage
                    elif my_type == opp_res:
                        targets[1] = 0    # désavantage
                    else:
                        targets[1] = 1    # neutre

    # ── Sonde 2 : prize_lead ─────────────────────────────────────────────
    my_prizes  = len(me.get("prize") or [])
    opp_prizes = len(opp.get("prize") or [])
    # Plus de prizes restants = plus à prendre = EN RETARD
    if my_prizes < opp_prizes:
        targets[2] = 2    # ahead (j'ai moins de prizes restants)
    elif my_prizes == opp_prizes:
        targets[2] = 1    # even
    else:
        targets[2] = 0    # behind

    # ── Sonde 3 : hand_advantage ─────────────────────────────────────────
    my_hand  = me.get("handCount", 0) or len(me.get("hand") or [])
    opp_hand = opp.get("handCount", 0)
    targets[3] = int(my_hand >= opp_hand + 2)

    # ── Sonde 4 : opp_energy_ready ───────────────────────────────────────
    # L'actif adverse a-t-il assez d'énergie pour attaquer (≥ 2 énergies) ?
    if opp_active_list and opp_active_list[0] is not None:
        opp_poke = opp_active_list[0]
        energies = opp_poke.get("energies") or []
        targets[4] = int(len(energies) >= 2)

    opp_bench = [p for p in (opp.get("bench") or []) if p is not None]
    targets[5] = int(any(_pokemon_attack_ready(p) for p in opp_bench))
    if my_active_list and my_active_list[0] is not None and opp_bench:
        targets[6] = int(any(
            _safe_int(p, "hp", 100) <= _estimate_max_damage(my_active_list[0])
            for p in opp_bench
        ))

    my_deck = _safe_int(me, "deckCount", -1)
    opp_deck = _safe_int(opp, "deckCount", -1)
    if my_deck >= 0 and opp_deck >= 0:
        targets[7] = int(my_deck <= 5 or opp_deck <= 5)

    board = [p for p in (my_active_list + (me.get("bench") or [])) if p is not None]
    hand = [c for c in (me.get("hand") or []) if c is not None]
    if board and hand:
        targets[8] = int(_has_matching_evolution(board, hand))

    if my_active_list and opp_active_list and my_active_list[0] is not None and opp_active_list[0] is not None:
        targets[9] = int(
            _pokemon_attack_ready(my_active_list[0]) and
            _estimate_max_damage(my_active_list[0]) >= _safe_int(opp_active_list[0], "hp", 100)
        )

    if me.get("hand") is not None:
        targets[10] = int(
            not bool(current.get("energyAttached", False)) and
            any(_is_energy_card(c) for c in hand)
        )

    return targets


def _safe_int(obj, key, default):
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    if hasattr(v, "value"):
        return int(v.value)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _estimate_max_damage(poke: dict) -> int:
    """Heuristique simple : max dégât parmi les mouvements de l'actif adverse."""
    # Le game state n'expose pas les dégâts directement ;
    # on utilise l'hp_ratio comme proxy inverse
    hp_total = _safe_int(poke, "maxHp", 100)
    # L'adversaire a potentiellement une attaque à ~60% de l'HP de n'importe qui
    # C'est une borne supérieure grossière — à affiner avec les données CSV
    return max(60, hp_total // 2)


def _pokemon_attack_ready(poke: dict) -> bool:
    return len(poke.get("energies") or []) >= 2 and _estimate_max_damage(poke) > 0


def _card_name(card) -> str:
    if isinstance(card, dict):
        return str(card.get("name", card.get("cardName", "")))
    return str(getattr(card, "name", getattr(card, "cardName", "")))


def _has_matching_evolution(board: list, hand: list) -> bool:
    db = _get_card_db()
    hand_ids = {_safe_int(c, "id", -1) for c in hand}
    hand_names = {_card_name(c).lower() for c in hand}
    for pokemon in board:
        cid = _safe_int(pokemon, "id", -1)
        base = db.get(cid)
        base_name = _card_name(base).lower() if base is not None else _card_name(pokemon).lower()
        for hid in hand_ids:
            card = db.get(hid)
            parent = str(getattr(card, "evolvesFrom", getattr(card, "evolves_from", "")) or "").lower() if card else ""
            if parent and parent == base_name:
                return True
        if base_name and any(base_name in name for name in hand_names):
            return True
    return False


def _is_energy_card(card) -> bool:
    keys = ("stage", "category", "type", "cardType")
    if isinstance(card, dict):
        text = " ".join(str(card.get(k, "")) for k in keys).lower()
    else:
        text = " ".join(str(getattr(card, k, "")) for k in keys).lower()
    return "energy" in text


# ─────────────────────────────────────────────────────────────────────────────
# Loss et métriques (JIT-ables)
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def probe_loss(
    probe_logits: List[jnp.ndarray],   # 5 × [B, num_classes_i]
    targets: jnp.ndarray,              # [B, NUM_PROBES]  int32, -1 = ignore
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calcule la cross-entropie moyenne sur les 5 sondes.
    Ignore les steps où la cible est -1.

    Returns:
        total_loss  : scalar
        per_probe   : [NUM_PROBES]  loss individuelle (pour le logging)
    """
    losses = []
    for i, (logits, pdef) in enumerate(zip(probe_logits, PROBE_DEFS)):
        tgt = targets[:, i]                          # [B]
        valid = (tgt >= 0)                           # [B] bool
        tgt_clamped = jnp.clip(tgt, 0, pdef["num_classes"] - 1)
        xent = optax_softmax_xent(logits, tgt_clamped)  # [B]
        # Mask invalid
        xent_masked = jnp.where(valid, xent, 0.0)
        n_valid = jnp.maximum(jnp.sum(valid), 1)
        losses.append(jnp.sum(xent_masked) / n_valid)

    per_probe = jnp.array(losses)
    return jnp.mean(per_probe), per_probe


@jax.jit
def probe_accuracy(
    probe_logits: List[jnp.ndarray],
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """
    Précision par sonde.  Returns [NUM_PROBES] float32.
    """
    accs = []
    for i, logits in enumerate(probe_logits):
        tgt   = targets[:, i]
        valid = tgt >= 0
        pred  = jnp.argmax(logits, axis=-1)
        correct = (pred == tgt) & valid
        n_valid = jnp.maximum(jnp.sum(valid), 1)
        accs.append(jnp.sum(correct).astype(jnp.float32) / n_valid)
    return jnp.array(accs)


def optax_softmax_xent(
    logits: jnp.ndarray,   # [B, C]
    labels: jnp.ndarray,   # [B]  int
) -> jnp.ndarray:
    """Softmax cross-entropy per sample [B]."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -log_probs[jnp.arange(logits.shape[0]), labels]


# ─────────────────────────────────────────────────────────────────────────────
# Résumé textuel pour le logging
# ─────────────────────────────────────────────────────────────────────────────
def probe_report(accs: np.ndarray, losses: np.ndarray) -> str:
    lines = ["── Probe Classifiers ─────────────────"]
    for i, pdef in enumerate(PROBE_DEFS):
        lines.append(
            f"  {pdef['name']:30s}  acc={accs[i]:.3f}  loss={losses[i]:.4f}"
        )
    return "\n".join(lines)
