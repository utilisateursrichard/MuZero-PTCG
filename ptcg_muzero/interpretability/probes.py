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
                    cards = all_card_data()
                    items = cards.values() if isinstance(cards, dict) else cards
                    # AUDIT §2.5 — résolution tolérante du nom d'attribut d'ID :
                    # un seul `AttributeError` faisait basculer TOUTES les sondes
                    # dépendant de card_db sur target=-1 (donc ignorées).
                    db = {}
                    for c in items:
                        for attr in ("cardId", "card_id", "id", "cardID"):
                            cid = getattr(c, attr, None)
                            if cid is not None:
                                db[int(cid)] = c
                                break
                    _card_db = db
                    _probe_logger.debug(
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
    # Même source que global_feat[6]/[7] : `len(prize)` était constant, donc la
    # cible de cette sonde l'était aussi — d'où une précision de 90-100 % qui ne
    # mesurait rien (cf. env/encoding.py:prize_left).
    from env.encoding import prize_left
    my_prizes  = prize_left(me)
    opp_prizes = prize_left(opp)
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


# ── AUDIT §2.5 : dégâts réels issus du moteur (all_attack) ────────────────────
# L'ancienne heuristique `max(60, maxHp // 2)` n'avait aucun rapport avec les
# attaques réelles de la carte : les sondes 0 (`active_in_ko_range`),
# 6 (`gust_ko_opportunity`) et 9 (`ko_next_turn_probable`) apprenaient du bruit.
# Le moteur expose `all_attack()` (déjà ré-exporté par env/cabt_api) : on en
# dérive une table card_id → dégât maximum, avec repli sur l'ancienne heuristique
# si l'API n'est pas disponible ou change de forme.
_attack_db: dict | None = None
_attack_db_lock = threading.Lock()

_ATTACK_CARD_KEYS = ("cardId", "card_id", "cardID", "pokemonId", "id")
_ATTACK_DMG_KEYS = ("damage", "dmg", "power", "baseDamage", "damageValue")


def _coerce_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        digits = "".join(c for c in value if c.isdigit())
        return int(digits) if digits else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_attack_db() -> dict:
    """card_id → dégât maximum parmi les attaques de la carte."""
    global _attack_db
    if _attack_db is None:
        with _attack_db_lock:
            if _attack_db is None:
                table: Dict[int, int] = {}
                try:
                    from env.cabt_api import all_attack, all_card_data
                    attacks = all_attack()
                    cards = all_card_data()

                    def _get(obj, key):
                        if isinstance(obj, dict):
                            return obj.get(key)
                        return getattr(obj, key, None)

                    # 1. Dictionnaire attackId -> dégâts
                    atk_items = attacks.values() if isinstance(attacks, dict) else attacks
                    atk_dmg_map: Dict[int, int] = {}
                    for atk in atk_items:
                        aid = _coerce_int(_get(atk, "attackId") or _get(atk, "id"), -1)
                        dmg = max((_coerce_int(_get(atk, k), 0) for k in _ATTACK_DMG_KEYS), default=0)
                        if aid >= 0:
                            atk_dmg_map[aid] = dmg

                    # 2. Association carte -> max dégâts parmi ses attaques
                    card_items = cards.values() if isinstance(cards, dict) else cards
                    for c in card_items:
                        cid = _coerce_int(_get(c, "id") or _get(c, "cardId"), -1)
                        if cid < 0:
                            continue
                        atks = _get(c, "attacks") or []
                        max_dmg = 0
                        for a in atks:
                            aid = _coerce_int(a if not isinstance(a, dict) else _get(a, "attackId"), -1)
                            if aid in atk_dmg_map:
                                max_dmg = max(max_dmg, atk_dmg_map[aid])
                        if max_dmg > 0:
                            table[cid] = max_dmg

                    _probe_logger.debug(
                        "[probes] attack_db chargée : %d cartes avec dégâts connus", len(table)
                    )
                except Exception as exc:
                    _probe_logger.warning(
                        "[probes] all_attack() indisponible (%s) — repli sur l'heuristique de dégâts.",
                        exc,
                    )
                _attack_db = table
    return _attack_db


def _estimate_max_damage(poke: dict) -> int:
    """Dégât maximum réel de la carte (moteur) ; repli heuristique si inconnu."""
    cid = _safe_int(poke, "id", -1)
    if cid >= 0:
        dmg = _get_attack_db().get(cid)
        if dmg:
            return int(dmg)
    hp_total = _safe_int(poke, "maxHp", 100)
    return max(60, hp_total // 2)


def _pokemon_attack_ready(poke: dict) -> bool:
    return len(poke.get("energies") or []) >= 2 and _estimate_max_damage(poke) > 0


def _card_id_of(card) -> int:
    """ID d'une carte, qu'elle soit un dict, un objet, ou un entier nu.

    Le moteur peut renvoyer les zones « main » / « défausse » / « prizes » comme
    de simples listes d'IDs (cf. `env/encoding.py`).  Sans ce cas, `_safe_int`
    renvoyait -1 pour toute carte entière et les sondes 8
    (`evolution_in_hand`) et 10 (`energy_attachment_available`) retournaient
    silencieusement 0 au lieu de leur vraie valeur.
    """
    if card is None or isinstance(card, bool):
        return -1
    if isinstance(card, (int, np.integer)):
        return int(card)
    return _safe_int(card, "id", -1)


def _card_name(card) -> str:
    """Nom d'une carte ; résolu via la card_db du moteur si l'entrée est un ID nu."""
    if isinstance(card, (int, np.integer)) and not isinstance(card, bool):
        entry = _get_card_db().get(int(card))
        return "" if entry is None else _card_name(entry)
    if isinstance(card, dict):
        return str(card.get("name", card.get("cardName", "")))
    return str(getattr(card, "name", getattr(card, "cardName", "")))


def _has_matching_evolution(board: list, hand: list) -> bool:
    db = _get_card_db()
    hand_ids = {_card_id_of(c) for c in hand}
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
    # Carte donnée sous forme d'ID nu : on résout via la card_db du moteur.
    if isinstance(card, (int, np.integer)) and not isinstance(card, bool):
        entry = _get_card_db().get(int(card))
        if entry is None:
            return False
        card = entry
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
