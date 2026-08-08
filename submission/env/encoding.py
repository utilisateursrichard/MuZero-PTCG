"""
ptcg_muzero/env/encoding.py
============================
Converts a cabt ``Observation`` dict (as passed to the agent function) into
fixed-size numpy arrays ready for the JAX networks.

Design choices
--------------
* Everything is padded to static shapes so JAX can JIT-compile the networks.
* ``card_id = 0`` is the "empty / unknown" sentinel (see cards/encoder.py).
* Pokemon per-slot features are encoded separately from card IDs so the
  transformer can attend to both the card's identity and its in-game state.
* Available options are also encoded and appended to the observation so the
  policy head knows what each action index refers to.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from config import ModelConfig

# ── Sentinel ──────────────────────────────────────────────────────────────────
NO_CARD: int = 0   # padding / unknown card

# ── Per-Pokemon feature vector layout ────────────────────────────────────────
# hp_ratio        (1)
# energy counts   (11)  — one counter per energy type
# status          (5)   — poison burn sleep paralyze confuse
# appear_this_turn(1)
POKEMON_FEAT_DIM: int = 1 + 11 + 5 + 1   # = 18

# ── Global state feature layout ───────────────────────────────────────────────
# turn (norm), my_idx, supporter_played, stadium_played,
# energy_attached, retreated, my_prizes, opp_prizes,
# my_deck, opp_deck, my_hand_count, opp_hand_count
GLOBAL_FEAT_DIM: int = 12

# ── Option feature layout ─────────────────────────────────────────────────────
# option_type one-hot (17)  + area one-hot (13) + inPlayArea one-hot (13)
# + player_idx (1) + has_card_id (1) + is_attack_1 (1) + is_attack_2+ (1) + extra_val (1) + norm_index (1) + norm_in_play_index (1) → total 50
OPTION_TYPE_DIM: int = 17
AREA_DIM: int = 13
OPTION_FEAT_DIM: int = OPTION_TYPE_DIM + AREA_DIM + AREA_DIM + 1 + 1 + 2 + 1 + 2  # = 50


# ─────────────────────────────────────────────────────────────────────────────
# Main encoding function
# ─────────────────────────────────────────────────────────────────────────────
def _as_dict(obj) -> dict:
    """Normalise un objet (dict OU dataclass/namedtuple) en dict accesseur par clé.

    Compatible avec les observations reçues depuis le moteur cabt (qui peuvent
    être des dicts Python purs sur Kaggle OU des dataclasses selon l'appelant).
    """
    import dataclasses as _dc
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if _dc.is_dataclass(obj) and not isinstance(obj, type):
        d = {}
        for f in _dc.fields(obj):
            v = getattr(obj, f.name, None)
            if _dc.is_dataclass(v) and not isinstance(v, type):
                d[f.name] = _as_dict(v)
            elif isinstance(v, list):
                d[f.name] = [
                    _as_dict(x) if _dc.is_dataclass(x) and not isinstance(x, type) else x
                    for x in v
                ]
            else:
                d[f.name] = v
        return d
    try:
        return dict(vars(obj))
    except (TypeError, ValueError):
        return {}


def encode_observation(
    obs_dict,
    my_idx: int,
    cfg: ModelConfig,
) -> Dict[str, np.ndarray]:
    """
    Convert a cabt observation dict OR Observation dataclass to fixed-size arrays.

    Keys returned
    -------------
    global_feat          float32 [GLOBAL_FEAT_DIM]
    my_active_id         int32   [1]
    my_active_feat       float32 [1, POKEMON_FEAT_DIM]
    my_bench_ids         int32   [max_bench_size]
    my_bench_feat        float32 [max_bench_size, POKEMON_FEAT_DIM]
    my_bench_mask        bool    [max_bench_size]
    my_hand_ids          int32   [max_hand_size]
    my_hand_mask         bool    [max_hand_size]
    my_discard_ids       int32   [max_discard_size]
    my_discard_mask      bool    [max_discard_size]
    my_prize_ids         int32   [max_prize_size]    (0 = face-down / unknown)
    opp_active_id        int32   [1]
    opp_active_feat      float32 [1, POKEMON_FEAT_DIM]
    opp_bench_ids        int32   [max_bench_size]
    opp_bench_feat       float32 [max_bench_size, POKEMON_FEAT_DIM]
    opp_bench_mask       bool    [max_bench_size]
    opp_discard_ids      int32   [max_discard_size]
    opp_discard_mask     bool    [max_discard_size]
    opp_prize_ids        int32   [max_prize_size]
    opp_hand_ids         int32   [max_hand_size]    (belief-filled, 0 if unknown)
    opp_hand_mask        bool    [max_hand_size]
    option_ids           int32   [max_actions]       (card_id of each option, 0 if n/a)
    option_feat          float32 [max_actions, OPTION_FEAT_DIM]
    option_mask          bool    [max_actions]
    """
    obs = _as_dict(obs_dict)
    current = _as_dict(obs.get("current"))
    select  = _as_dict(obs.get("select"))

    players = current.get("players") or [{}, {}]
    players = [_as_dict(p) for p in players]
    opp_idx = 1 - my_idx
    me  = players[my_idx] if my_idx < len(players) else {}
    opp = players[opp_idx] if opp_idx < len(players) else {}

    # ── Global features ───────────────────────────────────────────────────
    turn          = current.get("turn", 0)
    # prizes remaining = total - taken; face-down prize = still in play
    my_prizes_left  = len(me.get("prize") or [])
    opp_prizes_left = len(opp.get("prize") or [])

    global_feat = np.array([
        min(turn, 200) / 200.0,
        float(my_idx),
        float(current.get("supporterPlayed", False)),
        float(current.get("stadiumPlayed",   False)),
        float(current.get("energyAttached",  False)),
        float(current.get("retreated",       False)),
        my_prizes_left  / 6.0,
        opp_prizes_left / 6.0,
        min(int(me.get("deckCount") or 0), 60)  / 60.0,
        min(int(opp.get("deckCount") or 0), 60) / 60.0,
        min(int(me.get("handCount") or 0), 20)  / 20.0,
        min(int(opp.get("handCount") or 0), 20) / 20.0,
    ], dtype=np.float32)

    # ── My board ──────────────────────────────────────────────────────────
    my_active_id, my_active_feat = _encode_slot_list(
        me.get("active", []), cfg, is_active=True, player_state=me
    )
    my_bench_ids, my_bench_feat, my_bench_mask = _encode_bench(
        me.get("bench", []), cfg
    )
    my_hand_ids, my_hand_mask = _encode_card_list(
        me.get("hand", []), cfg.max_hand_size
    )
    my_discard_ids, my_discard_mask = _encode_card_list(
        me.get("discard", []), cfg.max_discard_size
    )
    my_prize_ids = _encode_prize(me.get("prize", []), cfg.max_prize_size)

    # ── Opponent board ────────────────────────────────────────────────────
    opp_active_id, opp_active_feat = _encode_slot_list(
        opp.get("active", []), cfg, is_active=True, player_state=opp
    )
    opp_bench_ids, opp_bench_feat, opp_bench_mask = _encode_bench(
        opp.get("bench", []), cfg
    )
    opp_discard_ids, opp_discard_mask = _encode_card_list(
        opp.get("discard", []), cfg.max_discard_size
    )
    opp_prize_ids = _encode_prize(opp.get("prize", []), cfg.max_prize_size)
    opp_hand_ids, opp_hand_mask = _encode_opp_hand(opp, cfg.max_hand_size)

    # ── Options ───────────────────────────────────────────────────────────
    raw_options = select.get("option") if isinstance(select, dict) else getattr(select, "option", None)
    if raw_options is None and isinstance(select, dict):
        raw_options = select.get("options")
    elif raw_options is None:
        raw_options = getattr(select, "options", [])
    if raw_options is None:
        raw_options = []

    option_ids, option_feat, option_mask = _encode_options(
        raw_options,
        cfg,
        active_card_id=int(my_active_id[0]),
        my_hand_ids=my_hand_ids,
        my_bench_ids=my_bench_ids,
        my_discard_ids=my_discard_ids,
    )

    return {
        "global_feat":      global_feat,
        "my_active_id":     my_active_id,
        "my_active_feat":   my_active_feat,
        "my_bench_ids":     my_bench_ids,
        "my_bench_feat":    my_bench_feat,
        "my_bench_mask":    my_bench_mask,
        "my_hand_ids":      my_hand_ids,
        "my_hand_mask":     my_hand_mask,
        "my_discard_ids":   my_discard_ids,
        "my_discard_mask":  my_discard_mask,
        "my_prize_ids":     my_prize_ids,
        "opp_active_id":    opp_active_id,
        "opp_active_feat":  opp_active_feat,
        "opp_bench_ids":    opp_bench_ids,
        "opp_bench_feat":   opp_bench_feat,
        "opp_bench_mask":   opp_bench_mask,
        "opp_discard_ids":  opp_discard_ids,
        "opp_discard_mask": opp_discard_mask,
        "opp_prize_ids":    opp_prize_ids,
        "opp_hand_ids":     opp_hand_ids,
        "opp_hand_mask":    opp_hand_mask,
        "option_ids":       option_ids,
        "option_feat":      option_feat,
        "option_mask":      option_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-entity encoders
# ─────────────────────────────────────────────────────────────────────────────
def _encode_pokemon(
    pokemon: Optional[dict],
    is_active: bool = False,
    player_state: Optional[dict] = None,
) -> tuple:
    """
    Returns (card_id: int, feat: float32 [POKEMON_FEAT_DIM]).
    pokemon may be None (face-down Pokémon).
    """
    if pokemon is None:
        return NO_CARD, np.zeros(POKEMON_FEAT_DIM, dtype=np.float32)

    card_id = pokemon.get("id", NO_CARD) or NO_CARD

    hp     = pokemon.get("hp", 0) or 0
    max_hp = pokemon.get("maxHp", 1) or 1
    hp_ratio = np.clip(hp / max_hp, 0.0, 1.0)

    # Energy counts — energies is a list of EnergyType int values
    energy_counts = np.zeros(11, dtype=np.float32)
    for e in pokemon.get("energies", []):
        v = e if isinstance(e, int) else (e.get("value", -1) if isinstance(e, dict) else -1)
        if 0 <= v < 11:
            energy_counts[v] += 1
        elif v == 11:  # TEAM_ROCKET -> mapped to Rainbow (10) for energy representation
            energy_counts[10] += 1
    # Normalise (max 5 energy of one type is generous)
    energy_counts = np.clip(energy_counts / 5.0, 0.0, 1.0)

    # Status conditions (only present on active Pokémon via player_state)
    status = np.zeros(5, dtype=np.float32)
    if is_active and player_state:
        status[0] = float(player_state.get("poisoned", False))
        status[1] = float(player_state.get("burned", False))
        status[2] = float(player_state.get("asleep", False))
        status[3] = float(player_state.get("paralyzed", False))
        status[4] = float(player_state.get("confused", False))

    feat = np.concatenate([
        np.array([hp_ratio], dtype=np.float32),
        energy_counts,
        status,
        np.array([float(pokemon.get("appearThisTurn", False))], dtype=np.float32),
    ])
    return card_id, feat


def _encode_slot_list(
    slot_list: list,
    cfg: ModelConfig,
    is_active: bool = False,
    player_state: Optional[dict] = None,
) -> tuple:
    """
    Encode a size-0-or-1 slot list (active Pokémon).
    Returns (ids [1], feats [1, POKEMON_FEAT_DIM]).
    """
    ids   = np.zeros(1, dtype=np.int32)
    feats = np.zeros((1, POKEMON_FEAT_DIM), dtype=np.float32)
    if slot_list:
        cid, feat = _encode_pokemon(slot_list[0], is_active, player_state)
        ids[0]    = cid
        feats[0]  = feat
    return ids, feats


def _encode_bench(
    bench: list,
    cfg: ModelConfig,
) -> tuple:
    """
    Encode bench Pokémon (up to max_bench_size).
    Returns (ids, feats, mask).
    """
    B     = cfg.max_bench_size
    ids   = np.zeros(B, dtype=np.int32)
    feats = np.zeros((B, POKEMON_FEAT_DIM), dtype=np.float32)
    mask  = np.zeros(B, dtype=bool)
    for i, pokemon in enumerate(bench[:B]):
        if pokemon is not None:
            cid, feat = _encode_pokemon(pokemon)
            ids[i]    = cid
            feats[i]  = feat
            mask[i]   = True
    return ids, feats, mask


def _encode_card_list(cards: Optional[list], max_len: int) -> tuple:
    """
    Encode a list of Card dicts into (int32 ids, bool mask).
    """
    ids  = np.zeros(max_len, dtype=np.int32)
    mask = np.zeros(max_len, dtype=bool)
    if not cards:
        return ids, mask
    for i, card in enumerate(cards[:max_len]):
        if card is not None:
            ids[i]  = card.get("id", NO_CARD) or NO_CARD
            mask[i] = True
    return ids, mask


def _encode_prize(prizes: list, max_size: int) -> np.ndarray:
    """Encode prize card IDs (None = face-down, encoded as 0)."""
    ids = np.zeros(max_size, dtype=np.int32)
    for i, p in enumerate(prizes[:max_size]):
        if p is not None:
            ids[i] = p.get("id", NO_CARD) or NO_CARD
    return ids


def _encode_opp_hand(player: dict, max_len: int) -> tuple:
    """
    Opponent hand cards are usually hidden. Keep known IDs when exposed, otherwise
    expose only a validity mask from handCount so belief sampling can fill IDs.
    """
    ids = np.zeros(max_len, dtype=np.int32)
    mask = np.zeros(max_len, dtype=bool)

    hand = player.get("hand") or []
    if hand:
        ids, mask = _encode_card_list(hand, max_len)

    hand_count = player.get("handCount", None)
    if hand_count is not None:
        n = min(max(int(hand_count or 0), 0), max_len)
        mask[:n] = True

    return ids, mask


def _encode_options(
    options: list,
    cfg: ModelConfig,
    active_card_id: int = NO_CARD,
    my_hand_ids: np.ndarray | None = None,
    my_bench_ids: np.ndarray | None = None,
    my_discard_ids: np.ndarray | None = None,
) -> tuple:
    """
    Encode available options with exhaustive card_id resolution and attack distinction.
    Returns (ids [max_actions], feat [max_actions, OPTION_FEAT_DIM], mask [max_actions]).
    """
    A    = cfg.max_actions
    ids  = np.zeros(A, dtype=np.int32)
    feat = np.zeros((A, OPTION_FEAT_DIM), dtype=np.float32)
    mask = np.zeros(A, dtype=bool)

    for i, opt in enumerate(options[:A]):
        if opt is None:
            continue

        opt_type  = _int_from(opt, "type", 0)
        area      = _int_from(opt, "area", 0)
        in_area   = _int_from(opt, "inPlayArea", 0)
        p_idx     = _int_from(opt, "playerIndex", 0)
        card_id   = _int_from(opt, "cardId", NO_CARD) or NO_CARD
        attack_id = _int_from(opt, "attackId", 0)
        idx       = _int_from(opt, "index", 0)

        # Resolution exhaustive du card_id si absent de l'option JSON C++
        if card_id == NO_CARD:
            if opt_type in (12, 13) and active_card_id != NO_CARD:
                # ATTACK (13) et RETREAT (12) -> Pokémon Actif
                card_id = active_card_id
            elif opt_type == 7 and my_hand_ids is not None and 0 <= idx < len(my_hand_ids):
                # PLAY (7) -> carte dans la main à 'index'
                card_id = int(my_hand_ids[idx])
            elif opt_type in (8, 9) and area == 2 and my_hand_ids is not None and 0 <= idx < len(my_hand_ids):
                # ATTACH (8) et EVOLVE (9) depuis la main (HAND=2)
                card_id = int(my_hand_ids[idx])
            elif opt_type in (3, 4, 5, 6, 10, 11):
                # CARD (3), TOOLCARD (4), ENERGYCARD (5), ENERGY (6), ABILITY (10), DISCARD (11)
                if area == 4 and active_card_id != NO_CARD:      # ACTIVE = 4
                    card_id = active_card_id
                elif area == 5 and my_bench_ids is not None and 0 <= idx < len(my_bench_ids):   # BENCH = 5
                    card_id = int(my_bench_ids[idx])
                elif area == 2 and my_hand_ids is not None and 0 <= idx < len(my_hand_ids):     # HAND = 2
                    card_id = int(my_hand_ids[idx])
                elif area == 3 and my_discard_ids is not None and 0 <= idx < len(my_discard_ids): # DISCARD = 3
                    card_id = int(my_discard_ids[idx])

        ids[i]  = card_id
        mask[i] = True

        # one-hot type (17), area (13), in_play_area (13), player_idx (1), has_card (1), is_attack_1 (1), is_attack_2 (1), extra_val (1)
        f = np.zeros(OPTION_FEAT_DIM, dtype=np.float32)
        if 0 <= opt_type < OPTION_TYPE_DIM:
            f[opt_type] = 1.0
        off = OPTION_TYPE_DIM
        if 0 <= area < AREA_DIM:
            f[off + area] = 1.0
        off += AREA_DIM
        if 0 <= in_area < AREA_DIM:
            f[off + in_area] = 1.0
        off += AREA_DIM
        f[off]     = float(p_idx)
        f[off + 1] = float(card_id != NO_CARD or attack_id > 0 or opt_type == 13)
        f[off + 2] = float(opt_type == 13 or attack_id > 0)  # Explicit flag for ATTACK actions
        f[off + 3] = np.clip(attack_id / 3000.0, 0.0, 1.0)
        num_val    = _int_from(opt, "number", 0) or _int_from(opt, "count", 0) or _int_from(opt, "energyIndex", 0)
        f[off + 4] = np.clip(num_val / 10.0, 0.0, 1.0)
        in_play_idx = _int_from(opt, "inPlayIndex", 0)
        f[off + 5] = np.clip(idx / 60.0, 0.0, 1.0)
        f[off + 6] = np.clip(in_play_idx / 5.0, 0.0, 1.0)
        feat[i]    = f

    return ids, feat, mask


# ─────────────────────────────────────────────────────────────────────────────
# Reward extraction from logs
# ─────────────────────────────────────────────────────────────────────────────
def extract_step_reward(logs: list, my_idx: int) -> float:
    """
    Sparse rewards only:
      +1.0  if I win
      -1.0  if I lose
       0.0  otherwise
    """
    for log in logs:
        log_type = _int_from(log, "type", -1)
        # LogType.RESULT = 23
        if log_type == 23:
            result = _int_from(log, "result", 2)
            if result == my_idx:
                return 1.0
            elif result != 2:      # != draw
                return -1.0
    return 0.0



def _int_from(obj, key: str, default: int) -> int:
    """Safely get an integer attribute from a dict or object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return int(v)
    # Enum-like: has .value
    if hasattr(v, "value"):
        try:
            return int(v.value)
        except (ValueError, TypeError):
            pass
    return default
