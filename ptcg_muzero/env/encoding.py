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
# + player_idx (1) + has_card_id (1) → total 45
OPTION_TYPE_DIM: int = 17
AREA_DIM: int = 13
OPTION_FEAT_DIM: int = OPTION_TYPE_DIM + AREA_DIM + AREA_DIM + 1 + 1  # = 45


# ─────────────────────────────────────────────────────────────────────────────
# Main encoding function
# ─────────────────────────────────────────────────────────────────────────────
def encode_observation(
    obs_dict: dict,
    my_idx: int,
    cfg: ModelConfig,
) -> Dict[str, np.ndarray]:
    """
    Convert a cabt observation dict to a dict of fixed-size numpy arrays.

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
    current = obs_dict.get("current") or {}
    select  = obs_dict.get("select")  or {}

    players = current.get("players", [{}, {}])
    opp_idx = 1 - my_idx
    me  = players[my_idx] if my_idx < len(players) else {}
    opp = players[opp_idx] if opp_idx < len(players) else {}

    # ── Global features ───────────────────────────────────────────────────
    turn          = current.get("turn", 0)
    my_prize_cnt  = sum(1 for p in (me.get("prize") or []) if p is not None)
    opp_prize_cnt = sum(1 for p in (opp.get("prize") or []) if p is not None)
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
        min(me.get("deckCount", 0), 60)  / 60.0,
        min(opp.get("deckCount", 0), 60) / 60.0,
        min(me.get("handCount", 0), 20)  / 20.0,
        min(opp.get("handCount", 0), 20) / 20.0,
    ], dtype=np.float32)

    # ── My board ──────────────────────────────────────────────────────────
    my_active_id, my_active_feat = _encode_slot_list(
        me.get("active", []), cfg, is_active=True
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
        opp.get("active", []), cfg, is_active=True
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
    option_ids, option_feat, option_mask = _encode_options(
        select.get("option", []), cfg
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
def _encode_pokemon(pokemon: Optional[dict], is_active: bool = False) -> tuple:
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
        if isinstance(e, int) and 0 <= e < 11:
            energy_counts[e] += 1
        elif isinstance(e, dict):
            v = e.get("value", -1)
            if 0 <= v < 11:
                energy_counts[v] += 1
    # Normalise (max 5 energy of one type is generous)
    energy_counts = np.clip(energy_counts / 5.0, 0.0, 1.0)

    feat = np.concatenate([
        np.array([hp_ratio], dtype=np.float32),
        energy_counts,
        np.zeros(5, dtype=np.float32),   # status: set by PlayerState, not Pokemon
        np.array([float(pokemon.get("appearThisTurn", False))], dtype=np.float32),
    ])
    return card_id, feat


def _encode_slot_list(
    slot_list: list,
    cfg: ModelConfig,
    is_active: bool = False,
) -> tuple:
    """
    Encode a size-0-or-1 slot list (active Pokémon).
    Returns (ids [1], feats [1, POKEMON_FEAT_DIM]).
    """
    ids   = np.zeros(1, dtype=np.int32)
    feats = np.zeros((1, POKEMON_FEAT_DIM), dtype=np.float32)
    if slot_list:
        cid, feat = _encode_pokemon(slot_list[0], is_active)
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


def _encode_options(options: list, cfg: ModelConfig) -> tuple:
    """
    Encode available options.
    Returns (ids [max_actions], feat [max_actions, OPTION_FEAT_DIM], mask [max_actions]).
    """
    A    = cfg.max_actions
    ids  = np.zeros(A, dtype=np.int32)
    feat = np.zeros((A, OPTION_FEAT_DIM), dtype=np.float32)
    mask = np.zeros(A, dtype=bool)

    for i, opt in enumerate(options[:A]):
        if opt is None:
            continue

        opt_type = _int_from(opt, "type", 0)
        area     = _int_from(opt, "area", 0)
        in_area  = _int_from(opt, "inPlayArea", 0)
        p_idx    = _int_from(opt, "playerIndex", 0)
        card_id  = _int_from(opt, "cardId", NO_CARD) or NO_CARD

        ids[i]  = card_id
        mask[i] = True

        # one-hot type (17), area (13), in_play_area (13), player_idx (1), has_card (1)
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
        f[off + 1] = float(card_id != NO_CARD)
        feat[i] = f

    return ids, feat, mask


# ─────────────────────────────────────────────────────────────────────────────
# Reward extraction from logs
# ─────────────────────────────────────────────────────────────────────────────
def extract_step_reward(logs: list, my_idx: int) -> float:
    """
    Scan the new logs since the last step and compute a dense reward signal.

    Reward shaping:
      +0.15  per prize card I take   (I KO'd opponent's Pokémon)
      -0.15  per prize card opponent takes
      +1.0   if I win
      -1.0   if I lose
    """
    reward = 0.0
    for log in logs:
        log_type = _int_from(log, "type", -1)
        # LogType.RESULT = 23
        if log_type == 23:
            result = _int_from(log, "result", 2)
            if result == my_idx:
                reward += 1.0
            elif result != 2:      # != draw
                reward -= 1.0
        # LogType.MOVE_CARD = 6: prize card moved to hand → prize taken
        # AreaType.PRIZE = 6, AreaType.HAND = 2
        if log_type == 6:
            from_area = _int_from(log, "fromArea", -1)
            to_area   = _int_from(log, "toArea", -1)
            p_idx     = _int_from(log, "playerIndex", -1)
            if from_area == 6 and to_area == 2:   # prize → hand
                if p_idx == my_idx:
                    reward += 0.15    # I took a prize
                else:
                    reward -= 0.15    # opponent took a prize
    return float(np.clip(reward, -1.0, 1.0))


def _int_from(obj, key: str, default: int) -> int:
    """Safely get an integer attribute from a dict or object."""
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    if isinstance(v, (int, float)):
        return int(v)
    # Enum-like: has .value
    if hasattr(v, "value"):
        return int(v.value)
    return default
