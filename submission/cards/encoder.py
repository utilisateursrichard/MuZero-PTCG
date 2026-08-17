"""
ptcg_muzero/cards/encoder.py
=============================
Two responsibilities:

1. **CardStaticFeatures** – parses cards.csv (multi-row per card, one row per
   move) into a fixed numpy matrix of shape [num_card_ids × CARD_STATIC_DIM].
   The matrix is built once at startup and frozen.

2. **CardEmbedding** – Flax module that concatenates a *trainable* embedding
   (nn.Embed) with the frozen static features to produce rich card tokens.

CSV expected columns (order may vary):
    Card ID, Card Name, Expansion, Collection No.,
    Stage (Pokémon)/Type (Energy and Trainer), Rule, Category,
    Previous stage, HP, Type, Weakness, Resistance (Type), Retreat,
    Move Name, Cost, Damage, Effect Explanation
"""
from __future__ import annotations

import csv
import glob
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary mappings
# ─────────────────────────────────────────────────────────────────────────────
_ENERGY_SYM: Dict[str, int] = {
    "{C}": 0, "{G}": 1, "{R}": 2, "{W}": 3, "{L}": 4,
    "{P}": 5, "{F}": 6, "{D}": 7, "{M}": 8, "{N}": 9,
    "{Y}": 10,
    "竜": 9,
    "{A}": 0, "{A}{A}": 0,
    "{C}{C}": 0, "{C}{C}{C}": 0,
    "{Team Rocket}": 0, "{Team Rocket}{Team Rocket}": 0,
}
NUM_ENERGY_TYPES = 11   # indices 0-10; -1 means "none / n/a"

_STAGE_MAP: Dict[str, int] = {
    "basic":                 0,
    "basic pokémon":         0,
    "basic pokemon":         0,
    "stage 1":               1,
    "stage 1 pokémon":       1,
    "stage 1 pokemon":       1,
    "stage 2":               2,
    "stage 2 pokémon":       2,
    "stage 2 pokemon":       2,
    "basic energy":          3,
    "special energy":        4,
    "item":                  5,
    "supporter":             5,
    "stadium":               5,
    "tool":                  5,
    "pokémon tool":          5,
    "pokemon tool":          5,
    "trainer":               5,
}
NUM_STAGES = 6    # 0-5; 5 = generic trainer/unknown

MAX_HP      = 340.0
MAX_RETREAT = 5.0
MAX_DAMAGE  = 320.0
MAX_ATTACKS = 4.0

# CARD_STATIC_DIM = NUM_STAGES + 3*NUM_ENERGY_TYPES + 2 + 7 = 6 + 33 + 2 + 7 = 48
CARD_STATIC_DIM: int = NUM_STAGES + 3 * NUM_ENERGY_TYPES + 2 + 7


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def _parse_energy_sym(s: str) -> int:
    """Return energy index for a symbol like '{G}', '竜', or -1 if absent."""
    if not s or s.strip().lower() in ("n/a", "none", ""):
        return -1
    st = s.strip()
    if st in _ENERGY_SYM:
        return _ENERGY_SYM[st]
    if "竜" in st or "{N}" in st:
        return 9
    for sym, idx in _ENERGY_SYM.items():
        if sym in st:
            return idx
    return -1


def _parse_float(s: str, default: float = 0.0) -> float:
    """Extract the first numeric run from a string (handles '120+' etc.)."""
    digits = "".join(c for c in s if c.isdigit() or c == ".")
    try:
        return float(digits)
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# CSV parser
# ─────────────────────────────────────────────────────────────────────────────
class CardStaticFeatures:
    """
    Parses cards.csv and exposes a numpy feature matrix.

    Usage::

        feats = CardStaticFeatures("data/cards.csv")
        matrix = feats.feature_matrix(num_card_ids=600)
        # matrix[card_id]  →  float32 vector of length CARD_STATIC_DIM
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._cards: Dict[int, dict] = {}
        self._load(Path(csv_path))
        logger.info(
            "CardStaticFeatures: loaded %d unique cards (max_id=%d)",
            len(self._cards), self.max_card_id,
        )

    # ── Public ────────────────────────────────────────────────────────────

    @property
    def card_ids(self) -> List[int]:
        return sorted(self._cards.keys())

    @property
    def max_card_id(self) -> int:
        return max(self._cards.keys()) if self._cards else 0

    @property
    def ace_spec_ids(self) -> List[int]:
        """IDs des cartes Ace Spec (limite d'1 exemplaire par deck)."""
        return [cid for cid, info in self._cards.items() if info.get("is_ace")]

    def feature_matrix(self, num_card_ids: int) -> np.ndarray:
        """
        Return float32 array of shape [num_card_ids, CARD_STATIC_DIM].
        Row 0 is reserved as the "no card / padding" token (all zeros).
        """
        assert num_card_ids > self.max_card_id, (
            f"num_card_ids={num_card_ids} must exceed max card id {self.max_card_id}. "
            "Increase ModelConfig.num_card_ids."
        )
        mat = np.zeros((num_card_ids, CARD_STATIC_DIM), dtype=np.float32)
        for cid, info in self._cards.items():
            if cid < num_card_ids:
                mat[cid] = self._encode(info)
        return mat

    def card_name(self, card_id: int) -> str:
        return self._cards.get(card_id, {}).get("name", f"<unknown:{card_id}>")

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        if not path.exists():
            workspace_root = Path(__file__).resolve().parent.parent.parent
            candidates = [
                Path("/kaggle/input/competitions/pokemon-tcg-ai-battle/EN Card Data.csv"),
                Path("/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv"),
                Path("competiton/EN Card Data.csv"),
                Path("competiton/EN_Card_Data.csv"),
                Path("competition/EN Card Data.csv"),
                Path("competition/EN_Card_Data.csv"),
                workspace_root / "competiton" / "EN Card Data.csv",
                workspace_root / "competiton" / "EN_Card_Data.csv",
                workspace_root / "competition" / "EN Card Data.csv",
                workspace_root / "competition" / "EN_Card_Data.csv",
                workspace_root / "ptcg_muzero" / "data" / "EN Card Data.csv",
                workspace_root / "ptcg_muzero" / "data" / "EN_Card_Data.csv",
                Path("/root/workspace/competiton/EN_Card_Data.csv"),
            ]
            for cand in candidates:
                if cand.exists():
                    logger.info("CardStaticFeatures: fallback path found at %s", cand)
                    path = cand
                    break

            if not path.exists():
                kaggle_matches = glob.glob("/kaggle/input/**/EN*Card*Data.csv", recursive=True) + glob.glob("/kaggle/input/**/*Card*Data*.csv", recursive=True)
                if kaggle_matches:
                    logger.info("CardStaticFeatures: fallback path found via Kaggle glob at %s", kaggle_matches[0])
                    path = Path(kaggle_matches[0])

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid_str = row.get("Card ID", "").strip()
                if not cid_str.isdigit():
                    continue
                cid = int(cid_str)

                if cid not in self._cards:
                    stage_raw = row.get(
                        "Stage (Pokémon)/Type (Energy and Trainer)", ""
                    ).strip()
                    rule_raw   = row.get("Rule", "").strip().lower()
                    name_raw   = row.get("Card Name", "").strip()

                    self._cards[cid] = {
                        "name":            name_raw,
                        "stage":           stage_raw,
                        "energy_type":     row.get("Type", "").strip(),
                        "weakness_type":   row.get("Weakness", "").strip(),
                        "resistance_type": row.get("Resistance (Type)", "").strip(),
                        "hp":              _parse_float(row.get("HP", "0")),
                        "retreat":         _parse_float(row.get("Retreat", "0")),
                        "is_ex":    ("ex" in rule_raw or " ex" in name_raw.lower()),
                        "is_tera":  ("tera" in rule_raw or "tera " in name_raw.lower()),
                        "is_ace":   ("ace spec" in rule_raw or stage_raw.strip().lower() == "ace spec"),
                        "damages":  [],
                    }

                # Aggregate damage values across all move rows
                dmg_str = row.get("Damage", "").strip()
                move     = row.get("Move Name", "").strip()
                if move and move.lower() not in ("n/a", ""):
                    dmg = _parse_float(dmg_str)
                    self._cards[cid]["damages"].append(dmg)

    def _encode(self, info: dict) -> np.ndarray:
        """Produce the fixed-length CARD_STATIC_DIM feature vector for one card."""
        parts: List[np.ndarray] = []

        # [6] stage one-hot
        stage_key = info["stage"].strip().lower()
        stage_idx = _STAGE_MAP.get(stage_key, 5)
        parts.append(_one_hot(stage_idx, NUM_STAGES))

        # [11] Pokémon / energy type
        et_idx = _parse_energy_sym(info["energy_type"])
        parts.append(_one_hot(et_idx, NUM_ENERGY_TYPES))

        # [11] Weakness type  +  [1] has_weakness
        w_idx = _parse_energy_sym(info["weakness_type"])
        parts.append(_one_hot(w_idx, NUM_ENERGY_TYPES))
        parts.append(np.array([float(w_idx >= 0)], dtype=np.float32))

        # [11] Resistance type  +  [1] has_resistance
        r_idx = _parse_energy_sym(info["resistance_type"])
        parts.append(_one_hot(r_idx, NUM_ENERGY_TYPES))
        parts.append(np.array([float(r_idx >= 0)], dtype=np.float32))

        # [7] numeric scalars
        damages = info["damages"]
        parts.append(np.array([
            info["hp"]      / MAX_HP,
            info["retreat"] / MAX_RETREAT,
            len(damages)    / MAX_ATTACKS,
            (max(damages) if damages else 0.0) / MAX_DAMAGE,
            float(info["is_ex"]),
            float(info["is_tera"]),
            float(info["is_ace"]),
        ], dtype=np.float32))

        feat = np.concatenate(parts)
        assert feat.shape == (CARD_STATIC_DIM,), feat.shape
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# JAX / Flax embedding module
# ─────────────────────────────────────────────────────────────────────────────
class CardEmbedding(nn.Module):
    """
    Combines a *trainable* nn.Embed with frozen static CSV features.

    Output dim = ``card_embed_dim + CARD_STATIC_DIM``.

    The static feature matrix is passed as a frozen jnp.ndarray so it is
    never part of the parameter tree (saves memory + keeps gradients clean).
    """
    num_card_ids: int
    embed_dim: int
    static_features: jnp.ndarray  # [num_card_ids, CARD_STATIC_DIM] – frozen

    @nn.compact
    def __call__(self, card_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            card_ids: int32 array of arbitrary shape [...].
        Returns:
            float32 array [..., embed_dim + CARD_STATIC_DIM].
        """
        learned = nn.Embed(
            num_embeddings=self.num_card_ids,
            features=self.embed_dim,
            name="card_embed",
        )(card_ids)                          # [..., embed_dim]

        static = self.static_features[card_ids]   # [..., CARD_STATIC_DIM]
        return jnp.concatenate([learned, static], axis=-1)
