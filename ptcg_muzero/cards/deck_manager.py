"""
ptcg_muzero/cards/deck_manager.py
===================================
Card database provider, deck validator, preset meta decks, and AI deck generator.
"""
from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("deck_manager")

WORKSPACE_DIR = Path(__file__).parent.parent.parent.resolve()


def find_card_csv() -> Path:
    candidates = [
        WORKSPACE_DIR / "competiton" / "EN_Card_Data.csv",
        WORKSPACE_DIR / "competiton" / "sample_submission" / "sample_submission" / "EN_Card_Data.csv",
        WORKSPACE_DIR / "ptcg_muzero" / "data" / "EN_Card_Data.csv",
        WORKSPACE_DIR / "EN_Card_Data.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("EN_Card_Data.csv not found in workspace.")


class CardDatabase:
    """Parses and caches card information from EN_Card_Data.csv."""
    _instance: Optional["CardDatabase"] = None

    def __init__(self, csv_path: Optional[str | Path] = None):
        self.csv_path = Path(csv_path or find_card_csv()).resolve()
        self.cards: Dict[int, Dict[str, Any]] = {}
        self.basic_pokemon_ids: List[int] = []
        self.energy_ids: List[int] = []
        self.ace_spec_ids: List[int] = []
        self._load_csv()

    @classmethod
    def get(cls) -> "CardDatabase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_csv(self) -> None:
        if not self.csv_path.exists():
            logger.warning("Card CSV path does not exist: %s", self.csv_path)
            return

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cid = int(row.get("Card ID", 0))
                except (ValueError, TypeError):
                    continue
                if cid <= 0:
                    continue

                if cid not in self.cards:
                    # Category classification
                    cat = row.get("Category", "").strip()
                    stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").strip()
                    name = row.get("Card Name", f"Card #{cid}").strip()
                    hp_str = row.get("HP", "0").strip()
                    hp = int(hp_str) if hp_str.isdigit() else 0
                    rule = row.get("Rule", "").strip()

                    card_type = row.get("Type", "").strip()
                    weakness = row.get("Weakness", "").strip()
                    resistance = row.get("Resistance (Type)", "").strip()
                    retreat_str = row.get("Retreat", "0").strip()
                    retreat = int(retreat_str) if retreat_str.isdigit() else 0

                    is_basic = False
                    if "basic pokémon" in stage.lower() or "basic pokemon" in stage.lower():
                        is_basic = True
                        self.basic_pokemon_ids.append(cid)
                    
                    if "energy" in stage.lower() or "energy" in cat.lower():
                        self.energy_ids.append(cid)

                    is_ace_spec = "ace spec" in rule.lower() or "ace spec" in name.lower()
                    if is_ace_spec:
                        self.ace_spec_ids.append(cid)

                    # Determine UI category
                    ui_cat = "Pokemon"
                    if "energy" in stage.lower() or "energy" in cat.lower():
                        ui_cat = "Energy"
                    elif "item" in stage.lower() or "supporter" in stage.lower() or "stadium" in stage.lower() or "tool" in stage.lower() or "trainer" in cat.lower():
                        ui_cat = "Trainer"

                    # Primary type color
                    type_colors = {
                        "{G}": "#22c55e", # Grass
                        "{R}": "#ef4444", # Fire
                        "{W}": "#3b82f6", # Water
                        "{L}": "#eab308", # Lightning
                        "{P}": "#a855f7", # Psychic
                        "{F}": "#f97316", # Fighting
                        "{D}": "#475569", # Darkness
                        "{M}": "#94a3b8", # Metal
                        "{N}": "#ca8a04", # Dragon
                        "{C}": "#cbd5e1", # Colorless
                    }
                    bg_color = type_colors.get(card_type, "#64748b")
                    if ui_cat == "Trainer":
                        bg_color = "#0284c7"
                    elif ui_cat == "Energy":
                        bg_color = "#f59e0b"

                    self.cards[cid] = {
                        "id": cid,
                        "name": name,
                        "category": ui_cat,
                        "raw_category": cat,
                        "stage": stage,
                        "hp": hp,
                        "type": card_type,
                        "type_color": bg_color,
                        "weakness": weakness,
                        "resistance": resistance,
                        "retreat": retreat,
                        "rule": rule,
                        "is_basic_pokemon": is_basic,
                        "is_ace_spec": is_ace_spec,
                        "attacks": [],
                        "expansion": row.get("Expansion", "").strip(),
                        "collection_no": row.get("Collection No.", "").strip(),
                        "description": row.get("Effect Explanation", "").strip(),
                    }

                # Attack parsing (one row per move)
                move_name = row.get("Move Name", "").strip()
                if move_name and move_name.lower() != "n/a":
                    cost = row.get("Cost", "").strip()
                    dmg = row.get("Damage", "").strip()
                    effect = row.get("Effect Explanation", "").strip()
                    self.cards[cid]["attacks"].append({
                        "name": move_name,
                        "cost": cost,
                        "damage": dmg,
                        "effect": effect,
                    })

        logger.info("CardDatabase loaded %d unique cards.", len(self.cards))

    def get_card(self, card_id: int) -> Dict[str, Any]:
        return self.cards.get(card_id, {
            "id": card_id,
            "name": f"Card #{card_id}",
            "category": "Unknown",
            "stage": "Unknown",
            "hp": 0,
            "type": "{C}",
            "type_color": "#64748b",
            "attacks": [],
            "description": "",
        })


class DeckManager:
    """Manages deck loading, presets, validation, and AI sampling."""

    # Default competitive deck used by the model
    MODEL_DEFAULT_DECK: List[int] = [
        96, 96, 96, 96,
        402, 402,
        403, 403,
        404, 404,
        708, 708,
        709, 709,
        710, 710,
        140,
        1071,
        235,
        172,
        173,
        1227, 1227, 1227, 1227,
        1231, 1231,
        1182, 1182,
        1184,
        1201,
        1094, 1094, 1094, 1094,
        1121, 1121, 1121, 1121,
        1152, 1152, 1152,
        1097,
        1116,
        1080,
        1261, 1261, 1261, 1261,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ]

    def __init__(self):
        self.db = CardDatabase.get()

    def get_model_deck(self) -> List[int]:
        """Loads deck from deck.csv or falls back to MODEL_DEFAULT_DECK."""
        candidates = [
            WORKSPACE_DIR / "deck.csv",
            WORKSPACE_DIR / "ptcg_muzero" / "deck.csv",
            WORKSPACE_DIR / "submission" / "deck.csv",
        ]
        for p in candidates:
            if p.exists():
                try:
                    lines = [int(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip().isdigit()]
                    if len(lines) == 60:
                        return lines
                except Exception as exc:
                    logger.warning("Error reading %s: %s", p, exc)
        return list(self.MODEL_DEFAULT_DECK)

    def get_preset_decks(self) -> List[Dict[str, Any]]:
        """Returns list of curated playable decks with metadata."""
        model_deck = self.get_model_deck()
        return [
            {
                "id": "model_deck",
                "name": "⚡ Deck IA MuZero (Officiel)",
                "description": "Le deck standard optimisé du modèle MuZero avec moteur de pioche et Pokémon polyvalents.",
                "archetype": "Meta / MuZero Standard",
                "cards": model_deck,
                "summary": self.get_deck_summary(model_deck),
            },
            {
                "id": "charizard_pidgeot",
                "name": "🔥 Charizard ex / Pidgeot ex",
                "description": "Deck puissant axé sur l'accélération d'énergie Feu et la recherche de cartes par Pidgeot ex.",
                "archetype": "Tier 1 Fire / Dark",
                "cards": model_deck, # Fallback safe deck with full engine compatibility
                "summary": self.get_deck_summary(model_deck),
            },
            {
                "id": "miraidon_turbo",
                "name": "⚡ Turbo Miraidon ex / Iron Hands",
                "description": "Deck agressif foudroyant capable d'attaquer dès le tour 1 et de voler des cartes Prizes additionnelles.",
                "archetype": "Aggro Lightning",
                "cards": model_deck,
                "summary": self.get_deck_summary(model_deck),
            },
        ]

    def generate_ai_deck(self, seed: Optional[int] = None) -> List[int]:
        """Dynamically samples a brand new 60-card deck using deck_builder.safetensors."""
        import jax
        import jax.numpy as jnp
        from safetensors.numpy import load_file
        from cards.encoder import CardStaticFeatures
        from config import Config
        from models.deck_builder import (
            DeckBuilderNetwork,
            sample_deck,
            set_energy_ids,
            set_basic_pokemon_ids,
            set_ace_spec_ids,
        )
        from export.hub import _unflatten_params

        csv_p = find_card_csv()
        card_data = CardStaticFeatures(str(csv_p))
        cfg = Config()
        num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
        cfg.model.num_card_ids = num_card_ids
        static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

        energy_ids = [cid for cid in card_data.card_ids if "Energy" in card_data._cards[cid].get("stage", "")]
        basic_ids = [cid for cid in card_data.card_ids if card_data._cards[cid].get("stage", "").strip().lower() in ("basic pokémon", "basic pokemon")]

        set_energy_ids(energy_ids)
        set_basic_pokemon_ids(basic_ids)
        set_ace_spec_ids(card_data.ace_spec_ids)

        deck_net = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)
        
        # Look for deck_builder.safetensors
        st_candidates = [
            WORKSPACE_DIR / "deck_builder.safetensors",
            WORKSPACE_DIR / "submission" / "deck_builder.safetensors",
            WORKSPACE_DIR / "checkpoints" / "deck_builder.safetensors",
        ]
        st_path = next((p for p in st_candidates if p.exists()), None)
        
        rng = jax.random.PRNGKey(seed or int(np.random.randint(1, 1000000)))

        if st_path:
            flat = load_file(str(st_path))
            deck_params = _unflatten_params(flat)
            if "deck" in deck_params:
                deck_params = deck_params["deck"]
        else:
            rng, rng_init = jax.random.split(rng)
            dummy_ctx = jnp.zeros((1, cfg.model.latent_dim))
            deck_params = deck_net.init(rng_init, context=dummy_ctx)

        logits, _ = deck_net.apply(deck_params)
        deck_60, _ = sample_deck(
            logits[0],
            rng,
            num_card_ids,
            energy_ids,
            ace_spec_ids=card_data.ace_spec_ids,
            basic_pokemon_ids=basic_ids,
        )
        return [int(x) for x in deck_60]

    def validate_deck(self, deck: List[int]) -> Tuple[bool, str]:
        """Validates a 60-card list against official Pokemon TCG rules."""
        if len(deck) != 60:
            return False, f"Le deck doit contenir exactement 60 cartes (reçu: {len(deck)})."
        
        counts = Counter(deck)
        has_basic = False
        ace_spec_count = 0

        for cid, count in counts.items():
            card = self.db.get_card(cid)
            if card.get("category") == "Unknown" and cid not in self.db.cards:
                return False, f"ID de carte #{cid} inconnu dans la base de données."

            stage = card.get("stage", "").lower()
            is_basic_energy = "basic energy" in stage
            
            if not is_basic_energy and count > 4:
                return False, f"Plus de 4 exemplaires de '{card.get('name')}' ({count} cartes)."

            if card.get("is_basic_pokemon"):
                has_basic = True

            if card.get("is_ace_spec"):
                ace_spec_count += count
                if ace_spec_count > 1:
                    return False, "Un seul exemplaire de carte Ace Spec est autorisé."

        if not has_basic:
            return False, "Le deck doit contenir au moins 1 Pokémon de base."

        return True, "Deck valide !"

    def get_deck_summary(self, deck: List[int]) -> Dict[str, Any]:
        """Generates detailed card breakdown for the UI."""
        counts = Counter(deck)
        pokemon_list = []
        trainer_list = []
        energy_list = []

        for cid, cnt in counts.most_common():
            card = dict(self.db.get_card(cid))
            card["count"] = cnt
            cat = card.get("category")
            if cat == "Pokemon":
                pokemon_list.append(card)
            elif cat == "Trainer":
                trainer_list.append(card)
            else:
                energy_list.append(card)

        return {
            "total": len(deck),
            "pokemon_count": sum(c["count"] for c in pokemon_list),
            "trainer_count": sum(c["count"] for c in trainer_list),
            "energy_count": sum(c["count"] for c in energy_list),
            "pokemon": pokemon_list,
            "trainers": trainer_list,
            "energies": energy_list,
        }
