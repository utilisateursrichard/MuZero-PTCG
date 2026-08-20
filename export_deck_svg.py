#!/usr/bin/env python3
"""
export_deck_svg.py
==================
Script pour charger 'deck_builder.safetensors' (ou un checkpoint),
échantillonner un deck de 60 cartes Pokémon TCG, et l'exporter sous forme
de visuel SVG moderne (deck.svg).

Usage :
    python export_deck_svg.py --safetensors /chemin/vers/deck_builder.safetensors --out deck.svg
"""

from __future__ import annotations

import argparse
import logging
import sys

# Flush logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("export_deck_svg")

from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.resolve()
MUZERO_DIR = SCRIPT_DIR / "ptcg_muzero"
if MUZERO_DIR.exists():
    sys.path.insert(0, str(MUZERO_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter

from cards.encoder import CardStaticFeatures
from config import Config
from models.deck_builder import (
    DeckBuilderNetwork, sample_deck, set_energy_ids, set_basic_pokemon_ids, set_ace_spec_ids
)
from export.hub import _unflatten_params


def find_card_csv() -> Path:
    candidates = [
        SCRIPT_DIR / "competiton" / "EN_Card_Data.csv",
        SCRIPT_DIR / "competiton" / "sample_submission" / "sample_submission" / "EN_Card_Data.csv",
        SCRIPT_DIR / "ptcg_muzero" / "data" / "EN_Card_Data.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("EN_Card_Data.csv not found in workspace.")


def generate_deck_svg(cards_info: list[dict], out_path: Path) -> None:
    """Generates an SVG file representing the 60-card deck."""
    # Group by category
    pokemon = [c for c in cards_info if c["category"] == "Pokemon"]
    trainers = [c for c in cards_info if c["category"] == "Trainer"]
    energies = [c for c in cards_info if c["category"] == "Energy"]
    others   = [c for c in cards_info if c["category"] not in ("Pokemon", "Trainer", "Energy")]

    sections = [
        ("Pokémon", pokemon, "#e53e3e"),
        ("Trainers", trainers, "#3182ce"),
        ("Energies", energies, "#d69e2e"),
    ]
    if others:
        sections.append(("Others", others, "#805ad5"))

    # Layout metrics
    width = 1000
    card_height = 42
    header_height = 90
    
    total_items = sum(len(items) for _, items, _ in sections)
    total_card_count = sum(c["count"] for c in cards_info)

    # Calculate SVG height
    calculated_height = header_height + len(sections) * 45 + total_items * (card_height + 8) + 40
    height = max(calculated_height, 600)

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">')
    svg_parts.append('''
    <defs>
        <style>
            .bg { fill: #0f172a; }
            .header-title { font-family: system-ui, -apple-system, sans-serif; font-size: 26px; font-weight: 800; fill: #f8fafc; }
            .header-sub { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; fill: #94a3b8; }
            .sec-title { font-family: system-ui, -apple-system, sans-serif; font-size: 18px; font-weight: 700; }
            .card-bg { fill: #1e293b; rx: 8px; ry: 8px; stroke: #334155; stroke-width: 1.5; }
            .card-name { font-family: system-ui, -apple-system, sans-serif; font-size: 15px; font-weight: 600; fill: #f1f5f9; }
            .card-detail { font-family: system-ui, -apple-system, sans-serif; font-size: 13px; fill: #94a3b8; }
            .badge-bg { rx: 12px; ry: 12px; }
            .badge-txt { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; font-weight: 800; fill: #0f172a; text-anchor: middle; }
            .id-txt { font-family: monospace; font-size: 12px; fill: #64748b; }
        </style>
        <linearGradient id="grad-header" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.2"/>
            <stop offset="100%" stop-color="#818cf8" stop-opacity="0.05"/>
        </linearGradient>
    </defs>
    ''')

    # Background
    svg_parts.append(f'<rect width="{width}" height="{height}" class="bg" />')

    # Header
    svg_parts.append(f'<rect x="20" y="20" width="{width - 40}" height="60" rx="10" fill="url(#grad-header)" stroke="#38bdf8" stroke-opacity="0.3" stroke-width="1" />')
    svg_parts.append(f'<text x="40" y="58" class="header-title">⚡ PTCG MuZero Deck</text>')
    svg_parts.append(f'<text x="{width - 40}" y="56" class="header-sub" text-anchor="end">Total cards: <tspan font-weight="bold" fill="#38bdf8">{total_card_count}</tspan> / 60</text>')

    curr_y = header_height + 10

    for sec_title, items, color in sections:
        if not items:
            continue
        
        sec_count = sum(c["count"] for c in items)
        svg_parts.append(f'<text x="30" y="{curr_y + 20}" class="sec-title" fill="{color}">● {sec_title} ({sec_count})</text>')
        svg_parts.append(f'<line x1="30" y1="{curr_y + 30}" x2="{width - 30}" y2="{curr_y + 30}" stroke="{color}" stroke-opacity="0.3" stroke-width="2" />')
        curr_y += 45

        for c in items:
            # Layout de carte en ligne
            card_w = width - 60
            svg_parts.append(f'<g transform="translate(30, {curr_y})">')
            svg_parts.append(f'<rect width="{card_w}" height="{card_height}" class="card-bg" />')
            
            # Badge nombre d'exemplaires (ex: 4x)
            svg_parts.append(f'<rect x="10" y="8" width="36" height="26" class="badge-bg" fill="{color}" />')
            svg_parts.append(f'<text x="28" y="25" class="badge-txt">{c["count"]}x</text>')

            # Nom de la carte
            name_str = c["name"]
            if c.get("is_ex"):
                name_str += " ex"
            svg_parts.append(f'<text x="60" y="26" class="card-name">{name_str}</text>')

            # Détails (HP, Stage, Type)
            details = []
            if c.get("stage"):
                details.append(f'Stage: {c["stage"]}')
            if c.get("hp") and c["hp"] > 0:
                details.append(f'HP: {int(c["hp"])}')
            if c.get("energy_type"):
                details.append(f'Type: {c["energy_type"]}')
            
            detail_str = " | ".join(details)
            if detail_str:
                svg_parts.append(f'<text x="320" y="26" class="card-detail">{detail_str}</text>')

            # ID de la carte
            svg_parts.append(f'<text x="{card_w - 20}" y="26" class="id-txt" text-anchor="end">ID #{c["id"]}</text>')

            svg_parts.append('</g>')
            curr_y += card_height + 8
        
        curr_y += 15

    svg_parts.append('</svg>')
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")
    logger.info("✓ Deck SVG exported successfully to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Generates deck.svg from deck_builder.safetensors")
    parser.add_argument("--safetensors", "--ckpt", default=None, help="Path to .safetensors file")
    parser.add_argument("--out", default="deck.svg", help="Output SVG file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deck sampling")
    args = parser.parse_args()

    csv_path = find_card_csv()
    logger.info("Loading cards from %s...", csv_path)
    card_data = CardStaticFeatures(csv_path)

    cfg = Config()
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    energy_ids = [cid for cid in card_data.card_ids if "Energy" in card_data._cards[cid].get("stage", "")]
    basic_ids  = [cid for cid in card_data.card_ids if card_data._cards[cid].get("stage", "").strip().lower() in ("basic pokémon", "basic pokemon")]

    set_energy_ids(energy_ids)
    set_basic_pokemon_ids(basic_ids)
    set_ace_spec_ids(card_data.ace_spec_ids)

    deck_net = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)

    rng = jax.random.PRNGKey(args.seed)

    if args.safetensors and Path(args.safetensors).exists():
        logger.info("Loading weights from %s...", args.safetensors)
        from safetensors.numpy import load_file
        flat = load_file(args.safetensors)
        deck_params = _unflatten_params(flat)
        # If the root is {"deck": ...}
        if "deck" in deck_params and isinstance(deck_params["deck"], dict):
            deck_params = deck_params["deck"]
    else:
        logger.info("No weights provided (or not found). Using randomly initialized parameters.")
        rng, rng_init = jax.random.split(rng)
        dummy_ctx = jnp.zeros((1, cfg.model.latent_dim))
        deck_params = deck_net.init(rng_init, context=dummy_ctx)

    logits, _ = deck_net.apply(deck_params)
    deck_60, _ = sample_deck(logits[0], rng, num_card_ids, energy_ids)

    # Compter les occurrences
    counts = Counter(deck_60)

    cards_info = []
    for cid, count in counts.most_common():
        info = card_data._cards.get(cid, {})
        category = "Pokemon"
        stage_raw = info.get("stage", "").lower()
        if "energy" in stage_raw:
            category = "Energy"
        elif any(k in stage_raw for k in ("trainer", "item", "supporter", "stadium", "tool")):
            category = "Trainer"

        cards_info.append({
            "id": cid,
            "count": count,
            "name": info.get("name", f"Card #{cid}"),
            "category": category,
            "stage": info.get("stage", ""),
            "hp": info.get("hp", 0),
            "energy_type": info.get("energy_type", ""),
            "is_ex": info.get("is_ex", False),
        })

    out_path = Path(args.out)
    generate_deck_svg(cards_info, out_path)


if __name__ == "__main__":
    main()
