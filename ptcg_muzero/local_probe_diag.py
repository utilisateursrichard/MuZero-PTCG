#!/usr/bin/env python
"""
ptcg_muzero/local_probe_diag.py
================================
Script standalone permettant d'exécuter un diagnostic de précision des probes 
sur de vraies parties simulées en local, sans dépendances vers le chemin /kaggle/.
"""

import os
import sys
import argparse
from pathlib import Path

# Ajouter le dossier ptcg_muzero au PYTHONPATH
sys.path.append(str(Path(__file__).parent.resolve()))

import jax
import jax.numpy as jnp
import numpy as np
import pickle

from config import Config
from cards.encoder import CardStaticFeatures
from models.networks import MuZeroNetwork
from models.deck_builder import DeckBuilderNetwork, sample_deck, set_basic_pokemon_ids, set_energy_ids, set_ace_spec_ids
from interpretability.probes import ProbeHeads, probe_accuracy, probe_loss, probe_report, extract_probe_targets
from training.trainer import make_agent_fn
from env.wrapper import run_self_play_game

def resolve_local_path(original_path: str) -> str:
    """Résout intelligemment les chemins Kaggle vers les fichiers locaux du workspace."""
    p = Path(original_path)
    if p.exists():
        return str(p.resolve())
    
    # Options de recherche dans le workspace local
    basename = p.name
    workspace_root = Path(__file__).parent.parent
    options = [
        workspace_root / "competiton" / basename,
        workspace_root / "competiton" / "sample_submission" / basename,
        workspace_root / "competiton" / "sample_submission" / "sample_submission" / basename,
        Path(".") / "competiton" / basename,
        Path(".") / "competiton" / "sample_submission" / basename,
        Path("/home/richard/Downloads/files/competiton") / basename,
        Path("/home/richard/Downloads/files/competiton/sample_submission") / basename,
    ]
    for opt in options:
        if opt.exists():
            return str(opt.resolve())
            
    print(f"Attention: Impossible de localiser le fichier {original_path} ou ses alternatives locales.")
    return original_path

def main():
    parser = argparse.ArgumentParser(description="Diagnostic local de la précision des Probes sur de vraies parties.")
    parser.add_argument("--ckpt", default=None, help="Chemin vers le checkpoint .pkl du modèle.")
    parser.add_argument("--config", default=None, help="Chemin vers config.json.")
    parser.add_argument("--games", type=int, default=3, help="Nombre de parties à simuler pour récolter des données.")
    parser.add_argument("--seed", type=int, default=42, help="Seed pour la génération aléatoire.")
    args = parser.parse_args()

    # 1. Chargement de la configuration
    if args.config and Path(args.config).exists():
        cfg = Config.load(args.config)
        print(f"Configuration chargée depuis {args.config}")
    else:
        cfg = Config()
        print("Utilisation de la configuration par défaut.")

    # Surcharger les chemins Kaggle par les chemins locaux résolus
    cfg.infra.card_csv = resolve_local_path(cfg.infra.card_csv)
    cfg.infra.reference_deck_csv = resolve_local_path(cfg.infra.reference_deck_csv)
    print(f"Chemin local card_csv : {cfg.infra.card_csv}")
    print(f"Chemin local reference_deck_csv : {cfg.infra.reference_deck_csv}")

    # 2. Chargement du checkpoint ou initialisation aléatoire
    params = {}
    deck_params = {}
    if args.ckpt and Path(args.ckpt).exists():
        with open(args.ckpt, "rb") as f:
            data = pickle.load(f)
        print(f"Checkpoint chargé depuis {args.ckpt} (step={data.get('step', -1)})")
        params = jax.tree_util.tree_map(jax.device_put, data["params"])
        deck_params = jax.tree_util.tree_map(jax.device_put, data.get("deck", {}))
    else:
        print("Aucun checkpoint valide fourni. Les probes seront évaluées sur un modèle aux poids aléatoires.")

    # 3. Initialisation des caractéristiques statiques des cartes
    card_data = CardStaticFeatures(cfg.infra.card_csv)
    num_card_ids = max(card_data.max_card_id + 1, cfg.model.num_card_ids)
    cfg.model.num_card_ids = num_card_ids
    static_jax = jnp.array(card_data.feature_matrix(num_card_ids))

    energy_ids = [
        cid for cid in card_data.card_ids
        if "Energy" in card_data._cards[cid].get("stage", "")
    ]
    set_energy_ids(energy_ids)
    set_basic_pokemon_ids([
        cid for cid in card_data.card_ids
        if card_data._cards[cid].get("stage", "").strip().lower()
        in ("basic pokémon", "basic pokemon")
    ])
    set_ace_spec_ids(card_data.ace_spec_ids)

    # 4. Initialisation des réseaux Flax
    network = MuZeroNetwork(cfg=cfg.model, static_features=static_jax)
    deck_net = DeckBuilderNetwork(cfg=cfg.model, static_features=static_jax)
    probes = ProbeHeads(cfg=cfg.model)

    rng = jax.random.PRNGKey(args.seed)

    # Si pas de paramètres chargés, initialisation avec des poids aléatoires
    if not params or not deck_params:
        from training.trainer import _make_dummy_obs
        dummy_obs = _make_dummy_obs(cfg.model)
        batch_obs = {k: jnp.array(v[None]) for k, v in dummy_obs.items()}
        rng, rng_mz, rng_pr, rng_dk = jax.random.split(rng, 4)
        
        if not params:
            mz_params = network.init(rng_mz, batch_obs, method=network.init_all)
            z_dummy = jnp.zeros((1, cfg.model.latent_dim))
            pr_params = probes.init(rng_pr, z_dummy)
            params = {"muzero": mz_params, "probes": pr_params}
        if not deck_params:
            dummy_ctx = jnp.zeros((1, cfg.model.latent_dim))
            deck_params = deck_net.init(rng_dk, context=dummy_ctx)

    # 5. Définition des agents pour la simulation
    # Agent 0 : MuZero (avec le checkpoint chargé)
    # Agent 1 : Aléatoire
    agent_muzero = make_agent_fn(network, params, cfg, rng, train_mode=False)

    def random_agent(obs_dict, player_idx, _cfg):
        opts = (obs_dict.get("select") or {}).get("option", [])
        n = max(len(opts), 1)
        # uniform policy representation
        policy = np.ones(cfg.model.max_actions) / cfg.model.max_actions
        return [np.random.randint(0, n)], policy, 0.0

    # 6. Simulation des parties pour collecter de vraies observations/cibles
    print(f"\nSimulation de {args.games} parties en cours pour récolter des données réelles...")
    all_observations = []
    all_targets = []

    np_rng = np.random.default_rng(args.seed)
    
    # Récupérer les decks
    deck_logits, _ = deck_net.apply(deck_params)
    deck0, _ = sample_deck(deck_logits[0], rng, num_card_ids, energy_ids)
    deck1 = deck0[:]  # même deck pour l'adversaire

    for g in range(args.games):
        h0, h1 = run_self_play_game(
            (agent_muzero, random_agent), deck0, deck1, cfg, np_rng
        )
        print(f"  Partie {g+1}/{args.games} terminée (états récoltés: {len(h0.observations)})")

        for obs, raw_state in zip(h0.observations, h0.raw_states):
            # Extraire les vraies cibles à partir des états bruts du jeu
            targets = extract_probe_targets(raw_state, my_idx=0)
            # Ne conserver le step que si au moins une cible est valide (différente de -1)
            if np.any(targets >= 0):
                all_observations.append(obs)
                all_targets.append(targets)

    if not all_observations:
        print("Erreur: Aucune donnée valide collectée pendant la simulation.")
        return

    # 7. Batcher les données récoltées
    keys = all_observations[0].keys()
    obs_batch = {k: jnp.stack([obs[k] for obs in all_observations], axis=0) for k in keys}
    target_batch = jnp.stack(all_targets, axis=0)

    print(f"\nCalcul des performances sur {len(all_observations)} états de jeu réels...")

    # 8. Forward pass et calcul des métriques
    z = network.apply(params["muzero"], obs_batch, method=network.represent)
    probe_logits = probes.apply(params["probes"], z)

    # Calcul des accuracies et de la loss sur les vraies targets
    accs = np.array(probe_accuracy(probe_logits, target_batch))
    loss, per_probe_losses = probe_loss(probe_logits, target_batch)

    # 9. Affichage du rapport
    print("\n" + probe_report(accs, np.array(per_probe_losses)))
    print(f"Total Probe Loss: {loss:.4f}\n")

if __name__ == "__main__":
    main()
