#!/usr/bin/env python3
"""
analyze_attack_transitions.py
=============================
Analyse diagnostique approfondie des transitions "ATTACK" et "ATTACH" dans le Replay Buffer MuZero.

Ce script :
1. Isole strictement les vraies cartes Énergie (Basic Energy, Special Energy) en excluant les Outils (Tools/TMs).
2. Reconstitue les épisodes complets du Replay Buffer pour suivre si un attachement d'énergie
   débouche sur une attaque sur TOUT L'ÉPISODE (pas seulement dans la fenêtre unroll).
3. Calcule le délai (nombre de steps) entre l'attachement d'énergie et l'attaque.
4. Analyse les distributions de récompenses, valeurs cibles z0/z1, politiques MCTS et opportunités manquées.

Usage:
------
    export HF_TOKEN="hf_..."
    .venv_gpu/bin/python ptcg_muzero/analyze_attack_transitions.py
"""

import os
import sys
from pathlib import Path

# ── Injection automatique des chemins de modules pour le dé-pickling ────────
CURRENT_DIR = Path(__file__).resolve().parent
POSSIBLE_ROOTS = [
    CURRENT_DIR,
    CURRENT_DIR.parent,
    Path("/home/richard/Downloads/files/ptcg_muzero"),
    Path("/home/richard/Downloads/files"),
    Path("/kaggle/working/ptcg_muzero"),
    Path("/kaggle/working"),
]
for p in POSSIBLE_ROOTS:
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    import training.replay_buffer
    from training.replay_buffer import ReplayEntry, PrioritizedReplayBuffer
except Exception:
    pass

import json
import pickle
import argparse
from collections import Counter, defaultdict

import numpy as np

OPTION_TYPE_NAMES = {
    0: "NONE",
    1: "NUMBER",
    2: "YES_NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SPECIAL_COND",
    16: "SKILL",
}

def get_opt_name(opt_type: int) -> str:
    return OPTION_TYPE_NAMES.get(opt_type, f"TYPE_{opt_type}")


def format_stats(arr: list | np.ndarray) -> dict:
    if len(arr) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    a = np.array(arr, dtype=np.float64)
    return {
        "count": len(a),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
    }


def download_buffer_from_hf(repo_id: str, token: str | None = None, local_dir: str = "./hf_download") -> Path:
    from huggingface_hub import hf_hub_download
    
    os.makedirs(local_dir, exist_ok=True)
    print(f"[*] Downloading buffer from HuggingFace Hub ({repo_id})...")
    
    try:
        meta_path = hf_hub_download(
            repo_id=repo_id,
            filename="buffer_meta.json",
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"    ✓ Metadata found: Step {meta.get('step', '?')} | {meta.get('size', '?')}/{meta.get('max_size', '?')} entries ({meta.get('fill_percentage', '?')}%) | Date: {meta.get('iso_date', '?')}")
    except Exception as e:
        print(f"    (i) buffer_meta.json not found or inaccessible: {e}")

    pkl_path = hf_hub_download(
        repo_id=repo_id,
        filename="replay_buffer.pkl",
        token=token,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"    ✓ Replay buffer downloaded successfully: {pkl_path} ({os.path.getsize(pkl_path) / (1024*1024):.1f} MB)")
    return Path(pkl_path)


def load_card_database():
    """Charge les métadonnées complètes des cartes (nom, type, stage)."""
    cards_info = {}
    try:
        from cards.encoder import CardStaticFeatures
        for path_cand in ["competiton/EN_Card_Data.csv", "competiton/EN Card Data.csv", "/home/richard/Downloads/files/competiton/EN_Card_Data.csv"]:
            if Path(path_cand).exists():
                csf = CardStaticFeatures(path_cand)
                cards_info = csf._cards
                print(f"[*] Card database loaded ({len(cards_info)} cards via CardStaticFeatures)")
                return cards_info
    except Exception:
        pass

    possible_paths = [
        Path("competiton/EN_Card_Data.csv"),
        Path("competiton/EN Card Data.csv"),
        Path("../competiton/EN_Card_Data.csv"),
        Path("/home/richard/Downloads/files/competiton/EN_Card_Data.csv"),
        Path("/kaggle/working/ptcg_muzero/competiton/EN_Card_Data.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            try:
                import csv
                with open(p, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cid_str = row.get("Card ID", "").strip() or row.get("cardId", "").strip() or row.get("id", "").strip()
                        if cid_str.isdigit():
                            cid = int(cid_str)
                            name = row.get("Card Name", "").strip() or row.get("name", "").strip() or ""
                            stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").strip() or row.get("stage", "").strip()
                            if cid not in cards_info and name:
                                cards_info[cid] = {
                                    "name": name,
                                    "stage": stage,
                                    "is_ace": "ace spec" in row.get("Rule", "").lower() or stage.lower() == "ace spec",
                                }
                print(f"[*] Card database loaded ({len(cards_info)} cards from {p})")
                return cards_info
            except Exception:
                pass
    return cards_info


def is_true_energy_card(card_id: int, cards_info: dict) -> bool:
    """Vérifie si la carte est STRICTEMENT une vraie carte Énergie (Basic ou Special Energy), hors Outils."""
    if card_id not in cards_info:
        return True  # Fallback neutre si ID non trouvé
    info = cards_info[card_id]
    stage = str(info.get("stage", "")).strip().lower()
    name = str(info.get("name", "")).strip().lower()
    
    # Exclure explicitement les outils et items
    if "tool" in stage or "item" in stage or "supporter" in stage or "stadium" in stage:
        return False
    if "machine" in name or "charm" in name or "belt" in name or "band" in name or "cape" in name:
        return False
        
    return "energy" in stage or "énergie" in stage or "energy" in name.lower()


def is_tool_card(card_id: int, cards_info: dict) -> bool:
    """Vérifie si la carte est un Outil (Tool/TM/Item attaché)."""
    if card_id not in cards_info:
        return False
    info = cards_info[card_id]
    stage = str(info.get("stage", "")).strip().lower()
    name = str(info.get("name", "")).strip().lower()
    return "tool" in stage or "item" in stage or "machine" in name or "charm" in name or "belt" in name


def reconstruct_episodes(entries):
    """
    Reconstitue les épisodes séquentiels complets à partir des transitions du Replay Buffer.
    Retourne une liste d'épisodes, où chaque épisode est une liste de dicts (transitions séquentielles).
    """
    episodes = []
    current_episode = []
    
    for idx, entry in enumerate(entries):
        if entry is None or len(entry.obs_seq) == 0:
            continue
            
        obs0 = entry.obs_seq[0]
        if "option_feat" not in obs0 or "option_mask" not in obs0:
            continue
            
        act_vec0 = entry.action_seq[0]
        chosen_indices = np.where(act_vec0 > 0.5)[0]
        chosen_a = int(chosen_indices[0]) if len(chosen_indices) > 0 else int(np.argmax(act_vec0))
        
        option_feat0 = obs0["option_feat"]
        option_mask0 = obs0["option_mask"]
        opt_type = int(np.argmax(option_feat0[chosen_a][:17])) if chosen_a < len(option_feat0) else 0
        
        r0 = float(entry.reward_seq[0]) if len(entry.reward_seq) > 0 else 0.0
        z0 = float(entry.target_val[0]) if len(entry.target_val) > 0 else 0.0
        z1 = float(entry.target_val[1]) if len(entry.target_val) > 1 else z0
        pol0 = float(entry.target_pol[0][chosen_a]) if len(entry.target_pol) > 0 and chosen_a < len(entry.target_pol[0]) else 0.0
        
        active_card_id = int(obs0.get("active_id", 0)) if isinstance(obs0.get("active_id"), (int, np.integer)) else 0
        option_ids = obs0.get("option_ids", None)
        chosen_card_id = int(option_ids[chosen_a]) if option_ids is not None and chosen_a < len(option_ids) else active_card_id

        trans_dict = {
            "entry_idx": idx,
            "entry": entry,
            "obs0": obs0,
            "opt_type": opt_type,
            "chosen_a": chosen_a,
            "chosen_card_id": chosen_card_id,
            "active_card_id": active_card_id,
            "r0": r0,
            "z0": z0,
            "z1": z1,
            "pol0": pol0,
            "sum_unroll_reward": float(np.sum(entry.reward_seq)),
            "is_terminal": abs(r0) > 0.5,
        }

        # Détection de frontière d'épisode
        if not current_episode:
            current_episode.append(trans_dict)
        else:
            prev_trans = current_episode[-1]
            prev_entry = prev_trans["entry"]
            
            # Si l'entrée précédente était terminale OU que l'obs_seq suivante ne correspond pas
            is_new_ep = prev_trans["is_terminal"]
            if not is_new_ep and len(prev_entry.obs_seq) > 1:
                # Vérifier la continuité temporelle via global_feat
                prev_next_gf = prev_entry.obs_seq[1].get("global_feat")
                curr_gf = obs0.get("global_feat")
                if prev_next_gf is not None and curr_gf is not None:
                    # Si le joueur a changé ou si le tour recule brusquement
                    if abs(prev_next_gf[0] - curr_gf[0]) > 0.2 or prev_next_gf[1] != curr_gf[1]:
                        is_new_ep = True
            
            if is_new_ep:
                episodes.append(current_episode)
                current_episode = [trans_dict]
            else:
                current_episode.append(trans_dict)

    if current_episode:
        episodes.append(current_episode)
        
    return episodes


def analyze_buffer(pkl_path: Path, cards_info: dict):
    print(f"\n" + "="*80)
    print(f" REPLAY BUFFER TRANSITIONS DIAGNOSTIC")
    print(f" File: {pkl_path}")
    print(f"="*80)

    print("[*] Deserializing buffer...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    entries = data.get("entries", [])
    total_entries = len(entries)
    buf_size = data.get("size", total_entries)
    step = data.get("step", 0)

    print(f"[*] Total entries in snapshot: {total_entries} (declared size: {buf_size}, step: {step})")

    if total_entries == 0:
        print("[!] Buffer is empty. No transitions to analyze.")
        return

    print("[*] Reconstructing full episode trajectories...")
    episodes = reconstruct_episodes(entries)
    print(f"[*] {len(episodes)} episodes identified (mean length: {total_entries/max(1, len(episodes)):.1f} transitions/game)\n")

    # Conteneurs globaux
    action_type_counts = Counter()
    action_type_rewards = defaultdict(list)
    action_type_values = defaultdict(list)
    
    # ── ATTACK ─────────────────────────────────────────────────────────────
    attack_transitions = []
    attack_rewards = []
    attack_values_z0 = []
    attack_values_z1 = []
    attack_delta_z = []
    attack_mcts_pols = []
    states_with_attack_legal = 0
    actions_chosen_when_attack_legal = Counter()
    values_when_attack_chosen = []
    values_when_attack_skipped = []
    mcts_prob_on_attack_when_skipped = []

    # ── ATTACH VRAIES ÉNERGIES (exclut Outils) ──────────────────────────────
    true_energy_attach_transitions = []
    true_energy_attach_rewards = []
    true_energy_attach_values_z0 = []
    true_energy_attach_values_z1 = []
    true_energy_attach_delta_z = []
    true_energy_attach_mcts_pols = []
    
    # Suivi horizon épisode entier
    true_energy_attack_in_unroll_count = 0
    true_energy_attack_in_episode_count = 0
    true_energy_steps_to_first_attack = []
    true_energy_final_outcome_wins = 0
    true_energy_final_outcome_losses = 0

    # ── ATTACH OUTILS / TOOLS (séparé) ─────────────────────────────────────
    tool_attach_count = 0
    tool_attach_values_z0 = []

    # Opportunités d'attachement
    states_with_true_energy_legal = 0
    actions_chosen_when_true_energy_legal = Counter()
    values_when_true_energy_chosen = []
    values_when_true_energy_skipped = []
    mcts_prob_on_true_energy_when_skipped = []

    for ep_idx, episode in enumerate(episodes):
        ep_len = len(episode)
        ep_terminal_reward = episode[-1]["r0"] if ep_len > 0 else 0.0

        for t_idx, trans in enumerate(episode):
            opt_type = trans["opt_type"]
            opt_name = get_opt_name(opt_type)
            r0 = trans["r0"]
            z0 = trans["z0"]
            z1 = trans["z1"]
            pol0 = trans["pol0"]
            chosen_cid = trans["chosen_card_id"]
            active_cid = trans["active_card_id"]
            obs0 = trans["obs0"]
            option_feat0 = obs0["option_feat"]
            option_mask0 = obs0["option_mask"]

            action_type_counts[opt_name] += 1
            action_type_rewards[opt_name].append(r0)
            action_type_values[opt_name].append(z0)

            # ── 1. ATTACK (type 13) ────────────────────────────────────────
            legal_attack_indices = [
                i for i, m in enumerate(option_mask0)
                if m and i < len(option_feat0) and int(np.argmax(option_feat0[i][:17])) == 13
            ]
            if len(legal_attack_indices) > 0:
                states_with_attack_legal += 1
                actions_chosen_when_attack_legal[opt_name] += 1
                if opt_type == 13:
                    values_when_attack_chosen.append(z0)
                else:
                    values_when_attack_skipped.append(z0)
                    att_prob = float(np.sum([trans["entry"].target_pol[0][ai] for ai in legal_attack_indices if ai < len(trans["entry"].target_pol[0])]))
                    mcts_prob_on_attack_when_skipped.append(att_prob)

            if opt_type == 13:
                attack_rewards.append(r0)
                attack_values_z0.append(z0)
                attack_values_z1.append(z1)
                attack_delta_z.append(z1 - z0)
                attack_mcts_pols.append(pol0)
                card_name = cards_info.get(active_cid, {}).get("name", f"ID_{active_cid}")
                attack_transitions.append({
                    "entry_idx": trans["entry_idx"],
                    "reward": r0,
                    "z0": z0,
                    "z1": z1,
                    "delta_z": z1 - z0,
                    "mcts_prob": pol0,
                    "card_name": card_name,
                })

            # ── 2. ATTACH (type 8) ─────────────────────────────────────────
            legal_attach_indices = [
                i for i, m in enumerate(option_mask0)
                if m and i < len(option_feat0) and int(np.argmax(option_feat0[i][:17])) == 8
            ]
            # Filtrer les options légales d'énergie stricte (hors outils)
            legal_true_energy_indices = []
            option_ids = obs0.get("option_ids", None)
            for i in legal_attach_indices:
                opt_cid = int(option_ids[i]) if option_ids is not None and i < len(option_ids) else 0
                if is_true_energy_card(opt_cid, cards_info):
                    legal_true_energy_indices.append(i)

            if len(legal_true_energy_indices) > 0:
                states_with_true_energy_legal += 1
                actions_chosen_when_true_energy_legal[opt_name] += 1
                if opt_type == 8 and is_true_energy_card(chosen_cid, cards_info):
                    values_when_true_energy_chosen.append(z0)
                else:
                    values_when_true_energy_skipped.append(z0)
                    te_prob = float(np.sum([trans["entry"].target_pol[0][ai] for ai in legal_true_energy_indices if ai < len(trans["entry"].target_pol[0])]))
                    mcts_prob_on_true_energy_when_skipped.append(te_prob)

            if opt_type == 8:
                is_energy = is_true_energy_card(chosen_cid, cards_info)
                is_tool = is_tool_card(chosen_cid, cards_info)

                if is_tool and not is_energy:
                    tool_attach_count += 1
                    tool_attach_values_z0.append(z0)
                else:
                    # VRAIE CARTE ÉNERGIE
                    true_energy_attach_rewards.append(r0)
                    true_energy_attach_values_z0.append(z0)
                    true_energy_attach_values_z1.append(z1)
                    true_energy_attach_delta_z.append(z1 - z0)
                    true_energy_attach_mcts_pols.append(pol0)

                    # a) Attaque dans la fenêtre unroll (court terme)
                    attack_in_unroll = False
                    entry = trans["entry"]
                    for u_step in range(1, len(entry.obs_seq)):
                        if u_step < len(entry.action_seq):
                            u_act = entry.action_seq[u_step]
                            u_a_idx = np.where(u_act > 0.5)[0]
                            u_chosen = int(u_a_idx[0]) if len(u_a_idx) > 0 else int(np.argmax(u_act))
                            u_obs = entry.obs_seq[u_step]
                            if "option_feat" in u_obs and u_chosen < len(u_obs["option_feat"]):
                                if int(np.argmax(u_obs["option_feat"][u_chosen][:17])) == 13:
                                    attack_in_unroll = True
                                    break
                    if attack_in_unroll:
                        true_energy_attack_in_unroll_count += 1

                    # b) Attaque sur TOUT L'ÉPISODE (horizon complet de la partie t+1 -> fin)
                    attack_in_episode = False
                    steps_to_att = None
                    for future_idx in range(t_idx + 1, ep_len):
                        future_trans = episode[future_idx]
                        if future_trans["opt_type"] == 13:
                            attack_in_episode = True
                            steps_to_att = future_idx - t_idx
                            break

                    if attack_in_episode:
                        true_energy_attack_in_episode_count += 1
                        true_energy_steps_to_first_attack.append(steps_to_att)

                    # c) Issue finale de la partie
                    if ep_terminal_reward > 0.5:
                        true_energy_final_outcome_wins += 1
                    elif ep_terminal_reward < -0.5:
                        true_energy_final_outcome_losses += 1

                    card_name = cards_info.get(chosen_cid, {}).get("name", f"ID_{chosen_cid}")
                    target_name = cards_info.get(active_cid, {}).get("name", f"ID_{active_cid}")

                    true_energy_attach_transitions.append({
                        "entry_idx": trans["entry_idx"],
                        "card_name": card_name,
                        "target_name": target_name,
                        "attack_in_unroll": attack_in_unroll,
                        "attack_in_episode": attack_in_episode,
                        "steps_to_att": steps_to_att,
                        "final_win": ep_terminal_reward > 0.5,
                        "z0": z0,
                        "z1": z1,
                        "delta_z": z1 - z0,
                        "mcts_prob": pol0,
                    })

    # =========================================================================
    # 1. VUE D'ENSEMBLE DES ACTIONS
    # =========================================================================
    print("┌" + "─"*78 + "┐")
    print("│ 1. GLOBAL ACTION DISTRIBUTION IN BUFFER                              │")
    print("├" + "─"*78 + "┤")
    print(f"│ {'Action Type':<16} │ {'Count':>8} │ {'Pct (%)':>8} │ {'Mean Reward':>12} │ {'Mean Target Val (z0)':>22} │")
    print("├" + "─"*78 + "┤")
    for act_name, count in action_type_counts.most_common():
        pct = 100.0 * count / max(1, total_entries)
        mean_r = np.mean(action_type_rewards[act_name]) if action_type_rewards[act_name] else 0.0
        mean_z = np.mean(action_type_values_z0[act_name]) if action_type_values_z0[act_name] else 0.0
        print(f"│ {act_name:<16} │ {count:>8d} │ {pct:>7.2f}% │ {mean_r:>12.4f} │ {mean_z:>22.4f} │")
    print("└" + "─"*78 + "┘\n")

    # =========================================================================
    # 2. FOCUS SUR LES TRANSITIONS "ATTACK" (TYPE 13)
    # =========================================================================
    n_attacks = len(attack_rewards)
    print("┌" + "─"*78 + "┐")
    print(f"│ 2. DETAILED ANALYSIS OF 'ATTACK' TRANSITIONS (N = {n_attacks:<6d})                │")
    print("├" + "─"*78 + "┤")

    if n_attacks == 0:
        print("│ [!] NO 'ATTACK' transitions found in this buffer.                   │")
        print("└" + "─"*78 + "┘\n")
    else:
        win_count = sum(1 for r in attack_rewards if r > 0.5)
        loss_count = sum(1 for r in attack_rewards if r < -0.5)
        zero_count = n_attacks - win_count - loss_count
        
        print(f"│ -- Immediate Rewards (r_t during attack):                           │")
        print(f"│    • Immediate Wins (+1.0) : {win_count:>6d} ({100.0 * win_count / n_attacks:5.1f}%)                                  │")
        print(f"│    • Non-terminal / neutral ( 0.0) : {zero_count:>6d} ({100.0 * zero_count / n_attacks:5.1f}%)                                  │")
        print(f"│    • Immediate Losses (-1.0) : {loss_count:>6d} ({100.0 * loss_count / n_attacks:5.1f}%)                                  │")
        print(f"│    • Mean r_t               : {np.mean(attack_rewards):>+7.4f} (min: {np.min(attack_rewards):+0.2f}, max: {np.max(attack_rewards):+0.2f})                 │")
        print("├" + "─"*78 + "┤")

        z0_stats = format_stats(attack_values_z0)
        z1_stats = format_stats(attack_values_z1)
        dz_stats = format_stats(attack_delta_z)
        pol_stats = format_stats(attack_mcts_pols)

        print(f"│ -- Target Value Estimations (Target Return z):                      │")
        print(f"│    • z_0 (Return at root state before attack):                              │")
        print(f"│        Mean: {z0_stats['mean']:>+6.3f} | Std: {z0_stats['std']:>5.3f} | Median: {z0_stats['median']:>+6.3f} | [P25: {z0_stats['p25']:>+6.3f}, P75: {z0_stats['p75']:>+6.3f}] │")
        print(f"│        Min:  {z0_stats['min']:>+6.3f} | Max: {z0_stats['max']:>+6.3f}                                           │")
        print(f"│    • z_1 (Return at next state after attack):                               │")
        print(f"│        Mean: {z1_stats['mean']:>+6.3f} | Std: {z1_stats['std']:>5.3f} | Median: {z1_stats['median']:>+6.3f} | [P25: {z1_stats['p25']:>+6.3f}, P75: {z1_stats['p75']:>+6.3f}] │")
        print(f"│    • Δz = (z_1 - z_0) (Value evolution post-attack):                        │")
        print(f"│        Mean: {dz_stats['mean']:>+6.3f} | Std: {dz_stats['std']:>5.3f} | Median: {dz_stats['median']:>+6.3f}                        │")
        print(f"│    • % attacks leading to positive value (z_0 > 0): {100.0 * sum(1 for z in attack_values_z0 if z > 0) / n_attacks:5.1f}%             │")
        print("├" + "─"*78 + "┤")
        print(f"│ -- MCTS Confidence on Attacks Played (π(a)):                               │")
        print(f"│    • Mean MCTS Proba: {pol_stats['mean']:>6.3f} (Median: {pol_stats['median']:>6.3f}, Min: {pol_stats['min']:>6.3f}, Max: {pol_stats['max']:>6.3f}) │")
        print("├" + "─"*78 + "┤")
        print(f"│ -- Attack Opportunities (States where attack was legal):                 │")
        print(f"│    • Total states with legal attack: {states_with_attack_legal:>6d} ({100.0*states_with_attack_legal/max(1, total_entries):5.1f}% of buffer)                │")
        for act_name, count in actions_chosen_when_attack_legal.most_common():
            pct = 100.0 * count / max(1, states_with_attack_legal)
            print(f"│      - {act_name:<14} : {count:>6d} times ({pct:5.1f}%)                                    │")
        if len(values_when_attack_chosen) > 0 and len(values_when_attack_skipped) > 0:
            print(f"│    • Mean Return z_0 when attack CHOSEN: {np.mean(values_when_attack_chosen):>+6.3f} (Median: {np.median(values_when_attack_chosen):>+6.3f})      │")
            print(f"│    • Mean Return z_0 when attack SKIPPED: {np.mean(values_when_attack_skipped):>+6.3f} (Median: {np.median(values_when_attack_skipped):>+6.3f})      │")
            if len(mcts_prob_on_attack_when_skipped) > 0:
                print(f"│    • MCTS Proba placed on attack when skipped: {np.mean(mcts_prob_on_attack_when_skipped):>6.3f}                   │")
        print("└" + "─"*78 + "┘\n")

    # =========================================================================
    # 3. FOCUS SUR LES TRANSITIONS "ATTACH" (VRAIES ÉNERGIES STRICTES)
    # =========================================================================
    n_true_attaches = len(true_energy_attach_rewards)
    total_raw_attaches = action_type_counts.get("ATTACH", 0)
    
    print("┌" + "─"*78 + "┐")
    print(f"│ 3. ANALYSIS OF TRUE ENERGY ATTACHMENTS (N = {n_true_attaches:<6d})                │")
    print("├" + "─"*78 + "┤")
    print(f"│ • Total raw ATTACH actions in buffer: {total_raw_attaches:>6d}                               │")
    print(f"│   - True Energy cards (Basic / Special) : {n_true_attaches:>6d} ({100.0 * n_true_attaches / max(1, total_raw_attaches):5.1f}%)                            │")
    print(f"│   - Tools / TMs attached (excluded)     : {tool_attach_count:>6d} ({100.0 * tool_attach_count / max(1, total_raw_attaches):5.1f}%)                            │")
    print("├" + "─"*78 + "┤")

    if n_true_attaches == 0:
        print("│ [!] NO true energy attachment found in this buffer.                 │")
        print("└" + "─"*78 + "┘\n")
    else:
        # Taux de conversion en attaque sur tout l'épisode vs unroll
        pct_unroll = 100.0 * true_energy_attack_in_unroll_count / n_true_attaches
        pct_episode = 100.0 * true_energy_attack_in_episode_count / n_true_attaches
        delay_stats = format_stats(true_energy_steps_to_first_attack)

        print(f"│ -- Energy → Attack Synergy (SHORT HORIZON VS ENTIRE EPISODE):              │")
        print(f"│    • Attachments followed by Attack in unroll window (5 steps):             │")
        print(f"│        → {true_energy_attack_in_unroll_count:>6d} / {n_true_attaches:>6d} ({pct_unroll:5.1f}%)                                          │")
        print(f"│    • Attachments followed by Attack over ENTIRE EPISODE (until game end):   │")
        print(f"│        → {true_energy_attack_in_episode_count:>6d} / {n_true_attaches:>6d} ({pct_episode:5.1f}%)                                          │")
        if len(true_energy_steps_to_first_attack) > 0:
            print(f"│    • Mean delay before first attack: {delay_stats['mean']:>5.2f} steps (Median: {delay_stats['median']:>4.1f}, Min: {delay_stats['min']:>2.0f}, Max: {delay_stats['max']:>2.0f})│")
        print("├" + "─"*78 + "┤")

        # Estimations de Valeur Cible
        z0_att = format_stats(true_energy_attach_values_z0)
        z1_att = format_stats(true_energy_attach_values_z1)
        dz_att = format_stats(true_energy_attach_delta_z)
        pol_att = format_stats(true_energy_attach_mcts_pols)

        print(f"│ -- Target Value Estimations (Target Return z):                          │")
        print(f"│    • z_0 (Return before attachment):                                        │")
        print(f"│        Mean: {z0_att['mean']:>+6.3f} | Std: {z0_att['std']:>5.3f} | Median: {z0_att['median']:>+6.3f} | [P25: {z0_att['p25']:>+6.3f}, P75: {z0_att['p75']:>+6.3f}] │")
        print(f"│    • z_1 (Return after attachment):                                         │")
        print(f"│        Mean: {z1_att['mean']:>+6.3f} | Std: {z1_att['std']:>5.3f} | Median: {z1_att['median']:>+6.3f} | [P25: {z1_att['p25']:>+6.3f}, P75: {z1_att['p75']:>+6.3f}] │")
        print(f"│    • Δz = (z_1 - z_0) (Value evolution post-attachment):                   │")
        print(f"│        Mean: {dz_att['mean']:>+6.3f} | Std: {dz_att['std']:>5.3f} | Median: {dz_att['median']:>+6.3f}                        │")
        print(f"│    • % attachments with z_0 > 0: {100.0 * sum(1 for z in true_energy_attach_values_z0 if z > 0) / n_true_attaches:5.1f}%                                     │")
        print(f"│    • Final outcome of games with energy attachment:                         │")
        print(f"│        Wins: {true_energy_final_outcome_wins:>5d} ({100.0*true_energy_final_outcome_wins/n_true_attaches:5.1f}%) | Losses: {true_energy_final_outcome_losses:>5d} ({100.0*true_energy_final_outcome_losses/n_true_attaches:5.1f}%)                 │")
        print("├" + "─"*78 + "┤")
        print(f"│ -- MCTS Confidence on Attached Energies (π(a)):                           │")
        print(f"│    • Mean MCTS Proba: {pol_att['mean']:>6.3f} (Median: {pol_att['median']:>6.3f}, Min: {pol_att['min']:>6.3f}, Max: {pol_att['max']:>6.3f}) │")
        print("├" + "─"*78 + "┤")
        print(f"│ -- True Energy Attachment Opportunities:                                   │")
        print(f"│    • Total states with legal energy: {states_with_true_energy_legal:>6d} ({100.0*states_with_true_energy_legal/max(1, total_entries):5.1f}% of buffer)             │")
        for act_name, count in actions_chosen_when_true_energy_legal.most_common():
            pct = 100.0 * count / max(1, states_with_true_energy_legal)
            print(f"│      - {act_name:<14} : {count:>6d} times ({pct:5.1f}%)                                    │")
        if len(values_when_true_energy_chosen) > 0 and len(values_when_true_energy_skipped) > 0:
            print(f"│    • Mean Return z_0 when energy ATTACHED: {np.mean(values_when_true_energy_chosen):>+6.3f} (Median: {np.median(values_when_true_energy_chosen):>+6.3f})    │")
            print(f"│    • Mean Return z_0 when energy SKIPPED : {np.mean(values_when_true_energy_skipped):>+6.3f} (Median: {np.median(values_when_true_energy_skipped):>+6.3f})    │")
            if len(mcts_prob_on_true_energy_when_skipped) > 0:
                print(f"│    • MCTS Proba placed on energy when skipped: {np.mean(mcts_prob_on_true_energy_when_skipped):>6.3f}                   │")
        print("└" + "─"*78 + "┘\n")

    # =========================================================================
    # 4. ÉCHANTILLONS D'ATTAQUES ET D'ATTACHEMENTS D'ÉNERGIE
    # =========================================================================
    if attack_transitions:
        print("┌" + "─"*78 + "┐")
        print("│ 4. RECORDED 'ATTACK' TRANSITION SAMPLES                                     │")
        print("├" + "─"*78 + "┤")
        print(f"│ {'Idx':>6} │ {'Card / Pokémon':<20} │ {'Reward r0':>9} │ {'z0 (Val)':>8} │ {'z1 (Next)':>9} │ {'π(MCTS)':>7} │")
        print("├" + "─"*78 + "┤")
        sample_indices = np.linspace(0, len(attack_transitions) - 1, min(10, len(attack_transitions)), dtype=int)
        for s_i in sample_indices:
            t = attack_transitions[s_i]
            c_name = str(t['card_name'])[:20]
            print(f"│ {t['entry_idx']:>6d} │ {c_name:<20} │ {t['reward']:>+9.2f} │ {t['z0']:>+8.3f} │ {t['z1']:>+9.3f} │ {t['mcts_prob']:>7.3f} │")
        print("└" + "─"*78 + "┘\n")

    if true_energy_attach_transitions:
        print("┌" + "─"*78 + "┐")
        print("│ 5. ENERGY ATTACHMENT SAMPLES (FULL EPISODE TRACKING)                        │")
        print("├" + "─"*78 + "┤")
        print(f"│ {'Idx':>6} │ {'Energy':<16} │ {'Episode Attack':<18} │ {'z0 (Val)':>8} │ {'z1 (Next)':>9} │ {'π(MCTS)':>7} │")
        print("├" + "─"*78 + "┤")
        sample_indices = np.linspace(0, len(true_energy_attach_transitions) - 1, min(10, len(true_energy_attach_transitions)), dtype=int)
        for s_i in sample_indices:
            t = true_energy_attach_transitions[s_i]
            e_name = str(t['card_name'])[:16]
            if t['attack_in_episode']:
                att_status = f"YES (+{t['steps_to_att']} steps)"
            else:
                att_status = "NO (0 att.)"
            print(f"│ {t['entry_idx']:>6d} │ {e_name:<16} │ {att_status:<18} │ {t['z0']:>+8.3f} │ {t['z1']:>+9.3f} │ {t['mcts_prob']:>7.3f} │")
        print("└" + "─"*78 + "┘\n")

    print("[✓] Full diagnostic completed.")


def main():
    parser = argparse.ArgumentParser(description="Analyze Attack and Attach transitions in the MuZero Replay Buffer")
    parser.add_argument("--repo-id", type=str, default="richard151111/muzero-V2", help="HuggingFace repository (e.g. richard151111/muzero-V2)")
    parser.add_argument("--local-path", type=str, default="", help="Path to a local replay_buffer.pkl (if already downloaded)")
    parser.add_argument("--token", type=str, default="", help="HuggingFace token (or via export HF_TOKEN=...)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or None

    default_downloaded = Path("./hf_download/replay_buffer.pkl")
    abs_downloaded = Path("/home/richard/Downloads/files/hf_download/replay_buffer.pkl")

    if args.local_path and Path(args.local_path).exists():
        pkl_path = Path(args.local_path)
    elif abs_downloaded.exists():
        pkl_path = abs_downloaded
    elif default_downloaded.exists():
        pkl_path = default_downloaded
    else:
        pkl_path = download_buffer_from_hf(repo_id=args.repo_id, token=token)

    cards_info = load_card_database()
    analyze_buffer(pkl_path, cards_info)


if __name__ == "__main__":
    main()
