"""
ptcg_muzero/env/battle_session.py
==================================
Interactive match session manager between a human player and the IREE MuZero agent.
Handles turn stepping, option resolution, action translation, and state serialization for the UI.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cards.deck_manager import CardDatabase, DeckManager
from config import Config
from env.cabt_api import AreaType, OptionType, SelectContext, SelectType
from env.wrapper import CabtEnv, DeckError
from models.iree_agent import IREEMuZeroAgent

logger = logging.getLogger("battle_session")


class BattleSession:
    """Manages an active game instance between Player 0 (Human) and Player 1 (MuZero AI)."""

    def __init__(
        self,
        player_deck: Optional[List[int]] = None,
        ai_deck: Optional[List[int]] = None,
        vmfb_path: Optional[str] = None,
        device_uri: str = "vulkan",
        ai_mode: str = "advanced",
    ):
        from pathlib import Path
        from models.iree_agent import create_agent
        self.deck_mgr = DeckManager()
        self.card_db = CardDatabase.get()
        self.cfg = Config()

        self.player_deck = player_deck or self.deck_mgr.get_model_deck()
        self.ai_deck = ai_deck or self.deck_mgr.get_model_deck()
        self.ai_mode = ai_mode

        if vmfb_path is None:
            if device_uri == "cpu":
                candidates = ["muzero_cpu.vmfb", "muzero_CPU.vmfb", "muzero_vulkan.vmfb"]
                vmfb_path = next((c for c in candidates if Path(c).exists()), "muzero_cpu.vmfb")
            else:
                candidates = ["muzero_vulkan.vmfb", "muzero_cpu.vmfb", "muzero_CPU.vmfb"]
                vmfb_path = next((c for c in candidates if Path(c).exists()), "muzero_vulkan.vmfb")

        self.env = CabtEnv()
        self.ai_agent = create_agent(
            mode=ai_mode,
            vmfb_path=vmfb_path,
            device_uri=device_uri,
            cfg=self.cfg,
        )
        if hasattr(self.ai_agent, "set_opponent_deck"):
            self.ai_agent.set_opponent_deck(self.player_deck)

        self.is_started = False
        self.is_done = False
        self.result: int = -1  # -1 in progress, 0 human win, 1 ai win, 2 draw
        self.logs: List[Dict[str, Any]] = []
        self.ai_thoughts: Dict[str, Any] = {"winrate": 50.0, "value": 0.0, "top_actions": []}
        self.current_obs: Optional[dict] = None
        self.step_counter = 0


    def start(self) -> Dict[str, Any]:
        """Starts the match and advances until the first human choice is required."""
        self.env.close()
        self.logs.clear()
        self.step_counter = 0
        self.is_done = False
        self.result = -1

        self._add_log("game", "🎮 Pokémon TCG battle vs MuZero started!", level="info")
        self._add_log("game", f"Player Deck: {len(self.player_deck)} cards | AI Deck: {len(self.ai_deck)} cards", level="info")

        obs_dict, done = self.env.reset(self.player_deck, self.ai_deck)
        self.current_obs = obs_dict
        self.is_started = True
        self.is_done = done

        if not done:
            self._advance_until_human()

        return self.get_state()

    def submit_action(self, selected_indices: List[int]) -> Dict[str, Any]:
        """Submits the human player's choice, then runs AI turns until the next human decision."""
        if not self.is_started or self.is_done:
            return self.get_state()

        select = self.current_obs.get("select") if self.current_obs else None
        if select is None:
            # Setup phase: submit deck
            self.current_obs, self.is_done = self.env.step(self.player_deck)
        else:
            options = select.get("option", [])
            chosen_labels = []
            for idx in selected_indices:
                if 0 <= idx < len(options):
                    opt = options[idx]
                    label = self._describe_option(opt, select)
                    chosen_labels.append(label["title"])
            
            if chosen_labels:
                self._add_log("player", f"👤 You chose: {', '.join(chosen_labels)}", level="action")

            self.current_obs, self.is_done = self.env.step([int(x) for x in selected_indices])

        if not self.is_done:
            self._advance_until_human()
        else:
            self._handle_game_over()

        return self.get_state()

    def _advance_until_human(self) -> None:
        """Executes AI turns automatically until it is human's turn or game terminates."""
        max_loop = 150
        loop_cnt = 0

        while not self.is_done and loop_cnt < max_loop:
            loop_cnt += 1
            self.step_counter += 1
            cur = self.current_obs.get("current", {})
            your_idx = cur.get("yourIndex", 0)
            select = self.current_obs.get("select")

            # Check if game is done according to current result
            res = cur.get("result", -1)
            if res >= 0:
                self.is_done = True
                self.result = res
                self._handle_game_over()
                break

            # If select is None (Deck submission)
            if select is None:
                deck = self.player_deck if your_idx == 0 else self.ai_deck
                self.current_obs, self.is_done = self.env.step(deck)
                continue

            options = select.get("option", [])
            if not options:
                self.current_obs, self.is_done = self.env.step([])
                continue

            if your_idx == 0:
                # Human player's turn to decide! Break loop and wait for input
                break
            else:
                # AI player's turn
                ai_choice, metadata = self.ai_agent.choose_action(self.current_obs, your_idx)
                self.ai_thoughts = metadata

                # Log AI decision
                if ai_choice:
                    ai_labels = []
                    for idx in ai_choice:
                        if 0 <= idx < len(options):
                            label = self._describe_option(options[idx], select)
                            ai_labels.append(label["title"])
                    if ai_labels:
                        self._add_log(
                            "ai",
                            f"🤖 MuZero ({metadata.get('winrate', 50)}% winrate): {', '.join(ai_labels)}",
                            level="ai",
                        )

                self.current_obs, self.is_done = self.env.step(ai_choice)

        if self.is_done:
            self._handle_game_over()

    def _handle_game_over(self) -> None:
        self.result = self.env.result
        
        # Determine exact game over reason from board state
        cur = self.current_obs.get("current", {}) if self.current_obs else {}
        players = cur.get("players", [{}, {}])
        p0 = players[0] if len(players) > 0 else {}
        p1 = players[1] if len(players) > 1 else {}
        
        p0_prizes = len(p0.get("prize") or [])
        p1_prizes = len(p1.get("prize") or [])
        p0_deck = p0.get("deckCount", len(p0.get("deck") or []))
        p1_deck = p1.get("deckCount", len(p1.get("deck") or []))
        p0_mons = len(p0.get("active") or []) + len(p0.get("bench") or [])
        p1_mons = len(p1.get("active") or []) + len(p1.get("bench") or [])
        
        if self.result == 0:
            if p0_prizes == 0:
                self.game_over_reason = "You took all 6 Prize cards!"
            elif p1_mons == 0 and cur.get("turn", 0) > 0:
                self.game_over_reason = "Opponent has no Pokémon left in play (total Knockout)."
            elif p1_deck == 0:
                self.game_over_reason = "Opponent has no cards left in deck (Deck Out)."
            else:
                self.game_over_reason = "Victory by game condition."
            self._add_log("game", f"🏆 VICTORY! {self.game_over_reason}", level="victory")
        elif self.result == 1:
            if p1_prizes == 0:
                self.game_over_reason = "MuZero AI took all 6 Prize cards."
            elif p0_mons == 0 and cur.get("turn", 0) > 0:
                self.game_over_reason = "You have no Pokémon left in play on your field."
            elif p0_deck == 0:
                self.game_over_reason = "You have no cards left in your deck (Deck Out)."
            else:
                self.game_over_reason = "MuZero AI won the match."
            self._add_log("game", f"💀 DEFEAT! {self.game_over_reason}", level="defeat")
        else:
            self.game_over_reason = "Match concluded in a draw."
            self._add_log("game", "🤝 DRAW MATCH!", level="draw")

    def _add_log(self, sender: str, message: str, level: str = "info") -> None:
        self.logs.append({
            "sender": sender,
            "message": message,
            "level": level,
            "turn": self.current_obs.get("current", {}).get("turn", 0) if self.current_obs else 0,
        })


    def _describe_option(self, opt: dict, select: dict) -> Dict[str, Any]:
        """Translates a low-level engine option dict into rich, human-readable details."""
        opt_type = opt.get("type", 0)
        context = select.get("context", 0)
        
        # Default description
        title = f"Action #{opt_type}"
        subtitle = ""
        badge = "Action"
        badge_color = "#64748b"
        card_id = None
        target_info = None

        if opt_type == 1: # YES
            title = "Yes / Confirm"
            badge = "Confirm"
            badge_color = "#10b981"
        elif opt_type == 2: # NO
            title = "No / Pass"
            badge = "Cancel"
            badge_color = "#ef4444"
        elif opt_type == 0: # NUMBER
            num = opt.get("number", opt.get("count", 0))
            title = f"Choose number: {num}"
            badge = "Number"
        elif opt_type == 3: # CARD
            area = opt.get("area", 0)
            idx = opt.get("index", 0)
            p_idx = opt.get("playerIndex", 0)
            
            # Resolve actual card from appropriate area
            cur = self.current_obs.get("current", {}) if self.current_obs else {}
            players = cur.get("players", [{}, {}])
            p_state = players[p_idx] if p_idx < len(players) else {}
            
            if area == 1: # DECK
                deck_cards = select.get("deck") or p_state.get("deck") or []
                if 0 <= idx < len(deck_cards) and deck_cards[idx]:
                    card_id = deck_cards[idx].get("id")
            elif area == 2: # HAND
                hand = p_state.get("hand") or []
                if 0 <= idx < len(hand) and hand[idx]:
                    card_id = hand[idx].get("id")
            elif area == 4: # ACTIVE
                act = p_state.get("active") or []
                if act:
                    card_id = act[0].get("id")
            elif area == 5: # BENCH
                bench = p_state.get("bench") or []
                if 0 <= idx < len(bench) and bench[idx]:
                    card_id = bench[idx].get("id")
            elif area == 3: # DISCARD
                disc = p_state.get("discard") or []
                if 0 <= idx < len(disc) and disc[idx]:
                    card_id = disc[idx].get("id")
            elif area == 6: # PRIZE
                prize = p_state.get("prize") or []
                if 0 <= idx < len(prize) and prize[idx]:
                    card_id = prize[idx].get("id")
            elif area == 12: # LOOKING
                look = cur.get("looking") or select.get("looking") or []
                if 0 <= idx < len(look) and look[idx]:
                    card_id = look[idx].get("id")

            area_names = {1: "Deck", 2: "Hand", 3: "Discard", 4: "Active", 5: "Bench", 6: "Prize", 12: "Revealed Cards"}
            area_str = area_names.get(area, f"Zone {area}")

            if card_id:
                card_data = self.card_db.get_card(card_id)
                title = f"{card_data.get('name', 'Card')} ({area_str})"
                subtitle = f"{card_data.get('stage', '')} | HP: {card_data.get('hp', 0)}"
                badge = area_str
                badge_color = card_data.get("type_color", "#38bdf8")
            else:
                title = f"Select card #{idx + 1} ({area_str})"
                badge = area_str

        elif opt_type == 7: # PLAY
            hand_idx = opt.get("index", 0)
            cur = self.current_obs.get("current", {}) if self.current_obs else {}
            players = cur.get("players", [{}, {}])
            my_hand = players[0].get("hand") or [] if players else []
            if 0 <= hand_idx < len(my_hand) and my_hand[hand_idx]:
                card_id = my_hand[hand_idx].get("id")
                card_data = self.card_db.get_card(card_id)
                title = f"Play {card_data.get('name')}"
                subtitle = card_data.get("description", "")
                badge = "Play"
                badge_color = card_data.get("type_color", "#3b82f6")
            else:
                title = f"Play card from hand #{hand_idx}"

        elif opt_type == 8: # ATTACH
            hand_idx = opt.get("index", 0)
            in_play_idx = opt.get("inPlayIndex", 0)
            cur = self.current_obs.get("current", {}) if self.current_obs else {}
            players = cur.get("players", [{}, {}])
            my_hand = players[0].get("hand") or [] if players else []
            if 0 <= hand_idx < len(my_hand) and my_hand[hand_idx]:
                card_id = my_hand[hand_idx].get("id")
                card_data = self.card_db.get_card(card_id)
                target_str = "Active" if in_play_idx == 0 else f"Bench #{in_play_idx}"
                title = f"Attach {card_data.get('name')} to {target_str}"
                badge = "Attach"
                badge_color = "#f59e0b"
            else:
                title = f"Attach card to {in_play_idx}"

        elif opt_type == 9: # EVOLVE
            title = "Evolve a Pokémon"
            badge = "Evolution"
            badge_color = "#8b5cf6"

        elif opt_type == 10: # ABILITY
            title = "Activate Ability"
            badge = "Ability"
            badge_color = "#ec4899"

        elif opt_type == 12: # RETREAT
            title = "Retreat to Bench"
            badge = "Retreat"
            badge_color = "#64748b"

        elif opt_type == 13: # ATTACK
            atk_idx = opt.get("index", 0)
            cur = self.current_obs.get("current", {}) if self.current_obs else {}
            players = cur.get("players", [{}, {}])
            my_act = players[0].get("active") or [] if players else []
            if my_act:
                card_id = my_act[0].get("id")
                card_data = self.card_db.get_card(card_id)
                attacks = card_data.get("attacks", [])
                if 0 <= atk_idx < len(attacks):
                    atk = attacks[atk_idx]
                    dmg_str = f" ({atk.get('damage')} damage)" if atk.get('damage') and atk.get('damage') != '0' else ""
                    title = f"⚔️ Attack: {atk.get('name')}{dmg_str}"
                    subtitle = atk.get("effect", "")
                    badge = "Attack"
                    badge_color = "#ef4444"
                else:
                    title = f"⚔️ Attack (Attack #{atk_idx})"
            else:
                title = f"⚔️ Attack (Attack #{atk_idx})"

        elif opt_type == 14: # END_TURN
            title = "🛑 End Turn"
            badge = "End Turn"
            badge_color = "#475569"

        return {
            "title": title,
            "subtitle": subtitle,
            "badge": badge,
            "badge_color": badge_color,
            "card_id": card_id,
            "raw": opt,
        }

    def _format_pokemon(self, p_dict: Optional[dict]) -> Optional[Dict[str, Any]]:
        if not p_dict:
            return None
        cid = p_dict.get("id", 0)
        card_data = self.card_db.get_card(cid)
        hp = p_dict.get("hp", card_data.get("hp", 0))
        max_hp = p_dict.get("maxHp", card_data.get("hp", 100))
        
        raw_energies = p_dict.get("energies", [])
        energy_ints = []
        energy_icons = []
        for e in raw_energies:
            e_val = e if isinstance(e, int) else (e.get("value", -1) if isinstance(e, dict) else -1)
            type_map = {0: "{C}", 1: "{G}", 2: "{R}", 3: "{W}", 4: "{L}", 5: "{P}", 6: "{F}", 7: "{D}", 8: "{M}", 9: "{N}", 10: "{Y}"}
            if 0 <= e_val < 11:
                energy_ints.append(e_val)
                if e_val in type_map:
                    energy_icons.append(type_map[e_val])

        energy_cards = p_dict.get("energyCards") or []
        tools = p_dict.get("tools") or []
        pre_evolution = p_dict.get("preEvolution") or []

        return {
            "id": cid,
            "name": card_data.get("name", "Pokémon"),
            "stage": card_data.get("stage", "Basic"),
            "type": card_data.get("type", "{C}"),
            "type_color": card_data.get("type_color", "#64748b"),
            "hp": hp,
            "maxHp": max_hp,
            "max_hp": max_hp,
            "hp_percent": round(max(0.0, min(1.0, float(hp) / max(1.0, float(max_hp)))) * 100.0, 1),
            "retreat": card_data.get("retreat", 1),
            "energies": energy_ints,
            "energy_icons": energy_icons,
            "energyCards": energy_cards,
            "tools": tools,
            "preEvolution": pre_evolution,
            "attacks": card_data.get("attacks", []),
            "weakness": card_data.get("weakness", ""),
            "resistance": card_data.get("resistance", ""),
            "description": card_data.get("description", ""),
        }

    def get_state(self) -> Dict[str, Any]:
        """Serializes current board state, players, options, and logs for the Web UI."""
        if not self.current_obs:
            return {
                "is_started": False,
                "is_done": False,
                "result": -1,
                "turn": 0,
                "your_index": 0,
                "player": None,
                "opponent": None,
                "options": [],
                "logs": self.logs,
                "ai_thoughts": self.ai_thoughts,
                "current": None,
                "select": None,
            }

        cur = self.current_obs.get("current", {})
        players = cur.get("players", [{}, {}])
        p0_state = players[0] if len(players) > 0 else {}
        p1_state = players[1] if len(players) > 1 else {}

        # Stadium
        stadium_list = cur.get("stadium") or []
        stadium_card = None
        if stadium_list:
            stadium_id = stadium_list[0].get("id") if isinstance(stadium_list[0], dict) else stadium_list[0]
            if stadium_id:
                stadium_card = self.card_db.get_card(int(stadium_id))

        # Player 0 (Human) formatting
        p0_active = self._format_pokemon(p0_state.get("active", [None])[0] if p0_state.get("active") else None)
        p0_bench = [self._format_pokemon(p) for p in (p0_state.get("bench") or []) if p is not None]
        p0_hand = []
        for card in (p0_state.get("hand") or []):
            if card:
                cid = card.get("id", 0)
                cdata = dict(self.card_db.get_card(cid))
                p0_hand.append(cdata)

        p0_discard = [self.card_db.get_card(c.get("id")) for c in (p0_state.get("discard") or []) if c]
        p0_prizes_left = len(p0_state.get("prize") or [])

        # Player 1 (AI) formatting
        p1_active = self._format_pokemon(p1_state.get("active", [None])[0] if p1_state.get("active") else None)
        p1_bench = [self._format_pokemon(p) for p in (p1_state.get("bench") or []) if p is not None]
        p1_hand_cards = []
        for card in (p1_state.get("hand") or []):
            if card:
                cid = card.get("id", 0)
                cdata = dict(self.card_db.get_card(cid))
                p1_hand_cards.append(cdata)
        p1_hand_count = p1_state.get("handCount", len(p1_hand_cards))
        p1_discard = [self.card_db.get_card(c.get("id")) for c in (p1_state.get("discard") or []) if c]
        p1_prizes_left = len(p1_state.get("prize") or [])


        # Options formatting for human
        select = self.current_obs.get("select") or {}
        raw_options = select.get("option", [])
        min_cnt = int(select.get("minCount", 1))
        max_cnt = int(select.get("maxCount", 1))
        context_id = select.get("context", 0)

        options_list = []
        for i, opt in enumerate(raw_options):
            desc = self._describe_option(opt, select)
            desc["index"] = i
            options_list.append(desc)

        return {
            "is_started": self.is_started,
            "is_done": self.is_done,
            "result": self.result,
            "turn": cur.get("turn", 0),
            "your_index": cur.get("yourIndex", 0),
            "is_human_turn": cur.get("yourIndex", 0) == 0 and not self.is_done,
            "player": {
                "active": p0_active,
                "bench": p0_bench,
                "hand": p0_hand,
                "discard": p0_discard,
                "deck_count": p0_state.get("deckCount", len(self.player_deck)),
                "prizes_left": p0_prizes_left,
                "poisoned": p0_state.get("poisoned", False),
                "burned": p0_state.get("burned", False),
                "asleep": p0_state.get("asleep", False),
                "paralyzed": p0_state.get("paralyzed", False),
                "confused": p0_state.get("confused", False),
            },
            "opponent": {
                "active": p1_active,
                "bench": p1_bench,
                "hand_count": p1_hand_count,
                "discard": p1_discard,
                "deck_count": p1_state.get("deckCount", len(self.ai_deck)),
                "prizes_left": p1_prizes_left,
                "poisoned": p1_state.get("poisoned", False),
                "burned": p1_state.get("burned", False),
                "asleep": p1_state.get("asleep", False),
                "paralyzed": p1_state.get("paralyzed", False),
                "confused": p1_state.get("confused", False),
            },
            "select_context": {
                "id": context_id,
                "type": select.get("type", 0),
                "min_count": min_cnt,
                "max_count": max_cnt,
                "prompt": self._get_context_prompt(context_id),
            },
            "options": options_list,
            "logs": self.logs[-40:],
            "ai_thoughts": self.ai_thoughts,
            "game_over_reason": getattr(self, "game_over_reason", ""),
            "ai_mode": getattr(self, "ai_mode", "basic"),
        }



    def _get_context_prompt(self, context_id: int) -> str:
        prompts = {
            0: "Main phase: choose an action or end your turn.",
            1: "Initial setup: choose your Active Pokémon.",
            2: "Initial setup: choose Pokémon for your Bench.",
            3: "Switch Pokémon: choose a Pokémon to put in Active Spot.",
            4: "Put a Pokémon into Active Spot.",
            5: "Put a Pokémon onto the Bench.",
            8: "Select target card or Pokémon.",
            13: "Choose an Attack to execute.",
            41: "Coin toss: Do you want to go first?",
            42: "Mulligan: Do you want to redraw your hand?",
        }
        return prompts.get(context_id, "Make your choice from the options below:")
