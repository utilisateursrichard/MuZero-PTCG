#!/usr/bin/env python3
"""
battle_server.py
================
Web server for the Pokémon TCG MuZero Interactive Battle Arena.
Provides a REST API to control game sessions and serves static UI assets.
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# Ensure ptcg_muzero is in sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
MUZERO_DIR = SCRIPT_DIR / "ptcg_muzero"
STATIC_DIR = SCRIPT_DIR / "static"
if MUZERO_DIR.exists() and str(MUZERO_DIR) not in sys.path:
    sys.path.insert(0, str(MUZERO_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("battle_server")

from cards.deck_manager import CardDatabase, DeckManager
from env.battle_session import BattleSession

# Global active session
_active_session: Optional[BattleSession] = None
_deck_manager: Optional[DeckManager] = None


def get_deck_manager() -> DeckManager:
    global _deck_manager
    if _deck_manager is None:
        _deck_manager = DeckManager()
    return _deck_manager


class BattleAPIHandler(SimpleHTTPRequestHandler):
    """Handles HTTP REST API endpoints and serves static UI files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, data: Any, status_code: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json({"status": "ok", "service": "PTCG MuZero Battle Arena"})
            return

        if path == "/api/decks":
            dm = get_deck_manager()
            presets = dm.get_preset_decks()
            self._send_json({"status": "success", "decks": presets})
            return

        if path == "/api/battle/state":
            global _active_session
            if _active_session is None:
                self._send_json({"status": "error", "message": "Aucune partie en cours."}, 404)
            else:
                self._send_json({"status": "success", "state": _active_session.get_state()})
            return

        # Serve static files fallback (or index.html)
        if path == "/" or not (STATIC_DIR / path.lstrip("/")).exists():
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self) -> None:
        global _active_session
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            req_data = json.loads(body) if body else {}
        except Exception:
            req_data = {}

        if path == "/api/generate_deck":
            dm = get_deck_manager()
            try:
                deck = dm.generate_ai_deck()
                summary = dm.get_deck_summary(deck)
                self._send_json({
                    "status": "success",
                    "deck": deck,
                    "summary": summary,
                    "name": "✨ AI-Generated Deck",
                })
            except Exception as e:
                logger.exception("Error generating deck: %s", e)
                # Fallback to model deck
                m_deck = dm.get_model_deck()
                self._send_json({
                    "status": "success",
                    "deck": m_deck,
                    "summary": dm.get_deck_summary(m_deck),
                    "name": "⚡ MuZero AI Deck (Fallback)",
                })
            return

        if path == "/api/battle/start":
            dm = get_deck_manager()
            player_deck = req_data.get("player_deck") or dm.get_model_deck()
            ai_deck = req_data.get("ai_deck") or dm.get_model_deck()
            device_uri = req_data.get("device", "vulkan")
            ai_mode = req_data.get("ai_mode", "basic")

            # Validate decks
            ok_p, msg_p = dm.validate_deck(player_deck)
            if not ok_p:
                self._send_json({"status": "error", "message": f"Invalid Player deck: {msg_p}"}, 400)
                return

            ok_ai, msg_ai = dm.validate_deck(ai_deck)
            if not ok_ai:
                self._send_json({"status": "error", "message": f"Invalid AI deck: {msg_ai}"}, 400)
                return

            try:
                if _active_session is not None:
                    _active_session.env.close()

                _active_session = BattleSession(
                    player_deck=player_deck,
                    ai_deck=ai_deck,
                    device_uri=device_uri,
                    ai_mode=ai_mode,
                )
                state = _active_session.start()
                self._send_json({"status": "success", "state": state})

            except Exception as e:
                logger.exception("Failed to start battle: %s", e)
                self._send_json({"status": "error", "message": f"Start error: {e}"}, 500)
            return

        if path == "/api/battle/step":
            if _active_session is None:
                self._send_json({"status": "error", "message": "No active game."}, 404)
                return

            selected_indices = req_data.get("selected_indices", [])
            try:
                state = _active_session.submit_action(selected_indices)
                self._send_json({"status": "success", "state": state})
            except Exception as e:
                logger.exception("Error during battle step: %s", e)
                self._send_json({"status": "error", "message": f"Error during action: {e}"}, 500)
            return

        if path == "/api/battle/reset":
            if _active_session is not None:
                _active_session.env.close()
                _active_session = None
            self._send_json({"status": "success", "message": "Game reset."})
            return

        self._send_json({"status": "error", "message": "Unknown endpoint."}, 404)


def main():
    parser = argparse.ArgumentParser(description="Interactive Web Battle Server against MuZero model (IREE)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server listening port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Listening host (default: 0.0.0.0)")
    args = parser.parse_args()

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, BattleAPIHandler)

    logger.info("==============================================================")
    logger.info("🔥 PTCG MuZero Interactive Battle Arena started successfully!")
    logger.info("👉 Open your browser at: http://localhost:%d", args.port)
    logger.info("==============================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nStopping server...")
        if _active_session:
            _active_session.env.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
