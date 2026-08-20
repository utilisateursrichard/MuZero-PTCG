#!/usr/bin/env python3
"""
battle_server.py
================
Multi-user Web server for the Pokémon TCG MuZero Interactive Battle Arena.
Provides a REST API to control isolated game sessions per user and serves static UI assets.
"""
from __future__ import annotations

import argparse
import http.cookies
import json
import logging
import mimetypes
import os
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

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

import subprocess


class SessionEntry:
    """Encapsulates a user's BattleSession with a per-session lock and timestamp."""

    def __init__(self, session_id: str, session: BattleSession):
        self.session_id = session_id
        self.session = session
        self.lock = threading.Lock()
        self.created_at = time.time()
        self.last_accessed = time.time()

    def touch(self) -> None:
        self.last_accessed = time.time()

    def close(self) -> None:
        try:
            self.session.env.close()
        except Exception:
            pass


class SessionManager:
    """Thread-safe manager for multiple concurrent battle sessions with TTL cleanup."""

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        session_id: Optional[str],
        player_deck: List[int],
        ai_deck: List[int],
        device_uri: str = "vulkan",
        ai_mode: str = "basic",
        vmfb_path: Optional[str] = None,
    ) -> SessionEntry:
        if not session_id:
            session_id = str(uuid.uuid4())

        with self._lock:
            # If an existing session with this ID is present, close it first
            if session_id in self._sessions:
                old = self._sessions.pop(session_id)
                old.close()

            new_session = BattleSession(
                player_deck=player_deck,
                ai_deck=ai_deck,
                vmfb_path=vmfb_path,
                device_uri=device_uri,
                ai_mode=ai_mode,
            )
            entry = SessionEntry(session_id=session_id, session=new_session)
            self._sessions[session_id] = entry
            logger.info("✨ Created new battle session [%s] (Active sessions: %d)", session_id, len(self._sessions))
            return entry

    def get_session(self, session_id: str) -> Optional[SessionEntry]:
        if not session_id:
            return None
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry:
                entry.touch()
            return entry

    def remove_session(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
            if entry:
                entry.close()
                logger.info("🗑️ Removed session [%s] (Remaining: %d)", session_id, len(self._sessions))
                return True
            return False

    def cleanup_expired(self) -> int:
        now = time.time()
        expired: List[SessionEntry] = []
        with self._lock:
            for sid, entry in list(self._sessions.items()):
                if now - entry.last_accessed > self.ttl_seconds:
                    expired.append(self._sessions.pop(sid))

        for entry in expired:
            entry.close()
            logger.info("⏱️ Expired inactive session [%s]", entry.session_id)
        return len(expired)

    def close_all(self) -> None:
        with self._lock:
            for entry in self._sessions.values():
                entry.close()
            self._sessions.clear()

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Global multi-user session manager
_session_manager = SessionManager(ttl_seconds=1800)
_deck_manager: Optional[DeckManager] = None


def _get_python_executable() -> str:
    venv_py = SCRIPT_DIR / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    venv_gpu_py = SCRIPT_DIR / ".venv_gpu" / "bin" / "python"
    if venv_gpu_py.exists():
        return str(venv_gpu_py)
    return sys.executable


def _get_models_status() -> Dict[str, Any]:
    vulkan_paths = [SCRIPT_DIR / "muzero_vulkan.vmfb", Path("muzero_vulkan.vmfb")]
    cpu_paths = [SCRIPT_DIR / "muzero_cpu.vmfb", SCRIPT_DIR / "muzero_CPU.vmfb", Path("muzero_cpu.vmfb"), Path("muzero_CPU.vmfb")]

    vulkan_found = next((p for p in vulkan_paths if p.exists()), None)
    cpu_found = next((p for p in cpu_paths if p.exists()), None)

    return {
        "vulkan": {
            "exists": vulkan_found is not None,
            "filename": vulkan_found.name if vulkan_found else "muzero_vulkan.vmfb",
            "size_kb": (vulkan_found.stat().st_size // 1024) if vulkan_found else 0,
        },
        "cpu": {
            "exists": cpu_found is not None,
            "filename": cpu_found.name if cpu_found else "muzero_cpu.vmfb",
            "size_kb": (cpu_found.stat().st_size // 1024) if cpu_found else 0,
        },
    }


def get_deck_manager() -> DeckManager:
    global _deck_manager
    if _deck_manager is None:
        _deck_manager = DeckManager()
    return _deck_manager


class BattleAPIHandler(SimpleHTTPRequestHandler):
    """Handles HTTP REST API endpoints with Multi-User Session isolation and serves static UI files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _get_session_id(self, req_data: Optional[dict] = None) -> str:
        """Resolves the caller's session ID from headers, body, cookies, or query string."""
        # 1. Header X-Session-ID
        header_sid = self.headers.get("X-Session-ID")
        if header_sid and header_sid.strip():
            return header_sid.strip()

        # 2. JSON request body
        if req_data and req_data.get("session_id"):
            body_sid = str(req_data.get("session_id")).strip()
            if body_sid:
                return body_sid

        # 3. Cookie header
        cookie_header = self.headers.get("Cookie")
        if cookie_header:
            cookie = http.cookies.SimpleCookie()
            try:
                cookie.load(cookie_header)
                if "ptcg_session_id" in cookie:
                    return cookie["ptcg_session_id"].value
                if "session_id" in cookie:
                    return cookie["session_id"].value
            except Exception:
                pass

        # 4. Query param ?session_id=...
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if "session_id" in qs and qs["session_id"]:
            return qs["session_id"][0]

        # 5. Generate fresh ID
        return str(uuid.uuid4())

    def _send_json(self, data: Any, status_code: int = 200, session_id: Optional[str] = None) -> None:
        if isinstance(data, dict) and session_id:
            data.setdefault("session_id", session_id)

        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session-ID")
        if session_id:
            self.send_header("Set-Cookie", f"ptcg_session_id={session_id}; Path=/; SameSite=Lax; Max-Age=86400")
            self.send_header("X-Session-ID", session_id)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session-ID")
        self.end_headers()

    def do_GET(self) -> None:
        # Periodically trigger cleanup of inactive sessions
        _session_manager.cleanup_expired()

        parsed = urlparse(self.path)
        path = parsed.path
        session_id = self._get_session_id()

        if path == "/api/health":
            self._send_json({
                "status": "ok",
                "service": "PTCG MuZero Battle Arena",
                "active_sessions": _session_manager.active_count(),
                "session_id": session_id,
            }, session_id=session_id)
            return

        if path == "/api/decks":
            dm = get_deck_manager()
            presets = dm.get_preset_decks()
            self._send_json({"status": "success", "decks": presets}, session_id=session_id)
            return

        if path == "/api/models/status":
            status_data = _get_models_status()
            self._send_json({"status": "success", "models": status_data}, session_id=session_id)
            return

        if path == "/api/battle/state":
            entry = _session_manager.get_session(session_id)
            if entry is None:
                self._send_json({
                    "status": "error",
                    "message": "No battle in progress for this session.",
                    "session_id": session_id,
                }, 404, session_id=session_id)
            else:
                with entry.lock:
                    state = entry.session.get_state()
                self._send_json({"status": "success", "state": state, "session_id": session_id}, session_id=session_id)
            return

        # Serve static files fallback (or index.html)
        if path == "/" or not (STATIC_DIR / path.lstrip("/")).exists():
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self) -> None:
        _session_manager.cleanup_expired()

        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            req_data = json.loads(body) if body else {}
        except Exception:
            req_data = {}

        session_id = self._get_session_id(req_data)

        if path == "/api/models/compile":
            target = req_data.get("target", "vulkan").lower()
            if target not in ("vulkan", "cpu"):
                target = "vulkan"

            output_name = "muzero_vulkan.vmfb" if target == "vulkan" else "muzero_cpu.vmfb"
            output_path = SCRIPT_DIR / output_name
            py_exe = _get_python_executable()
            export_script = SCRIPT_DIR / "export_iree.py"

            logger.info("Starting online model download & IREE compilation (target: %s, output: %s)...", target, output_name)
            cmd = [
                py_exe,
                str(export_script),
                "-m", "HF",
                "-o", str(output_path),
                "--target", target,
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(SCRIPT_DIR),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if proc.returncode != 0:
                    err_msg = proc.stderr.strip() or proc.stdout.strip() or f"Process exited with code {proc.returncode}"
                    logger.error("Model compilation failed: %s", err_msg)
                    self._send_json({"status": "error", "message": f"Compilation failed: {err_msg}"}, 500, session_id=session_id)
                    return

                if not output_path.exists():
                    self._send_json({"status": "error", "message": f"Compiled output not found: {output_name}"}, 500, session_id=session_id)
                    return

                size_kb = output_path.stat().st_size // 1024
                logger.info("✅ Model compilation completed successfully: %s (%d KB)", output_name, size_kb)
                self._send_json({
                    "status": "success",
                    "message": f"Model successfully compiled for {target.upper()}!",
                    "target": target,
                    "filename": output_name,
                    "size_kb": size_kb,
                }, session_id=session_id)
            except subprocess.TimeoutExpired:
                logger.error("Model compilation timed out after 300s.")
                self._send_json({"status": "error", "message": "Compilation timed out after 5 minutes."}, 504, session_id=session_id)
            except Exception as e:
                logger.exception("Unexpected error during model compilation: %s", e)
                self._send_json({"status": "error", "message": f"Compilation error: {e}"}, 500, session_id=session_id)
            return

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
                }, session_id=session_id)
            except Exception as e:
                logger.exception("Error generating deck: %s", e)
                m_deck = dm.get_model_deck()
                self._send_json({
                    "status": "success",
                    "deck": m_deck,
                    "summary": dm.get_deck_summary(m_deck),
                    "name": "⚡ MuZero AI Deck (Fallback)",
                }, session_id=session_id)
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
                self._send_json({"status": "error", "message": f"Invalid Player deck: {msg_p}"}, 400, session_id=session_id)
                return

            ok_ai, msg_ai = dm.validate_deck(ai_deck)
            if not ok_ai:
                self._send_json({"status": "error", "message": f"Invalid AI deck: {msg_ai}"}, 400, session_id=session_id)
                return

            try:
                entry = _session_manager.create_session(
                    session_id=session_id,
                    player_deck=player_deck,
                    ai_deck=ai_deck,
                    device_uri=device_uri,
                    ai_mode=ai_mode,
                )
                with entry.lock:
                    state = entry.session.start()
                self._send_json({"status": "success", "state": state, "session_id": entry.session_id}, session_id=entry.session_id)

            except Exception as e:
                logger.exception("Failed to start battle for session [%s]: %s", session_id, e)
                err_str = str(e)
                if "VMFB module not found" in err_str or isinstance(e, FileNotFoundError):
                    missing = "muzero_vulkan.vmfb" if device_uri == "vulkan" else "muzero_cpu.vmfb"
                    if ":" in err_str:
                        missing = err_str.split(":", 1)[1].strip()
                    self._send_json({
                        "status": "error",
                        "error_code": "VMFB_NOT_FOUND",
                        "target": device_uri,
                        "missing_file": missing,
                        "message": f"Start error: VMFB module not found: {missing}",
                    }, 404, session_id=session_id)
                else:
                    self._send_json({"status": "error", "message": f"Start error: {e}"}, 500, session_id=session_id)
            return

        if path == "/api/battle/step":
            entry = _session_manager.get_session(session_id)
            if entry is None:
                self._send_json({"status": "error", "message": "No active game for this session."}, 404, session_id=session_id)
                return

            selected_indices = req_data.get("selected_indices", [])
            try:
                with entry.lock:
                    state = entry.session.submit_action(selected_indices)
                self._send_json({"status": "success", "state": state, "session_id": session_id}, session_id=session_id)
            except Exception as e:
                logger.exception("Error during battle step for session [%s]: %s", session_id, e)
                self._send_json({"status": "error", "message": f"Error during action: {e}"}, 500, session_id=session_id)
            return

        if path == "/api/battle/reset":
            _session_manager.remove_session(session_id)
            self._send_json({"status": "success", "message": "Game reset for this session.", "session_id": session_id}, session_id=session_id)
            return

        self._send_json({"status": "error", "message": "Unknown endpoint."}, 404, session_id=session_id)


def main():
    parser = argparse.ArgumentParser(description="Multi-User Web Battle Server against MuZero model (IREE)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server listening port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Listening host (default: 0.0.0.0)")
    parser.add_argument("--ttl", type=int, default=1800, help="Session TTL timeout in seconds (default: 1800 / 30 min)")
    args = parser.parse_args()

    _session_manager.ttl_seconds = args.ttl
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, BattleAPIHandler)

    logger.info("==============================================================")
    logger.info("🔥 PTCG MuZero Multi-User Battle Arena started successfully!")
    logger.info("👉 Open your browser at: http://localhost:%d", args.port)
    logger.info("👥 Multi-User mode: Enabled (isolated sessions, TTL %ds)", args.ttl)
    logger.info("==============================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nStopping server...")
        _session_manager.close_all()
        httpd.server_close()


if __name__ == "__main__":
    main()
