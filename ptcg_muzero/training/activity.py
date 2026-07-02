import time
import sys
import traceback
import logging

logger = logging.getLogger("ptcg_muzero.activity")

class ActivityTracker:
    def __init__(self):
        self.phase = "Initialisation"
        self.buffer_size = 0
        self.deck_errors = 0
        self.current_game_steps = 0
        self.last_activity_time = time.time()

    def update(self, phase=None, buffer_size=None, deck_errors=None, current_game_steps=None):
        if phase is not None:
            self.phase = phase
        if buffer_size is not None:
            self.buffer_size = buffer_size
        if deck_errors is not None:
            self.deck_errors = deck_errors
        if current_game_steps is not None:
            self.current_game_steps = current_game_steps
        self.last_activity_time = time.time()

# Singleton partagé
tracker = ActivityTracker()

def dump_all_stacks():
    """Affiche la stack-trace de tous les threads actifs."""
    logger.warning("=== DETECTED FREEZE - DUMPING ALL THREAD STACKS ===")
    for thread_id, frame in sys._current_frames().items():
        logger.warning(f"\\n--- Thread ID: {thread_id} ---")
        traceback.print_stack(frame)
    logger.warning("==================================================")
