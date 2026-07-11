import time
import sys
import traceback
import logging

logger = logging.getLogger("ptcg_muzero.activity")

class ActivityTracker:
    def __init__(self):
        self.phase = "Initialisation"
        self._live_buffer = None   # référence directe au PrioritizedReplayBuffer (setée par trainer)
        self._buffer_size_fallback = 0  # utilisé si _live_buffer n'est pas encore attaché
        self.deck_errors = 0
        self.current_game_steps = 0
        self.last_activity_time = time.time()
        self.games_completed = 0
        self.start_time = time.time()
        self.current_step = 0   # step d'entraînement courant
        
        # Running lists for intelligent ETA estimation
        self.train_step_times = []
        self.self_play_times = []
        self.transitions_per_game_list = []

    @property
    def buffer_size(self) -> int:
        """Taille du buffer EN DIRECT : lit `len(buffer)` si le buffer est attaché."""
        if self._live_buffer is not None:
            try:
                return len(self._live_buffer)
            except Exception:
                pass
        return self._buffer_size_fallback

    def attach_buffer(self, buffer) -> None:
        """Attache le replay buffer pour une lecture en direct de la taille."""
        self._live_buffer = buffer

    def update(self, phase=None, buffer_size=None, deck_errors=None, current_game_steps=None, games_completed=None, step=None):
        if phase is not None:
            self.phase = phase
        if buffer_size is not None:
            self._buffer_size_fallback = buffer_size  # fallback si buffer pas encore attaché
        if deck_errors is not None:
            self.deck_errors = deck_errors
        if current_game_steps is not None:
            self.current_game_steps = current_game_steps
        if games_completed is not None:
            self.games_completed = games_completed
        if step is not None:
            self.current_step = step
        self.last_activity_time = time.time()

    @property
    def avg_train_step_time(self) -> float:
        if not self.train_step_times:
            return 0.04  # default guess
        return sum(self.train_step_times[-100:]) / len(self.train_step_times[-100:])

    @property
    def avg_self_play_time(self) -> float:
        if not self.self_play_times:
            return 80.0  # default guess (80 seconds for 8 games)
        return sum(self.self_play_times[-10:]) / len(self.self_play_times[-10:])

    @property
    def avg_transitions_per_game(self) -> float:
        if not self.transitions_per_game_list:
            return 50.0  # default guess
        return sum(self.transitions_per_game_list[-50:]) / len(self.transitions_per_game_list[-50:])

# Singleton partagé
tracker = ActivityTracker()

def dump_all_stacks():
    """Affiche la stack-trace de tous les threads actifs."""
    logger.warning("=== DETECTED FREEZE - DUMPING ALL THREAD STACKS ===")
    for thread_id, frame in sys._current_frames().items():
        logger.warning(f"\\n--- Thread ID: {thread_id} ---")
        traceback.print_stack(frame)
    logger.warning("==================================================")
