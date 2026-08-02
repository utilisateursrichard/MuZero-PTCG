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
        self.start_step = 0     # step initial de la session (depuis la reprise/changement)
        self.current_step = 0   # step d'entraînement courant (global)
        self.rep_frozen = True
        self.h_status = "GÉLÉ (initialisation)"
        
        # Running lists for intelligent ETA estimation
        self.train_step_times = []
        self.self_play_times = []
        self.transitions_per_game_list = []

    @property
    def new_step(self) -> int:
        """Nombre de steps exécutés durant cette session d'entraînement."""
        return max(0, self.current_step - self.start_step)

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

    def update(
        self,
        phase=None,
        buffer_size=None,
        deck_errors=None,
        current_game_steps=None,
        games_completed=None,
        step=None,
        start_step=None,
        rep_frozen=None,
        h_status=None,
    ):
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
        if start_step is not None:
            self.start_step = start_step
        if rep_frozen is not None:
            self.rep_frozen = rep_frozen
        if h_status is not None:
            self.h_status = h_status
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

def format_h_status(
    step: int,
    start_step: int,
    rep_frozen: bool,
    loss_window_len: int,
    w_size: int,
    s_min: int,
    eps: float,
    current_gain: float = None,
    avg_step_time: float = 0.04,
) -> str:
    """Génère un indicateur textuel décrivant si h(s) est gelé et estimant quand il sera dégelé."""
    if not rep_frozen:
        return "DÉGELÉ"

    new_step = max(0, step - start_step)

    # Phase 1 : Attente des conditions (steps nouveaux < s_min ou fenêtre loss non pleine)
    if new_step < s_min or loss_window_len < w_size:
        rem_steps_s_min = max(0, s_min - new_step)
        rem_steps_w = max(0, w_size - loss_window_len)
        rem_steps = max(rem_steps_s_min, rem_steps_w)

        est_sec = rem_steps * avg_step_time
        if est_sec >= 60:
            m = int(est_sec // 60)
            s = int(est_sec % 60)
            t_str = f"{m}m{s:02d}s"
        else:
            t_str = f"{est_sec:.0f}s"

        return f"GÉLÉ (≥{rem_steps} steps nouv. restants [{new_step}/{s_min}] | ~{t_str})"

    # Phase 2 : Évaluation active du plateau
    if current_gain is not None:
        target_pct = eps * 100.0
        gain_pct = current_gain * 100.0
        if current_gain > eps:
            return f"GÉLÉ (gain={gain_pct:.2f}% -> cible <{target_pct:.2f}%)"
        else:
            return f"GÉLÉ (plateau atteint: gain={gain_pct:.2f}% < {target_pct:.2f}%)"

    return "GÉLÉ (évaluation plateau...)"


def dump_all_stacks():
    """Affiche la stack-trace de tous les threads actifs."""
    logger.warning("=== DETECTED FREEZE - DUMPING ALL THREAD STACKS ===")
    for thread_id, frame in sys._current_frames().items():
        logger.warning(f"\n--- Thread ID: {thread_id} ---")
        traceback.print_stack(frame)
    logger.warning("==================================================")
