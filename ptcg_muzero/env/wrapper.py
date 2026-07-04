"""
ptcg_muzero/env/wrapper.py
===========================
Wraps the cabt Kaggle environment and provides:

* ``CabtEnv``        – thin sync wrapper around ``battle_start / battle_select``
                       from ``cg.game`` (same API as the reference notebook).
* ``GameHistory``    – trajectory object stored in the replay buffer.
* ``run_self_play_game`` – plays one full game between two agent functions and
  returns two ``GameHistory`` objects (one per player perspective).

Import pattern
--------------
All cabt / cg-lib symbols come from ``env.cabt_api`` which handles the
``glob``-based path discovery exactly as the reference Kaggle notebook does::

    sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0])
    from cg.game import battle_start, battle_select, battle_finish

The self-play loop lives here (not in JAX) because the cabt engine is a
Python/C++ binary that cannot be JIT-compiled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config import Config
from env.cabt_api import (
    battle_finish,
    battle_select,
    battle_start,
    to_observation_class,
)
from env.encoding import encode_observation, extract_step_reward

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Game trajectory (one perspective)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GameHistory:
    """
    Stores one player's perspective of a completed game.

    All lists share the same length ``T`` (number of decision steps
    for this player).
    """
    player_idx: int

    observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions:      List[np.ndarray]             = field(default_factory=list)
    rewards:      List[float]                  = field(default_factory=list)
    search_pols:  List[np.ndarray]             = field(default_factory=list)
    search_vals:  List[float]                  = field(default_factory=list)
    # Per-step decoded select context (for probing target extraction)
    select_types: List[int]                    = field(default_factory=list)
    raw_states:   List[dict]                   = field(default_factory=list)

    # Computed after the game ends
    returns:  Optional[np.ndarray] = None   # TD-n bootstrapped returns [T]
    game_won: Optional[bool]       = None

    def __len__(self) -> int:
        return len(self.actions)

    def compute_returns(self, gamma: float, td_steps: int) -> None:
        """
        Compute n-step bootstrap targets in-place.
        Returns are used as value targets during MuZero training.
        """
        T = len(self.rewards)
        returns = np.zeros(T, dtype=np.float32)
        for t in range(T):
            G = 0.0
            for k in range(td_steps):
                if t + k < T:
                    G += (gamma ** k) * self.rewards[t + k]
            # Bootstrap with search value if within horizon
            if t + td_steps < T:
                G += (gamma ** td_steps) * self.search_vals[t + td_steps]
            returns[t] = G
        self.returns = returns


# ─────────────────────────────────────────────────────────────────────────────
# Deck error (rethrown as a specific type so callers can retry)
# ─────────────────────────────────────────────────────────────────────────────
class DeckError(ValueError):
    """Raised when battle_start() rejects a deck (invalid cards, duplicates, etc)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# cabt environment thin wrapper
# ─────────────────────────────────────────────────────────────────────────────
class CabtEnv:
    """
    Minimal synchronous wrapper around the Kaggle cabt environment.

    Mirrors the reference notebook's loop::

        obs, start_data = battle_start(deck0, deck1)
        while obs["current"]["result"] < 0:
            selected = agent(obs)
            obs = battle_select(selected)
        battle_finish()

    Usage::

        env = CabtEnv()
        obs_dict, done = env.reset(deck0, deck1)
        while not done:
            obs_dict, done = env.step([action])
        result = env.result   # 0 / 1 / 2  (player 0 win / player 1 win / draw)
    """

    def __init__(self) -> None:
        self._battle_started = False
        self._last_obs: Optional[dict] = None
        self.result: int = -1

    # ------------------------------------------------------------------
    def reset(
        self,
        deck0: List[int],
        deck1: List[int],
    ) -> Tuple[Optional[dict], bool]:
        """
        Start a new battle.

        Calls ``battle_start(deck0, deck1)`` exactly as the reference notebook.
        Validates the ``start_data`` error flags and raises on deck errors.

        Returns:
            obs_dict, done
        """
        obs_raw, start_data = battle_start(deck0, deck1)

        # ── Deck validation (copied verbatim from reference notebook) ──────
        if start_data.errorPlayer >= 0:
            error = "Deck error."
            if start_data.errorType == 1:
                error = "The deck contains invalid card ID."
            elif start_data.errorType == 2:
                error = (
                    "You can include up to four cards with the same name in the "
                    "deck, excluding basic Energy cards."
                )
            elif start_data.errorType == 3:
                error = "There are no Basic Pokémon in the deck."
            elif start_data.errorType == 4:
                error = "You can include only one Ace Spec card in the deck."
            raise DeckError(error)

        self._battle_started = True
        self.result = -1

        # obs_raw is already a plain dict (the cg.game API returns a dict)
        obs_dict = obs_raw if isinstance(obs_raw, dict) else _obs_to_dict(obs_raw)
        done = _is_done(obs_dict)
        self._last_obs = obs_dict
        return obs_dict, done

    # ------------------------------------------------------------------
    def step(self, select_list: List[int]) -> Tuple[Optional[dict], bool]:
        """
        Advance the game by submitting the current player's selection.

        Calls ``battle_select(select_list)`` exactly as the reference notebook::

            obs = battle_select(selected)

        Returns:
            obs_dict, done
        """
        obs_raw = battle_select(select_list)

        obs_dict = obs_raw if isinstance(obs_raw, dict) else _obs_to_dict(obs_raw)
        done = _is_done(obs_dict)
        if done:
            self.result = obs_dict.get("current", {}).get("result", 2)
        self._last_obs = obs_dict
        return obs_dict, done

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Call ``battle_finish()`` to release engine resources."""
        if self._battle_started:
            try:
                battle_finish()
            except Exception:
                pass
            self._battle_started = False


# ─────────────────────────────────────────────────────────────────────────────
# Self-play game runner
# ─────────────────────────────────────────────────────────────────────────────
AgentFn = Callable[[dict, int, "Config"], Tuple[List[int], np.ndarray, float]]
"""
Signature:  action_indices, search_policy, search_value
              = agent_fn(obs_dict, player_idx, config)
"""


def run_self_play_game(
    agent_fn: AgentFn,
    deck0: List[int],
    deck1: List[int],
    cfg: Config,
    rng: np.random.Generator,
    add_dirichlet_noise: bool = True,
) -> Tuple[GameHistory, GameHistory]:
    """
    Play one complete game and return trajectories for both players.

    The game loop matches the reference notebook::

        obs, start_data = battle_start(deck0, deck1)
        while obs["current"]["result"] < 0:
            your_index = obs["current"]["yourIndex"]
            selected = agent(obs)
            obs = battle_select(selected)
        battle_finish()

    ``agent_fn`` is called for whichever player the env is waiting on.
    Supports passing a tuple/list of two separate agent functions
    (one per player index).

    Returns:
        hist0, hist1  – GameHistory for player 0 and player 1.
    """
    env  = CabtEnv()
    hist = [GameHistory(player_idx=0), GameHistory(player_idx=1)]
    prev_logs: List[list] = [[], []]
    terminal_seen = [False, False]

    try:
        obs_dict, done = env.reset(deck0, deck1)
        if done or obs_dict is None:
            return hist[0], hist[1]

        step_count = 0
        while not done:
            step_count += 1
            your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
            select   = obs_dict.get("select")
            
            # Diagnostic log
            from training.trainer import logger as t_logger
            t_logger.info(
                "[game step %d] your_idx=%d select_is_none=%s options_count=%d",
                step_count, your_idx, select is None, 0 if select is None else len(select.get("option", []))
            )
            
            from training.activity import tracker
            tracker.update(current_game_steps=step_count)
            
            # Si select est absent, c'est l'initialisation (soumission du deck)
            if select is None:
                deck_to_submit = deck0 if your_idx == 0 else deck1
                obs_dict, done = env.step(deck_to_submit)
                continue
                
            options  = select.get("option", [])
            if not options:
                # No options available – auto-pass (engine may still advance)
                obs_dict, done = env.step([])
                continue

            # ── Encode observation ──────────────────────────────────────────
            enc_obs = encode_observation(obs_dict, your_idx, cfg.model)
            option_mask = enc_obs["option_mask"]
            if np.sum(option_mask) == 0:
                obs_dict, done = env.step([])
                continue

            hist[your_idx].raw_states.append(obs_dict)

            # ── Select agent ───────────────────────────────────────────────
            active_agent = (
                agent_fn[your_idx]
                if isinstance(agent_fn, (tuple, list))
                else agent_fn
            )
            action_indices, search_pol, search_val = active_agent(
                obs_dict, your_idx, cfg
            )

            # Validation des indices d'action pour éviter les IndexError fatals
            valid_action_indices = []
            for idx in action_indices:
                idx_int = int(idx)
                if 0 <= idx_int < len(options) and options[idx_int] is not None:
                    valid_action_indices.append(idx_int)
                else:
                    fallback_idx = next((i for i, opt in enumerate(options) if opt is not None), 0)
                    logger.warning(
                        f"[run_self_play_game] Action choisie invalide ({idx}) pour options {options}. "
                        f"Fallback vers {fallback_idx}."
                    )
                    valid_action_indices.append(fallback_idx)
            action_indices = valid_action_indices

            # ── Reward from logs ───────────────────────────────────────────
            logs = obs_dict.get("logs", [])
            if len(logs) >= len(prev_logs[your_idx]):
                new_logs = logs[len(prev_logs[your_idx]):]
            else:
                new_logs = logs
            prev_logs[your_idx] = list(logs)
            terminal_seen[your_idx] = terminal_seen[your_idx] or any(
                _log_int(log, "type", -1) == 23 for log in new_logs
            )
            reward = extract_step_reward(new_logs, your_idx)

            # ── Store step in history ──────────────────────────────────────
            action_vec = np.zeros(cfg.model.max_actions, dtype=np.float32)
            for idx in action_indices:
                if 0 <= int(idx) < cfg.model.max_actions:
                    action_vec[int(idx)] = 1.0
            hist[your_idx].observations.append(enc_obs)
            hist[your_idx].actions.append(action_vec)
            hist[your_idx].rewards.append(reward)
            hist[your_idx].search_pols.append(search_pol)
            hist[your_idx].search_vals.append(float(search_val))
            hist[your_idx].select_types.append(int(select.get("type", 0)))

            # ── Advance environment ────────────────────────────────────────
            obs_dict, done = env.step(action_indices)

        # ── Terminal reward adjustment ─────────────────────────────────────
        result = env.result
        for p in range(2):
            if hist[p].rewards:
                if result == p:
                    if not terminal_seen[p]:
                        hist[p].rewards[-1] += 1.0
                    hist[p].game_won = True
                elif result == 1 - p:
                    if not terminal_seen[p]:
                        hist[p].rewards[-1] -= 1.0
                    hist[p].game_won = False
                else:
                    hist[p].game_won = None   # draw

        # ── Compute bootstrapped returns ───────────────────────────────────
        for p in range(2):
            hist[p].compute_returns(cfg.train.gamma, cfg.train.td_steps)

    finally:
        env.close()

    return hist[0], hist[1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _obs_to_dict(obs) -> dict:
    """
    Convert a cg Observation object (or any dataclass/object) to a plain dict.

    ``cg.game.battle_start`` / ``battle_select`` return plain Python dicts on
    the Kaggle platform, but we guard against other representations just in case.
    """
    if isinstance(obs, dict):
        return obs
    try:
        import dataclasses
        if dataclasses.is_dataclass(obs):
            return dataclasses.asdict(obs)
    except Exception:
        pass
    try:
        return dict(vars(obs))
    except Exception:
        return {"select": None, "logs": [], "current": None}


def _is_done(obs_dict: dict) -> bool:
    """Return True when ``current.result`` is 0, 1, or 2 (game over)."""
    current = obs_dict.get("current")
    if current is None:
        return False
    result = current.get("result", -1) if isinstance(current, dict) else getattr(current, "result", -1)
    return int(result) >= 0


def _log_int(obj, key: str, default: int) -> int:
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    if hasattr(v, "value"):
        return int(v.value)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def self_play_worker_fn(pipe, worker_id, cfg):
    """
    Worker loop designed to run in a spawned subprocess.
    """
    import os
    # Force JAX to only use CPU in subprocesses to avoid CUDA conflicts/allocations
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    jax.config.update("jax_platform_name", "cpu")
    import numpy as np

    from env.wrapper import CabtEnv, DeckError
    from env.encoding import encode_observation, extract_step_reward
    from search.ismcts import sample_belief

    env = CabtEnv()

    while True:
        try:
            msg = pipe.recv()
            if msg is None:  # Shutdown signal
                break

            cmd = msg.get("cmd")
            if cmd == "start":
                deck0 = msg["deck0"]
                deck1 = msg["deck1"]
                num_belief_samples = msg.get("num_belief_samples", cfg.search.num_belief_samples)

                try:
                    obs_dict, done = env.reset(deck0, deck1)
                except DeckError as e:
                    pipe.send({"status": "deck_error", "error": str(e)})
                    continue

                step_count = 0
                prev_logs = [[], []]
                from env.wrapper import GameHistory
                hist = [GameHistory(player_idx=0), GameHistory(player_idx=1)]

                while not done:
                    step_count += 1
                    your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
                    select = obs_dict.get("select")

                    if select is None:
                        deck_to_submit = deck0 if your_idx == 0 else deck1
                        obs_dict, done = env.step(deck_to_submit)
                        continue

                    options = select.get("option", [])
                    if not options:
                        obs_dict, done = env.step([])
                        continue

                    mc = cfg.model
                    N_samples = int(num_belief_samples)

                    rng_beliefs = np.random.randint(0, 2**31, size=N_samples)

                    det_list = []
                    for s in range(N_samples):
                        det_list.append(sample_belief(obs_dict, rng_beliefs[s], mc))

                    encoded_samples = [encode_observation(d, your_idx, mc) for d in det_list]

                    option_mask = encoded_samples[0]["option_mask"]
                    if np.sum(option_mask) == 0:
                        obs_dict, done = env.step([])
                        continue

                    # Stack observations
                    batched_enc = {}
                    for k in encoded_samples[0].keys():
                        batched_enc[k] = np.stack([x[k] for x in encoded_samples], axis=0)

                    option_mask = encoded_samples[0]["option_mask"]

                    # Request action from GPU coordinator
                    pipe.send({
                        "status": "need_action",
                        "batched_enc": batched_enc,
                        "option_mask": option_mask,
                        "player_idx": your_idx,
                        "step_count": step_count
                    })

                    # Receive action from GPU coordinator
                    action_msg = pipe.recv()
                    best_action = action_msg["action_indices"][0]
                    search_pol = action_msg["search_pol"]
                    search_val = action_msg["search_val"]

                    # Déterminer les indices réels en gérant maxCount > 1 (sélection top-k par politique MCTS)
                    max_count = int(select.get("maxCount", 1))
                    if max_count > 1:
                        valid_mask = np.array([
                            (opt is not None) for opt in options
                        ] + [False] * (mc.max_actions - len(options)))
                        masked = np.where(option_mask & valid_mask, search_pol, -1e9)
                        action_indices = np.argsort(masked)[::-1][:max_count].tolist()
                    else:
                        action_indices = [int(best_action)]

                    # Validation finale des indices d'action pour éviter les IndexError fatals
                    valid_action_indices = []
                    for idx in action_indices:
                        idx_int = int(idx)
                        if 0 <= idx_int < len(options) and options[idx_int] is not None:
                            valid_action_indices.append(idx_int)
                        else:
                            fallback_idx = next((i for i, opt in enumerate(options) if opt is not None), 0)
                            logger.warning(
                                f"[worker-{worker_id}] Action choisie invalide ({idx}) pour options {options}. "
                                f"Fallback vers {fallback_idx}."
                            )
                            valid_action_indices.append(fallback_idx)
                    action_indices = valid_action_indices

                    hist[your_idx].raw_states.append(obs_dict)

                    logs = obs_dict.get("logs", [])
                    if len(logs) >= len(prev_logs[your_idx]):
                        new_logs = logs[len(prev_logs[your_idx]):]
                    else:
                        new_logs = logs
                    prev_logs[your_idx] = list(logs)

                    reward = extract_step_reward(new_logs, your_idx)

                    action_vec = np.zeros(mc.max_actions, dtype=np.float32)
                    for idx in action_indices:
                        if 0 <= int(idx) < mc.max_actions:
                            action_vec[int(idx)] = 1.0

                    hist[your_idx].observations.append(encoded_samples[0])
                    hist[your_idx].actions.append(action_vec)
                    hist[your_idx].rewards.append(reward)
                    hist[your_idx].search_pols.append(search_pol)
                    hist[your_idx].search_vals.append(float(search_val))
                    hist[your_idx].select_types.append(int(select.get("type", 0)))

                    obs_dict, done = env.step(action_indices)

                result = env.result
                for p in range(2):
                    if hist[p].rewards:
                        if result == p:
                            hist[p].rewards[-1] += 1.0
                            hist[p].game_won = True
                        elif result == 1 - p:
                            hist[p].rewards[-1] -= 1.0
                            hist[p].game_won = False
                        else:
                            hist[p].game_won = None

                for p in range(2):
                    hist[p].compute_returns(cfg.train.gamma, cfg.train.td_steps)

                pipe.send({
                    "status": "game_over",
                    "hist0": hist[0],
                    "hist1": hist[1]
                })

        except Exception as e:
            import traceback
            err_msg = f"{e}\n{traceback.format_exc()}"
            logger.error(f"[worker-{worker_id}] Exception fatale rencontree : {err_msg}")
            try:
                pipe.send({"status": "error", "error": err_msg})
            except Exception as pipe_err:
                logger.error(f"[worker-{worker_id}] Impossible d'envoyer l'erreur via le pipe : {pipe_err}")
            break

    env.close()

