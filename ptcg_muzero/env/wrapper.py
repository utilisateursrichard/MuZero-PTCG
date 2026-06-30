"""
ptcg_muzero/env/wrapper.py
===========================
Wraps the cabt Kaggle environment and provides:

* ``CabtEnv``   – thin sync wrapper around ``battle_start / battle_select``.
* ``GameHistory`` – trajectory object stored in the replay buffer.
* ``run_self_play_game`` – plays one full game between two MuZero policies
  and returns two ``GameHistory`` objects (one per player perspective).

The self-play loop lives here (not in JAX) because the cabt engine is a
Python/C++ binary that cannot be JIT-compiled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config import Config
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
# cabt environment thin wrapper
# ─────────────────────────────────────────────────────────────────────────────
class CabtEnv:
    """
    Minimal synchronous wrapper around the Kaggle cabt environment.

    Usage::

        env = CabtEnv()
        obs0, obs1, done = env.reset(deck0, deck1)
        while not done:
            obs0, obs1, done = env.step([action0], [action1])
        result = env.result  # 0 / 1 / 2 (draw)
    """

    def __init__(self) -> None:
        self._battle_started = False
        self._last_obs: Optional[Any] = None
        self.result: int = -1

    def reset(
        self,
        deck0: List[int],
        deck1: List[int],
    ) -> Tuple[Optional[dict], Optional[dict], bool]:
        """
        Start a new battle.

        Returns:
            obs_for_player0, obs_for_player1, done
        """
        try:
            from cabt import game as cabt_game  # type: ignore
            result = cabt_game.battle_start(deck0, deck1)
            obs_raw, start_data = result
        except Exception as exc:
            logger.error("battle_start failed: %s", exc)
            raise

        self._battle_started = True
        self.result = -1

        if obs_raw is None:
            logger.error("battle_start returned None observation: %s", start_data)
            return None, None, True

        obs_dict = _to_dict(obs_raw)
        done = _is_done(obs_dict)
        return obs_dict, obs_dict, done   # same obs sent to both; yourIndex disambiguates

    def step(
        self,
        select_p0: List[int],
        select_p1: List[int],
    ) -> Tuple[Optional[dict], Optional[dict], bool]:
        """
        Advance the game state by submitting both players' selections.
        Returns the new observation and whether the game is over.
        """
        try:
            from cabt import game as cabt_game  # type: ignore
            obs_raw = cabt_game.battle_select([select_p0, select_p1])
        except Exception as exc:
            logger.error("battle_select failed: %s", exc)
            raise

        obs_dict = _to_dict(obs_raw) if obs_raw is not None else {}
        done = _is_done(obs_dict)
        if done:
            self.result = obs_dict.get("current", {}).get("result", 2)
        return obs_dict, obs_dict, done

    def close(self) -> None:
        if self._battle_started:
            try:
                from cabt import game as cabt_game  # type: ignore
                cabt_game.battle_finish()
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

    The agent_fn is called for whichever player the env is waiting on.
    Steps where neither / both players need to act simultaneously are
    handled by forwarding the other player's action as the "pass" action
    (index 0 when they have only one legal option, or the first option).

    Returns:
        hist0, hist1  – GameHistory for player 0 and player 1.
    """
    env  = CabtEnv()
    hist = [GameHistory(player_idx=0), GameHistory(player_idx=1)]
    prev_logs: List[list] = [[], []]
    terminal_seen = [False, False]

    try:
        obs_dict, _, done = env.reset(deck0, deck1)
        if done or obs_dict is None:
            return hist[0], hist[1]

        while not done:
            your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
            select   = obs_dict.get("select") or {}
            options  = select.get("option", [])

            if not options:
                # Shouldn't happen, but guard
                obs_dict, _, done = env.step([], [])
                continue

            # Get current player's encoded obs for probing targets
            enc_obs = encode_observation(obs_dict, your_idx, cfg.model)
            hist[your_idx].raw_states.append(obs_dict)

            # Call agent function → returns selected indices + search info
            active_agent = (
                agent_fn[your_idx]
                if isinstance(agent_fn, (tuple, list))
                else agent_fn
            )
            action_indices, search_pol, search_val = active_agent(
                obs_dict, your_idx, cfg
            )

            # Compute reward from accumulated logs
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

            # Store in history
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

            # Step environment
            if your_idx == 0:
                obs_dict, _, done = env.step(action_indices, [0])
            else:
                obs_dict, _, done = env.step([0], action_indices)

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

        # ── Compute returns ───────────────────────────────────────────────
        for p in range(2):
            hist[p].compute_returns(cfg.train.gamma, cfg.train.td_steps)

    finally:
        env.close()

    return hist[0], hist[1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_dict(obs) -> dict:
    """Convert a cabt Observation object or dict to a plain dict."""
    if isinstance(obs, dict):
        return obs
    # cabt returns dataclass-like objects; use vars() or __dict__
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
    current = obs_dict.get("current")
    if current is None:
        return False
    result = current.get("result", -1)
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
