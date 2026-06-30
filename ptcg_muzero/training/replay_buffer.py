"""
ptcg_muzero/training/replay_buffer.py
========================================
Replay buffer priorisé (Prioritized Experience Replay).

Implémentation numpy pure avec un segment tree pour les opérations O(log N).
Le buffer est pensé pour tourner dans le process principal (CPU) pendant que
le learner GPU consomme des batches.

Contenu d'une entrée (``ReplayEntry``)
---------------------------------------
  obs_seq     : list[dict]     observations encodées (longueur = num_unroll+1)
  action_seq  : float[][]      multi-hot actions  (num_unroll × max_actions)
  reward_seq  : float[]        récompenses  (longueur = num_unroll)
  target_pol  : float[][]      politique MCTS  (longueur = num_unroll+1, A)
  target_val  : float[]        valeurs retour  (longueur = num_unroll+1)
  probe_tgts  : int[][]        cibles de sonde (longueur = num_unroll+1, 5)
  priority    : float          priorité initiale
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ModelConfig, TrainConfig


# ─────────────────────────────────────────────────────────────────────────────
# Segment tree (min + sum) pour PER
# ─────────────────────────────────────────────────────────────────────────────
class SegmentTree:
    """Segment tree supporting sum and min queries in O(log N)."""

    def __init__(self, capacity: int, reduction: str = "sum") -> None:
        self._capacity = capacity
        self._size     = 1
        while self._size < capacity:
            self._size *= 2
        self._tree = np.zeros(2 * self._size, dtype=np.float64)
        self._reduction = np.add if reduction == "sum" else np.minimum
        if reduction == "min":
            self._tree[:] = np.inf

    def _reduce(self, a, b):
        if self._reduction is np.add:
            return a + b
        return np.minimum(a, b)

    def set(self, idx: int, value: float) -> None:
        idx += self._size
        self._tree[idx] = value
        idx >>= 1
        while idx >= 1:
            self._tree[idx] = self._reduce(
                self._tree[2 * idx], self._tree[2 * idx + 1]
            )
            idx >>= 1

    def query_all(self) -> float:
        return float(self._tree[1])

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Find largest idx such that sum(tree[0:idx]) <= prefixsum."""
        idx = 1
        while idx < self._size:
            if self._tree[2 * idx] <= prefixsum:
                prefixsum -= self._tree[2 * idx]
                idx = 2 * idx + 1
            else:
                idx = 2 * idx
        return idx - self._size


# ─────────────────────────────────────────────────────────────────────────────
# Replay entry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ReplayEntry:
    obs_seq:    List[Dict]
    action_seq: np.ndarray   # [num_unroll, max_actions] float32
    reward_seq: np.ndarray   # [num_unroll]     float32
    target_pol: np.ndarray   # [num_unroll+1, A] float32
    target_val: np.ndarray   # [num_unroll+1]   float32
    probe_tgts: np.ndarray   # [num_unroll+1, 5] int32
    priority:   float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Buffer principal
# ─────────────────────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:

    def __init__(self, cfg_train: TrainConfig, cfg_model: ModelConfig) -> None:
        self._max_size  = cfg_train.replay_buffer_size
        self._alpha     = cfg_train.replay_alpha
        self._beta      = cfg_train.replay_beta
        self._unroll    = cfg_train.num_unroll_steps
        self._batch_sz  = cfg_train.batch_size
        self._max_acts  = cfg_model.max_actions

        self._entries: List[Optional[ReplayEntry]] = [None] * self._max_size
        self._cursor  = 0
        self._size    = 0

        self._sum_tree = SegmentTree(self._max_size, "sum")
        self._min_tree = SegmentTree(self._max_size, "min")
        self._max_prio = 1.0

        self._lock = threading.Lock()

    # ── Public ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def add_game(
        self,
        obs_list:    List[dict],       # T+1 encoded obs
        actions:     List[np.ndarray], # T, each [max_actions] multi-hot
        rewards:     List[float],      # T
        search_pols: List[np.ndarray], # T+1, each [max_actions]
        returns:     np.ndarray,       # T+1 float32
        probe_tgts:  np.ndarray,       # T+1, [5] int32
        unroll:      int = -1,
    ) -> int:
        """
        Slice the game into (num_unroll+1)-length windows and add each.
        Returns number of entries added.
        """
        if unroll < 0:
            unroll = self._unroll
        T = len(actions)
        added = 0

        for t in range(T):
            end = min(t + unroll, T)
            act_window  = np.array(actions[t:end],  dtype=np.float32)
            rew_window  = np.array(rewards[t:end],  dtype=np.float32)
            obs_window  = obs_list[t:end + 1]
            pol_window  = np.stack([
                search_pols[s] if s < len(search_pols)
                else np.zeros(self._max_acts, np.float32)
                for s in range(t, end + 1)
            ])
            val_window  = returns[t:end + 1]
            prb_window  = probe_tgts[t:end + 1]

            # Pad to fixed length
            pad_len = unroll - len(act_window)
            if pad_len > 0:
                act_window = np.concatenate(
                    [
                        act_window,
                        np.zeros((pad_len, self._max_acts), np.float32),
                    ],
                    axis=0,
                )
                rew_window = np.concatenate(
                    [rew_window, np.zeros(pad_len, np.float32)]
                )
                # obs / pol / val / prb pads are handled by the learner

            entry = ReplayEntry(
                obs_seq    = obs_window,
                action_seq = act_window,
                reward_seq = rew_window,
                target_pol = pol_window,
                target_val = val_window,
                probe_tgts = prb_window,
                priority   = self._max_prio,
            )
            self._insert(entry)
            added += 1

        return added

    def sample(self, batch_size: Optional[int] = None) -> Tuple[List[ReplayEntry], np.ndarray, np.ndarray]:
        """
        Sample a batch using PER.

        Returns:
            entries     : List[ReplayEntry]  length B
            indices     : np.ndarray [B]     for priority updates
            is_weights  : np.ndarray [B]     importance-sampling weights
        """
        with self._lock:
            if batch_size is None:
                batch_size = self._batch_sz
            assert self._size > 0, "Buffer is empty."

            total   = self._sum_tree.query_all()
            segment = total / batch_size
            entries, indices = [], []

            rng = np.random.default_rng()
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = rng.uniform(a, b)
                idx = self._sum_tree.find_prefixsum_idx(s)
                idx = np.clip(idx, 0, self._size - 1)
                indices.append(idx)
                entries.append(self._entries[idx])

            # IS weights
            min_p      = self._min_tree.query_all() / total
            max_weight = (min_p * self._size) ** (-self._beta)
            weights = np.array([
                ((self._sum_tree._tree[i + self._sum_tree._size] / total)
                 * self._size) ** (-self._beta)
                for i in indices
            ], dtype=np.float32)
            weights = weights / max_weight

            return entries, np.array(indices), weights

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        with self._lock:
            for idx, prio in zip(indices, priorities):
                prio = float(max(prio, 1e-6)) ** self._alpha
                self._max_prio = max(self._max_prio, prio)
                self._sum_tree.set(int(idx), prio)
                self._min_tree.set(int(idx), prio)

    # ── Internal ──────────────────────────────────────────────────────────

    def _insert(self, entry: ReplayEntry) -> None:
        with self._lock:
            idx = self._cursor
            self._entries[idx] = entry
            prio = self._max_prio ** self._alpha
            self._sum_tree.set(idx, prio)
            self._min_tree.set(idx, prio)
            self._cursor = (self._cursor + 1) % self._max_size
            self._size   = min(self._size + 1, self._max_size)
