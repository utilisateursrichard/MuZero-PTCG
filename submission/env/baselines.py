"""
ptcg_muzero/env/baselines.py
=============================
Adversaires de référence pour ``main.py test``.

Pourquoi ce module
------------------
En self-play pur, le taux de victoire vaut 50 % par construction : les deux
camps partagent le même réseau et le même deck.  Aucune courbe d'entraînement
(entropie, p_max, distribution d'actions) ne dit si l'agent joue *mieux* —
seulement *comment* il joue.  Il faut un étalon extérieur et stable dans le
temps.

Deux adversaires, sans aucun apprentissage :

* ``random``  — plancher absolu.  Un agent entraîné qui ne le bat pas
  largement a un problème grave et immédiat.
* ``greedy``  — heuristique gloutonne pilotée par les ``OptionType`` du moteur,
  avec une molette de difficulté ``epsilon`` : ``epsilon=1.0`` équivaut
  exactement à ``random``, ``epsilon=0.0`` donne la pleine force.  Le continuum
  permet de situer le réseau sans avoir besoin d'un Elo.

Les deux respectent la signature ``AgentFn`` de ``env/wrapper.py`` :

    action_indices, search_policy, search_value = agent_fn(obs_dict, player_idx, cfg)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# ── OptionType du moteur (cf. training/activity.py:record_action) ─────────────
T_CARD, T_TOOLCARD, T_ENERGYCARD, T_ENERGY = 3, 4, 5, 6
T_PLAY, T_ATTACH, T_EVOLVE, T_ABILITY = 7, 8, 9, 10
T_DISCARD, T_RETREAT, T_ATTACK, T_END = 11, 12, 13, 14

# Ordre de préférence de l'heuristique.  Attaquer prime sur tout : c'est la
# seule action qui prend des prizes, donc la seule qui gagne la partie.
# END arrive en dernier — passer son tour est toujours le pire choix légal.
GREEDY_PRIORITY: List[int] = [
    T_ATTACK,      # 13 — prendre des prizes
    T_EVOLVE,      # 9  — développer le board
    T_ATTACH,      # 8  — armer l'attaquant
    T_PLAY,        # 7  — poser une carte
    T_ABILITY,     # 10 — utiliser un talent
    T_ENERGYCARD,  # 5
    T_ENERGY,      # 6
    T_CARD,        # 3
    T_TOOLCARD,    # 4
    T_DISCARD,     # 11
    T_RETREAT,     # 12 — coûteux, rarement bon
    T_END,         # 14 — dernier recours
]
_PRIORITY_RANK = {t: i for i, t in enumerate(GREEDY_PRIORITY)}
_UNKNOWN_RANK = len(GREEDY_PRIORITY) - 1  # juste avant END


def _opt_type(opt) -> int:
    if opt is None:
        return -1
    v = opt.get("type", 0) if isinstance(opt, dict) else getattr(opt, "type", 0)
    if hasattr(v, "value"):
        v = v.value
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def _legal_indices(obs_dict) -> tuple:
    """Retourne (indices_legaux, options)."""
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else getattr(obs_dict, "select", None)
    if select is None:
        return [], []
    options = select.get("option", []) if isinstance(select, dict) else (getattr(select, "option", []) or [])
    idx = [i for i, o in enumerate(options) if o is not None]
    return idx, options


def _max_count(obs_dict) -> int:
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else getattr(obs_dict, "select", None)
    if select is None:
        return 1
    v = select.get("maxCount", 1) if isinstance(select, dict) else getattr(select, "maxCount", 1)
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return 1


def make_random_agent(cfg, seed: Optional[int] = None):
    """Agent uniforme sur les options légales — le plancher de comparaison."""
    rng = np.random.default_rng(seed)
    A = cfg.model.max_actions

    def agent_fn(obs_dict, player_idx, _cfg):
        idx, _ = _legal_indices(obs_dict)
        if not idx:
            return [0], np.zeros(A, dtype=np.float32), 0.0
        k = min(_max_count(obs_dict), len(idx))
        chosen = rng.choice(idx, size=k, replace=False).tolist()
        pol = np.zeros(A, dtype=np.float32)
        for i in idx:
            if i < A:
                pol[i] = 1.0 / len(idx)
        return [int(c) for c in chosen], pol, 0.0

    return agent_fn


def make_greedy_agent(cfg, epsilon: float = 0.0, seed: Optional[int] = None):
    """Heuristique gloutonne par type d'option, avec bruit ε-greedy.

    Paramètres
    ----------
    epsilon : molette de difficulté dans [0, 1].
        ``0.0`` → toujours le meilleur type disponible (pleine force).
        ``1.0`` → strictement équivalent à ``make_random_agent``.
        Les valeurs intermédiaires donnent un continuum de force, ce qui permet
        de situer le réseau : on monte ε jusqu'à ce qu'il revienne à 50 %.

    L'heuristique ne regarde que le *type* de l'action, jamais l'état du board :
    c'est volontaire.  Elle doit rester triviale à auditer et parfaitement
    stable dans le temps, sinon elle cesse d'être un étalon.
    """
    rng = np.random.default_rng(seed)
    A = cfg.model.max_actions
    eps = float(np.clip(epsilon, 0.0, 1.0))

    def agent_fn(obs_dict, player_idx, _cfg):
        idx, options = _legal_indices(obs_dict)
        if not idx:
            return [0], np.zeros(A, dtype=np.float32), 0.0

        ranks = np.array(
            [_PRIORITY_RANK.get(_opt_type(options[i]), _UNKNOWN_RANK) for i in idx]
        )
        order = np.argsort(ranks, kind="stable")
        ranked = [idx[j] for j in order]

        k = min(_max_count(obs_dict), len(idx))
        if eps > 0.0 and rng.random() < eps:
            chosen = rng.choice(idx, size=k, replace=False).tolist()
        else:
            chosen = ranked[:k]

        # Politique rapportée : masse sur le meilleur rang (pour information ;
        # elle n'entre dans aucun apprentissage).
        pol = np.zeros(A, dtype=np.float32)
        best = ranks.min()
        tops = [idx[j] for j, r in enumerate(ranks) if r == best and idx[j] < A]
        for i in tops:
            pol[i] = 1.0 / len(tops)
        return [int(c) for c in chosen], pol, 0.0

    return agent_fn


def make_baseline_agent(kind: str, cfg, epsilon: float = 0.0, seed: Optional[int] = None):
    if kind == "random":
        return make_random_agent(cfg, seed=seed)
    if kind == "greedy":
        return make_greedy_agent(cfg, epsilon=epsilon, seed=seed)
    raise ValueError(f"Adversaire de référence inconnu : {kind!r} (attendu: random, greedy)")
