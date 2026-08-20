"""
ptcg_muzero/evaluation.py
==========================
Métriques de test d'un modèle, calculées à partir des ``GameHistory``.

Toutes les mesures ici sont *objectives* : elles ne dépendent d'aucune
hyper-paramétrisation de l'entraînement et restent comparables entre runs.

Trois familles
--------------
1. **Résultat**   — victoire / nul / défaite contre l'adversaire de référence.
2. **Compétence** — proxies indépendants de l'adversaire : prizes prises,
   attaques jouées quand une attaque était légale, taux de passe, deck-out.
   Un agent qui ne KO jamais rien prend 0 prize, quel que soit l'adversaire.
3. **Calibration de la valeur** — corrélation entre ``v(s)`` prédit au pas *t*
   et l'issue réelle de la partie.  Ne nécessite aucun adversaire, et teste
   directement ce sur quoi MCTS s'appuie pour trier ses branches : si la valeur
   ne prédit pas l'issue, la recherche n'a rien pour travailler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Indices dans global_feat (cf. env/encoding.py : GLOBAL_FEAT_DIM = 12)
G_MY_PRIZES, G_OPP_PRIZES = 6, 7
G_MY_DECK = 8

# Indices dans option_feat (cf. env/encoding.py : OPTION_FEAT_DIM = 50)
OPT_TYPE_SLICE = slice(0, 17)
OPT_IS_ATTACK = 45

TYPE_NAMES = {
    3: "card", 4: "tool", 5: "energycard", 6: "energy", 7: "play",
    8: "attach", 9: "evolve", 10: "ability", 11: "discard",
    12: "retreat", 13: "attack", 14: "end",
}


@dataclass
class GameStats:
    won: object = None            # True / False / None (nul)
    decisions: int = 0
    prizes_taken: float = 0.0
    prizes_conceded: float = 0.0
    attack_available: int = 0
    attack_played: int = 0
    decked_out: bool = False
    action_counts: Dict[str, int] = field(default_factory=dict)
    values: List[float] = field(default_factory=list)
    outcome: float = 0.0          # +1 victoire, -1 défaite, 0 nul


def analyse_history(hist) -> GameStats:
    """Extrait les métriques d'une trajectoire (perspective d'un seul joueur)."""
    st = GameStats()
    st.won = hist.game_won
    st.outcome = 1.0 if hist.game_won is True else (-1.0 if hist.game_won is False else 0.0)
    st.decisions = len(hist.actions)
    if st.decisions == 0:
        return st

    obs = hist.observations
    g_last = obs[-1]["global_feat"]

    # ── Prizes prises = pic observé − valeur finale ──────────────────────────
    # PAS « première valeur − dernière ». Avant la distribution (turn 0) le
    # moteur renvoie `prize == []`, donc le compteur vaut 0, monte à 6 une fois
    # les prizes distribuées, puis décroît. Prendre la PREMIÈRE observation comme
    # référence donnait donc une différence négative, que le `max(0.0, …)`
    # écrasait silencieusement à 0.00 — d'où « 0 prize des deux côtés » alors que
    # le compteur fonctionne parfaitement.
    def _taken(idx: int) -> float:
        seq = [float(o["global_feat"][idx]) * 6.0 for o in obs]
        seq = [v for v in seq if v > 0.0]      # ignorer l'avant-distribution
        if not seq:
            return 0.0
        return max(0.0, max(seq) - seq[-1])

    st.prizes_taken = _taken(G_MY_PRIZES)
    st.prizes_conceded = _taken(G_OPP_PRIZES)
    st.decked_out = bool(float(g_last[G_MY_DECK]) <= 0.0)

    # ── Attaques comptées PAR TOUR, pas par décision ─────────────────────────
    # Attaquer termine le tour : il est normal d'attacher, évoluer et jouer des
    # cartes avant. Un ratio par décision sous-estime donc mécaniquement l'agent
    # (une seule attaque possible pour ~7 décisions). On regroupe les pas par
    # numéro de tour, lisible dans global_feat[0] = min(turn,200)/200.
    turns_avail: set = set()
    turns_attacked: set = set()

    for o, act in zip(obs, hist.actions):
        mask = np.asarray(o["option_mask"]).astype(bool)
        feat = np.asarray(o["option_feat"])
        turn_id = int(round(float(o["global_feat"][0]) * 200.0))
        if mask.any() and float(np.max(feat[mask, OPT_IS_ATTACK])) > 0.5:
            turns_avail.add(turn_id)
        chosen = np.where(np.asarray(act) > 0.5)[0]
        for i in chosen:
            if i >= len(mask) or not mask[i]:
                continue
            row = feat[i, OPT_TYPE_SLICE]
            t = int(np.argmax(row)) if float(np.max(row)) > 0.5 else -1
            name = TYPE_NAMES.get(t, "other")
            st.action_counts[name] = st.action_counts.get(name, 0) + 1
            if t == 13:
                turns_attacked.add(turn_id)

    st.attack_available = len(turns_avail)
    st.attack_played = len(turns_attacked & turns_avail)
    st.values = [float(v) for v in hist.search_vals]
    return st


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """Intervalle de confiance de Wilson pour une proportion.

    Indispensable ici : sur 20 parties, 55 % de victoires donne [34 %, 74 %] —
    indistinguable de 50 % comme de 70 %. Sans cet intervalle on lit du bruit
    comme un résultat.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(stats: List[GameStats]) -> dict:
    """Agrège les statistiques de N parties en un rapport."""
    n = len(stats)
    if n == 0:
        return {"games": 0}

    wins = sum(1 for s in stats if s.won is True)
    losses = sum(1 for s in stats if s.won is False)
    draws = n - wins - losses

    total_actions = sum(sum(s.action_counts.values()) for s in stats)
    act_pct = {}
    if total_actions:
        merged: Dict[str, int] = {}
        for s in stats:
            for k, v in s.action_counts.items():
                merged[k] = merged.get(k, 0) + v
        act_pct = {k: round(100.0 * v / total_actions, 1)
                   for k, v in sorted(merged.items(), key=lambda kv: -kv[1])}

    avail = sum(s.attack_available for s in stats)
    played = sum(s.attack_played for s in stats)

    # ── Calibration : v(s) prédit vs issue réelle ────────────────────────────
    vs, ys = [], []
    for s in stats:
        vs.extend(s.values)
        ys.extend([s.outcome] * len(s.values))
    vs_a, ys_a = np.asarray(vs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    calib = {}
    if vs_a.size >= 2 and np.std(vs_a) > 1e-9 and np.std(ys_a) > 1e-9:
        calib["value_outcome_pearson_r"] = round(float(np.corrcoef(vs_a, ys_a)[0, 1]), 4)
    else:
        calib["value_outcome_pearson_r"] = None
    if vs_a.size:
        nz = ys_a != 0.0
        calib["value_sign_accuracy_pct"] = (
            round(100.0 * float(np.mean(np.sign(vs_a[nz]) == np.sign(ys_a[nz]))), 1)
            if nz.any() else None
        )
        calib["value_mean"] = round(float(np.mean(vs_a)), 4)
        calib["value_abs_mean"] = round(float(np.mean(np.abs(vs_a))), 4)

    lo, hi = wilson_interval(wins, n)
    return {
        "games": n,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate_pct": round(100.0 * wins / n, 1),
        "win_rate_ci95_pct": [round(100.0 * lo, 1), round(100.0 * hi, 1)],
        "conclusive": bool(lo > 0.5 or hi < 0.5),
        "score_pct": round(100.0 * (wins + 0.5 * draws) / n, 1),
        "avg_decisions_per_game": round(float(np.mean([s.decisions for s in stats])), 1),
        "avg_prizes_taken": round(float(np.mean([s.prizes_taken for s in stats])), 2),
        "avg_prizes_conceded": round(float(np.mean([s.prizes_conceded for s in stats])), 2),
        "attack_taken_when_available_pct": (
            round(100.0 * played / avail, 1) if avail else None
        ),
        "decked_out_pct": round(100.0 * sum(1 for s in stats if s.decked_out) / n, 1),
        "action_dist_pct": act_pct,
        **calib,
    }


def _games_needed(p: float, z: float = 1.96) -> int:
    """Parties nécessaires pour que l'intervalle EXCLUE 50 %, au taux observé ``p``.

    C'est bien la question posée par « combien de parties pour trancher ? », et
    non la demi-largeur de l'intervalle. Le coût explose quand on s'approche de
    50 % : un écart de 5 points demande ~380 parties, un écart de 20 points en
    demande ~21. D'où l'intérêt de la molette ``--epsilon`` : viser un
    adversaire nettement plus faible ou plus fort rend la mesure bien moins
    chère que de chipoter autour de la parité.

    Le résultat est cherché numériquement AVEC ``wilson_interval`` : la formule
    normale usuelle z^2*p(1-p)/(p-0.5)^2 tombe pile sur la frontière et échoue
    au test affiché juste au-dessus, ce qui donnerait un conseil faux.
    """
    p = min(max(p, 0.01), 0.99)
    if abs(p - 0.5) < 0.01:
        return 100_000        # à la parité stricte, aucun échantillon ne tranche

    def decides(n: int) -> bool:
        lo, hi = wilson_interval(int(round(p * n)), n, z=z)
        return lo > 0.5 or hi < 0.5

    n = 8
    while n < 100_000 and not decides(n):
        n *= 2
    if n >= 100_000:
        return 100_000
    lo_n, hi_n = n // 2, n
    while lo_n + 1 < hi_n:      # plus petit n qui tranche
        mid = (lo_n + hi_n) // 2
        if decides(mid):
            hi_n = mid
        else:
            lo_n = mid
    return hi_n


def format_report(title: str, agg: dict) -> str:
    """Formatted human-readable test report."""
    if agg.get("games", 0) == 0:
        return f"── {title} ──\n  no games played."
    L = [f"── {title} ──"]
    L.append(f"  Games              : {agg['games']}  "
             f"(W {agg['wins']} / D {agg['draws']} / L {agg['losses']})")
    ci = agg["win_rate_ci95_pct"]
    L.append(f"  Win Rate           : {agg['win_rate_pct']:.1f} %   "
             f"95% CI [{ci[0]:.0f} %, {ci[1]:.0f} %]   Score {agg['score_pct']:.1f} %")
    if not agg["conclusive"]:
        L.append(f"  ⚠ INCONCLUSIVE — CI contains 50%. "
                 f"Requires ~{_games_needed(agg['win_rate_pct'] / 100.0)} games to establish significance.")
    L.append(f"  Decisions/game     : {agg['avg_decisions_per_game']}")
    L.append("")
    L.append("  ── Competence (Opponent-Independent) ──")
    L.append(f"  Prizes taken       : {agg['avg_prizes_taken']:.2f} / 6")
    L.append(f"  Prizes conceded    : {agg['avg_prizes_conceded']:.2f} / 6")
    if agg["avg_prizes_taken"] == 0.0 and agg["avg_prizes_conceded"] == 0.0:
        L.append("    ⚠ 0 prizes on BOTH sides: prize counter likely constant "
                 "(see diag_prize.py).")
    atk = agg["attack_taken_when_available_pct"]
    L.append(f"  Turns ended by atk : {'n/a' if atk is None else f'{atk:.1f} %'}"
             "   (on turns where attacking was legal)")
    L.append(f"  Deck-out           : {agg['decked_out_pct']:.1f} %")
    if agg["action_dist_pct"]:
        top = ", ".join(f"{k}={v}%" for k, v in list(agg["action_dist_pct"].items())[:8])
        L.append(f"  Distribution       : {top}")
    L.append("")
    L.append("  ── Value Head Calibration (Ground Truth) ──")
    r = agg["value_outcome_pearson_r"]
    L.append(f"  corr(v(s), outcome): {'n/a' if r is None else f'{r:+.4f}'}"
             "    (0 = value predicts nothing)")
    sa = agg.get("value_sign_accuracy_pct")
    L.append(f"  Sign accuracy      : {'n/a' if sa is None else f'{sa:.1f} %'}"
             "    (50% = chance)")
    L.append(f"  Mean v / Mean |v|  : {agg.get('value_mean')} / {agg.get('value_abs_mean')}"
             "   (mean v should gravitate towards 0 in symmetric games)")
    return "\n".join(L)
