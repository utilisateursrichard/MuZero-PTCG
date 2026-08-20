#!/usr/bin/env python
"""
ptcg_muzero/diag_prize.py
==========================
Diagnostic : comment le moteur cabt représente-t-il les prizes ?

Pourquoi
--------
``env/encoding.py`` calcule le nombre de prizes restants avec
``len(player["prize"])`` et l'expose au réseau via ``global_feat[6]`` et
``global_feat[7]``.  ``interpretability/probes.py`` fait le même calcul pour la
sonde ``prize_lead``.

Or cette sonde atteint 90–100 % de précision, ce qui n'arrive que si sa cible
est constante : le compteur ne bouge donc jamais de la partie.  Au Pokémon TCG
on gagne en prenant 6 prizes — c'est la variable d'état la plus importante du
jeu, et le réseau ne la voit pas.

Ce script joue UNE partie en choisissant toujours la première option légale
(aucun réseau, aucun JAX) et affiche l'évolution réelle de la structure des
prizes, plus toutes les clés du dict joueur susceptibles de contenir un
compteur exploitable.

Usage (sur Kaggle, là où le moteur est disponible) ::

    python diag_prize.py
    python diag_prize.py --max-steps 400
"""
from __future__ import annotations

import argparse
import sys

# Clés candidates pour un compteur scalaire de prizes.
CANDIDATE_KEYS = (
    "prizeCount", "prizeRemaining", "remainingPrize", "prizesLeft", "prizeLeft",
    "prize_count", "prizeNum", "prizeRemain", "prizes", "prizeTaken", "prizeCards",
)


def _describe_prize(player) -> dict:
    """Résume tout ce qui ressemble à une information de prize chez un joueur."""
    out = {}
    if isinstance(player, dict):
        items = player.items()
        get = player.get
    else:
        items = [(k, getattr(player, k, None)) for k in dir(player) if not k.startswith("_")]
        get = lambda k, d=None: getattr(player, k, d)  # noqa: E731

    prize = get("prize")
    if isinstance(prize, (list, tuple)):
        out["prize_len"] = len(prize)
        out["prize_non_none"] = sum(1 for p in prize if p is not None)
        out["prize_repr"] = repr(prize)[:160]
    elif prize is not None:
        out["prize_scalar"] = repr(prize)[:80]

    # Toute clé contenant "prize", quelle qu'elle soit.
    for k, v in items:
        if "prize" in str(k).lower() and k != "prize":
            out[f"key:{k}"] = repr(v)[:80]
    for k in CANDIDATE_KEYS:
        v = get(k)
        if v is not None and f"key:{k}" not in out and k != "prize":
            out[f"cand:{k}"] = repr(v)[:80]
    return out


def choose_indices(select) -> list:
    """Sélection valide respectant minCount / maxCount.

    La version initiale n'envoyait qu'un seul indice : dès qu'un ``select``
    exigeait ``minCount >= 2`` le moteur levait ``IndexError``.
    """
    if select is None:
        return []
    get = select.get if isinstance(select, dict) else (lambda k, d=None: getattr(select, k, d))
    options = get("option", []) or get("options", []) or []
    valid = [i for i, o in enumerate(options) if o is not None]
    if not valid:
        return []

    def _int(v, d):
        try:
            return int(v)
        except (TypeError, ValueError):
            return d

    lo = max(1, _int(get("minCount", 1), 1))
    hi = max(lo, _int(get("maxCount", lo), lo))
    k = min(max(lo, 1), hi, len(valid))
    return valid[:k]


def _log_type(log) -> int:
    v = log.get("type") if isinstance(log, dict) else getattr(log, "type", None)
    if hasattr(v, "value"):
        v = v.value
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description="Prize representation diagnostics.")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=25, help="Snapshot frequency.")
    args = ap.parse_args()

    from config import Config
    from env.wrapper import CabtEnv, DeckError
    from models.deck_builder import DEFAULT_COMPETITIVE_DECK

    cfg = Config()
    deck = list(DEFAULT_COMPETITIVE_DECK)
    env = CabtEnv()

    try:
        obs, done = env.reset(deck, deck)
    except DeckError as e:
        print(f"ERROR: engine rejected deck — {e}")
        sys.exit(1)

    print("=" * 78)
    print("PRIZE DIAGNOSTICS — 1 game, first legal option at each step")
    print("=" * 78)

    # Inventaire complet des clés du dict joueur, une seule fois.
    players = (obs.get("current") or {}).get("players") or []
    if players:
        p0 = players[0]
        keys = sorted(p0.keys()) if isinstance(p0, dict) else \
            sorted(k for k in dir(p0) if not k.startswith("_"))
        print("\nAvailable keys on a player:")
        print("  " + ", ".join(str(k) for k in keys))

    from collections import Counter
    snaps = []
    log_types: Counter = Counter()
    log_samples: dict = {}
    seen_logs = 0
    step = 0
    while not done and step < args.max_steps:
        step += 1
        cur = obs.get("current") or {}
        your_idx = cur.get("yourIndex", 0)
        select = obs.get("select")

        # Recensement des logs : sans compteur scalaire de prizes chez le joueur,
        # c'est ici que doit se trouver l'événement « prize prise ».
        logs = obs.get("logs") or []
        for lg in logs[seen_logs:]:
            t = _log_type(lg)
            log_types[t] += 1
            log_samples.setdefault(t, repr(lg)[:200])
        seen_logs = max(seen_logs, len(logs))

        if step == 1 or step % args.every == 0:
            pl = cur.get("players") or [{}, {}]
            rec = {"step": step, "turn": cur.get("turn")}
            for i, p in enumerate(pl[:2]):
                rec[f"p{i}"] = _describe_prize(p)
            snaps.append(rec)

        if select is None:
            obs, done = env.step(list(deck))
            continue

        picks = choose_indices(select)
        try:
            obs, done = env.step(picks)
        except Exception as exc:
            # Un select refusé ne doit pas interrompre le diagnostic : on trace
            # le contexte exact et on tente des repositionnements simples.
            print(f"\n[warning] step {step}: engine rejected {picks} "
                  f"({type(exc).__name__}: {exc})")
            print(f"    select = minCount={select.get('minCount')} "
                  f"maxCount={select.get('maxCount')} type={select.get('type')} "
                  f"n_options={len(select.get('option') or [])}")
            recovered = False
            for alt in ([], [0], picks[:1]):
                try:
                    obs, done = env.step(alt)
                    print(f"    → fallback to {alt}")
                    recovered = True
                    break
                except Exception:
                    continue
            if not recovered:
                print("    → unable to proceed, stopping game.")
                break

    # Relevé final
    cur = obs.get("current") or {}
    pl = cur.get("players") or [{}, {}]
    rec = {"step": step, "turn": cur.get("turn")}
    for i, p in enumerate(pl[:2]):
        rec[f"p{i}"] = _describe_prize(p)
    snaps.append(rec)
    env.close()

    print(f"\nGame completed in {step} engine steps — result = {cur.get('result')}\n")
    for s in snaps:
        print(f"— step {s['step']:4d} (turn={s['turn']})")
        for side in ("p0", "p1"):
            if side in s:
                fields = ", ".join(f"{k}={v}" for k, v in s[side].items())
                print(f"    {side}: {fields}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for side in ("p0", "p1"):
        vals = [s[side].get("prize_len") for s in snaps if side in s]
        vals = [v for v in vals if v is not None]
        nn = [s[side].get("prize_non_none") for s in snaps if side in s]
        nn = [v for v in nn if v is not None]
        if vals:
            verdict = "CONSTANT → unusable" if len(set(vals)) == 1 else "VARIES → usable"
            print(f"  {side} len(prize)      : {vals}  → {verdict}")
        if nn:
            verdict = "CONSTANT → unusable" if len(set(nn)) == 1 else "VARIES → usable"
            print(f"  {side} non-None        : {nn}  → {verdict}")
        for key in sorted({k for s in snaps if side in s for k in s[side]
                           if k.startswith(("key:", "cand:"))}):
            seq = [s[side].get(key) for s in snaps if side in s]
            verdict = "constant" if len(set(map(str, seq))) == 1 else "VARIES → CANDIDATE"
            print(f"  {side} {key:22s}: {seq[0]} … {seq[-1]}  → {verdict}")
    print("\nLook for a 'VARIES' line: this is the field to use in")
    print("env/encoding.py (global_feat[6]/[7]) and interpretability/probes.py (prize_lead).")

    # ── Logs : la piste de repli si aucun champ ne varie ──────────────────────
    print("\n" + "=" * 78)
    print("LOG TYPES ENCOUNTERED  (look for 'prize taken' event)")
    print("=" * 78)
    if not log_types:
        print("  No logs captured.")
    for t, count in sorted(log_types.items(), key=lambda kv: -kv[1]):
        note = "  ← RESULT (game over)" if t == 23 else ""
        print(f"  type={t:3d}  ×{count:<5d}{note}")
        print(f"      example : {log_samples.get(t)}")
    print("\nIf len(prize) is constant, the prize counter must be reconstructed")
    print("by counting occurrences of the log corresponding to a prize taken.")


if __name__ == "__main__":
    main()
