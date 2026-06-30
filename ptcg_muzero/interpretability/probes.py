"""
ptcg_muzero/interpretability/probes.py
========================================
Linear Probing Classifiers — interprétabilité mécanique.

Principe
--------
On détache le gradient de l'état latent (``jax.lax.stop_gradient``) et on
entraîne 5 sondes linéaires indépendantes pour prédire des concepts critiques
du jeu directement depuis z.  Si une sonde atteint une précision élevée,
cela prouve que le concept est *linéairement encodé* dans l'espace latent.

Les 5 sondes
------------
  0. ``active_in_ko_range``   — Mon Pokémon actif peut-il être KO en 1 attaque ?
                                 cible : bool  (binaire)
  1. ``type_advantage``        — Mon type bat-il le type actif adverse ?
                                 cible : {-1, 0, +1} → 3 classes
  2. ``prize_lead``            — Suis-je en avance sur les prizes ?
                                 cible : {behind, even, ahead} → 3 classes
  3. ``hand_advantage``        — Ai-je ≥ 2 cartes de plus que l'adversaire ?
                                 cible : bool
  4. ``opp_energy_ready``      — L'actif adverse a-t-il assez d'énergie pour attaquer ?
                                 cible : bool

Toutes les sondes partagent la même structure : une seule Dense(num_classes).
Leur gradient ne remonte PAS dans le réseau principal (stop_gradient).

API principale
--------------
  ProbeHeads  – Flax module contenant les 5 têtes linéaires
  extract_probe_targets  – extrait les cibles depuis l'obs numpy
  probe_loss  – calcule la cross-entropie pour toutes les sondes
  probe_accuracy – métriques de précision (pas de gradient)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from config import ModelConfig
from env.encoding import GLOBAL_FEAT_DIM

# ─────────────────────────────────────────────────────────────────────────────
# Définition des 5 sondes
# ─────────────────────────────────────────────────────────────────────────────
PROBE_DEFS: List[Dict] = [
    {"name": "active_in_ko_range",  "num_classes": 2, "idx": 0},
    {"name": "type_advantage",      "num_classes": 3, "idx": 1},
    {"name": "prize_lead",          "num_classes": 3, "idx": 2},
    {"name": "hand_advantage",      "num_classes": 2, "idx": 3},
    {"name": "opp_energy_ready",    "num_classes": 2, "idx": 4},
]
NUM_PROBES      = len(PROBE_DEFS)
MAX_PROBE_CLASS = max(d["num_classes"] for d in PROBE_DEFS)


# ─────────────────────────────────────────────────────────────────────────────
# Flax module
# ─────────────────────────────────────────────────────────────────────────────
class ProbeHeads(nn.Module):
    """
    Cinq têtes linéaires indépendantes branchées sur stop_gradient(z).

    Les poids de ces sondes font PARTIE du paramètre tree du MuZeroNetwork
    (dans un sous-dict ``probes``) mais sont aussi disponibles en standalone.
    """
    cfg: ModelConfig

    @nn.compact
    def __call__(
        self, z: jnp.ndarray
    ) -> List[jnp.ndarray]:
        """
        Args:
            z: [B, latent_dim]  — latent state (gradient stopped internally)
        Returns:
            list of 5 logit tensors, each [B, num_classes_i]
        """
        z_sg = jax.lax.stop_gradient(z)   # ← gradient barrier

        logits_list = []
        for pdef in PROBE_DEFS:
            h = nn.Dense(
                pdef["num_classes"],
                name=f"probe_{pdef['name']}",
                use_bias=True,
            )(z_sg)
            logits_list.append(h)
        return logits_list


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des cibles depuis l'observation
# ─────────────────────────────────────────────────────────────────────────────
def extract_probe_targets(
    obs: dict,
    my_idx: int,
) -> np.ndarray:
    """
    Extrait les 5 cibles entières depuis un obs non-batché (numpy).

    Returns:
        int32 array [NUM_PROBES] — classe cible pour chaque sonde.
        -1  signifie « pas de cible disponible ce step ».
    """
    targets = np.full(NUM_PROBES, -1, dtype=np.int32)

    current = obs.get("current") or {}
    players = current.get("players", [{}, {}])
    opp_idx = 1 - my_idx
    me  = players[my_idx] if my_idx < len(players) else {}
    opp = players[opp_idx] if opp_idx < len(players) else {}

    # ── Sonde 0 : active_in_ko_range ─────────────────────────────────────
    # Mon actif peut-il être KO en 1 attaque adverse ?
    my_active_list = me.get("active") or []
    opp_active_list = opp.get("active") or []
    if my_active_list and opp_active_list:
        my_poke = my_active_list[0]
        opp_poke = opp_active_list[0]
        if my_poke is not None and opp_poke is not None:
            my_hp    = _safe_int(my_poke, "hp", 100)
            max_dmg  = _estimate_max_damage(opp_poke)
            targets[0] = int(max_dmg >= my_hp)

    # ── Sonde 1 : type_advantage ─────────────────────────────────────────
    # -1 = désavantage, 0 = neutre, 1 = avantage
    if my_active_list and opp_active_list:
        my_poke  = my_active_list[0]
        opp_poke = opp_active_list[0]
        if my_poke is not None and opp_poke is not None:
            my_type   = _safe_int(my_poke, "type", -1)
            opp_type  = _safe_int(opp_poke, "type", -1)
            opp_weak  = _safe_int(opp_poke, "weakness", -1)
            opp_res   = _safe_int(opp_poke, "resistance", -1)
            if my_type >= 0:
                if my_type == opp_weak:
                    targets[1] = 2    # avantage
                elif my_type == opp_res:
                    targets[1] = 0    # désavantage
                else:
                    targets[1] = 1    # neutre

    # ── Sonde 2 : prize_lead ─────────────────────────────────────────────
    my_prizes  = len(me.get("prize") or [])
    opp_prizes = len(opp.get("prize") or [])
    # Plus de prizes restants = plus à prendre = EN RETARD
    if my_prizes < opp_prizes:
        targets[2] = 2    # ahead (j'ai moins de prizes restants)
    elif my_prizes == opp_prizes:
        targets[2] = 1    # even
    else:
        targets[2] = 0    # behind

    # ── Sonde 3 : hand_advantage ─────────────────────────────────────────
    my_hand  = me.get("handCount", 0) or len(me.get("hand") or [])
    opp_hand = opp.get("handCount", 0)
    targets[3] = int(my_hand >= opp_hand + 2)

    # ── Sonde 4 : opp_energy_ready ───────────────────────────────────────
    # L'actif adverse a-t-il assez d'énergie pour attaquer (≥ 2 énergies) ?
    if opp_active_list and opp_active_list[0] is not None:
        opp_poke = opp_active_list[0]
        energies = opp_poke.get("energies") or []
        targets[4] = int(len(energies) >= 2)

    return targets


def _safe_int(obj, key, default):
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


def _estimate_max_damage(poke: dict) -> int:
    """Heuristique simple : max dégât parmi les mouvements de l'actif adverse."""
    # Le game state n'expose pas les dégâts directement ;
    # on utilise l'hp_ratio comme proxy inverse
    hp_total = _safe_int(poke, "maxHp", 100)
    # L'adversaire a potentiellement une attaque à ~60% de l'HP de n'importe qui
    # C'est une borne supérieure grossière — à affiner avec les données CSV
    return max(60, hp_total // 2)


# ─────────────────────────────────────────────────────────────────────────────
# Loss et métriques (JIT-ables)
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def probe_loss(
    probe_logits: List[jnp.ndarray],   # 5 × [B, num_classes_i]
    targets: jnp.ndarray,              # [B, NUM_PROBES]  int32, -1 = ignore
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calcule la cross-entropie moyenne sur les 5 sondes.
    Ignore les steps où la cible est -1.

    Returns:
        total_loss  : scalar
        per_probe   : [NUM_PROBES]  loss individuelle (pour le logging)
    """
    losses = []
    for i, (logits, pdef) in enumerate(zip(probe_logits, PROBE_DEFS)):
        tgt = targets[:, i]                          # [B]
        valid = (tgt >= 0)                           # [B] bool
        tgt_clamped = jnp.clip(tgt, 0, pdef["num_classes"] - 1)
        xent = optax_softmax_xent(logits, tgt_clamped)  # [B]
        # Mask invalid
        xent_masked = jnp.where(valid, xent, 0.0)
        n_valid = jnp.maximum(jnp.sum(valid), 1)
        losses.append(jnp.sum(xent_masked) / n_valid)

    per_probe = jnp.array(losses)
    return jnp.mean(per_probe), per_probe


@jax.jit
def probe_accuracy(
    probe_logits: List[jnp.ndarray],
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """
    Précision par sonde.  Returns [NUM_PROBES] float32.
    """
    accs = []
    for i, logits in enumerate(probe_logits):
        tgt   = targets[:, i]
        valid = tgt >= 0
        pred  = jnp.argmax(logits, axis=-1)
        correct = (pred == tgt) & valid
        n_valid = jnp.maximum(jnp.sum(valid), 1)
        accs.append(jnp.sum(correct).astype(jnp.float32) / n_valid)
    return jnp.array(accs)


def optax_softmax_xent(
    logits: jnp.ndarray,   # [B, C]
    labels: jnp.ndarray,   # [B]  int
) -> jnp.ndarray:
    """Softmax cross-entropy per sample [B]."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -log_probs[jnp.arange(logits.shape[0]), labels]


# ─────────────────────────────────────────────────────────────────────────────
# Résumé textuel pour le logging
# ─────────────────────────────────────────────────────────────────────────────
def probe_report(accs: np.ndarray, losses: np.ndarray) -> str:
    lines = ["── Probe Classifiers ─────────────────"]
    for i, pdef in enumerate(PROBE_DEFS):
        lines.append(
            f"  {pdef['name']:30s}  acc={accs[i]:.3f}  loss={losses[i]:.4f}"
        )
    return "\n".join(lines)
