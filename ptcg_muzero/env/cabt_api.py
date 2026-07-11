"""
ptcg_muzero/env/cabt_api.py
============================
Re-export de cg.api et cg.game.
Setup identique au script de référence.
"""
import glob
import os
import sys

# Cherche le parent du dossier 'cg' (ou 'cg-lib') dans /kaggle/input
_cg_parent = None

# 1. Chemin connu de la compétition Kaggle PTCG
_competition_path = (
    '/kaggle/input/competitions/pokemon-tcg-ai-battle'
    '/sample_submission/sample_submission'
)
if os.path.isdir(os.path.join(_competition_path, 'cg')):
    _cg_parent = _competition_path

# 2. Fallback : glob sur 'cg-lib' (ancien nom)
if _cg_parent is None:
    _hits = glob.glob('/kaggle/input/**/cg-lib', recursive=True)
    if _hits:
        _cg_parent = _hits[0]

# 3. Fallback : os.walk sur 'cg-lib' (mount points non traversés par glob)
if _cg_parent is None:
    for _root, _dirs, _ in os.walk('/kaggle/input'):
        if 'cg-lib' in _dirs:
            _cg_parent = os.path.join(_root, 'cg-lib')
            break

# 4. Fallback : os.walk sur le dossier 'cg' lui-même
if _cg_parent is None:
    for _root, _dirs, _ in os.walk('/kaggle/input'):
        if 'cg' in _dirs:
            _cg_parent = _root
            break

# 5. Fallback local pour le développement et test hors Kaggle
if _cg_parent is None:
    from pathlib import Path
    workspace_root = Path(__file__).parent.parent.parent
    local_paths = [
        workspace_root / "competiton" / "sample_submission" / "sample_submission",
        workspace_root / "competiton" / "sample_submission",
        workspace_root / "competiton",
    ]
    for lp in local_paths:
        if (lp / "cg").is_dir():
            _cg_parent = str(lp.resolve())
            break

if _cg_parent is None:
    raise RuntimeError(
        "Impossible de localiser la bibliothèque 'cg' dans /kaggle/input ni dans les chemins locaux du workspace. "
        "Vérifiez que le dataset de la compétition est bien attaché au notebook ou disponible dans le dossier 'competiton'."
    )

sys.path.append(_cg_parent)

from cg.api import (   # type: ignore[import]
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SearchState,
    SelectContext,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)
from cg.game import (  # type: ignore[import]
    battle_finish,
    battle_start,
    battle_select,
)

__all__ = [
    "AreaType", "Card", "Observation", "OptionType", "PlayerState",
    "Pokemon", "SearchState", "SelectContext", "all_attack", "all_card_data",
    "battle_finish", "battle_select", "battle_start",
    "search_begin", "search_end", "search_step", "to_observation_class",
]
