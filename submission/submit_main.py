"""
PTCG MuZero — Kaggle Submission Entrypoint (main.py)
Généré automatiquement par main.py submit.
"""
import glob
import logging
import os
import sys
from pathlib import Path

# Silence noisy JAX backend discovery logs in stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

_AGENT_DIR = "/kaggle_simulations/agent"


def _find_submission_root() -> str:
    """Localise le dossier racine (cards/, muzero.safetensors, …).

    Sur Kaggle, seuls main.py et deck.csv sont copiés dans /kaggle_simulations/agent/.
    Le reste du code est généralement attaché comme dataset sous /kaggle/input/.
    """
    candidates: list[str] = []

    for root in (_AGENT_DIR, os.getcwd(), os.path.abspath(".")):
        if root and os.path.isdir(root):
            candidates.append(root)

    try:
        candidates.append(str(Path(__file__).resolve().parent))
    except (NameError, Exception):
        pass

    for pattern in (
        "/kaggle/input/**/cards/encoder.py",
        "/kaggle/input/**/muzero.safetensors",
        "/kaggle/input/**/config.py",
        "/kaggle/input/**/mctx/__init__.py",
        "/kaggle/input/**/search/ismcts.py",
    ):
        for hit in glob.glob(pattern, recursive=True):
            p = Path(hit).resolve()
            if p.parent.name in ("cards", "mctx"):
                candidates.append(str(p.parent.parent))
            elif p.parent.name == "search":
                candidates.append(str(p.parent.parent))
            else:
                candidates.append(str(p.parent))

    seen: set[str] = set()
    for root in candidates:
        if not root or root in seen:
            continue
        seen.add(root)
        if os.path.isfile(os.path.join(root, "cards", "encoder.py")):
            return root
        if os.path.isfile(os.path.join(root, "mctx", "__init__.py")):
            return root
        if os.path.isfile(os.path.join(root, "muzero.safetensors")):
            return root

    return _AGENT_DIR


_SUBMISSION_ROOT = _find_submission_root()


def _setup_paths() -> None:
    for d in (_SUBMISSION_ROOT, _AGENT_DIR, os.getcwd(), os.path.abspath(".")):
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    for hit in glob.glob("/kaggle/input/**/cg-lib", recursive=True):
        if hit not in sys.path:
            sys.path.append(hit)


_setup_paths()

from cg.api import Observation, to_observation_class


def _resolve(filename: str) -> str:
    """Résout un fichier dans la soumission ou les datasets Kaggle."""
    if os.path.exists(filename):
        return filename

    for d in (_SUBMISSION_ROOT, _AGENT_DIR, os.getcwd(), os.path.abspath(".")):
        alt = os.path.join(d, filename)
        if os.path.exists(alt):
            return alt

    hits = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if hits:
        return hits[0]

    return filename


DEFAULT_DECK: list[int] = [
    96, 96, 96, 96,
    402, 402,
    403, 403,
    404, 404,
    708, 708,
    709, 709,
    710, 710,
    140,
    1071,
    235,
    172,
    173,
    1227, 1227, 1227, 1227,
    1231, 1231,
    1182, 1182,
    1184,
    1201,
    1094, 1094, 1094, 1094,
    1121, 1121, 1121, 1121,
    1152, 1152, 1152,
    1097,
    1116,
    1080,
    1261, 1261, 1261, 1261,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]


def read_deck_csv() -> list[int]:
    """Lecture du fichier deck.csv (60 IDs de cartes)."""
    try:
        file_path = _resolve("deck.csv")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        deck = [int(x) for x in lines[:60]]
        if len(deck) == 60:
            return deck
    except Exception:
        pass
    return list(DEFAULT_DECK)


_INITIALIZED = False
_mz_params = None
_cfg = None
_network = None
_rng = None


def _find_cards_csv() -> str:
    candidates = [
        "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN Card Data.csv",
        "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv",
        "/kaggle/input/cards.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/input/**/EN*Card*Data.csv", recursive=True) + glob.glob("/kaggle/input/**/*Card*Data*.csv", recursive=True)
    if hits:
        return hits[0]
    return ""


def _merge_params(defaults, loaded):
    from collections.abc import Mapping
    import numpy as np
    import jax.numpy as jnp

    if isinstance(defaults, Mapping) and isinstance(loaded, Mapping):
        # AUDIT §1.1 — les clés du checkpoint absentes de l'architecture courante
        # (pi_q / pi_k / a_feat_emb supprimés) sont écartées au lieu d'être
        # réinjectées dans l'arbre de paramètres.
        res = {}
        for k, v in defaults.items():
            if k in loaded:
                res[k] = _merge_params(v, loaded[k])
            else:
                res[k] = v
        return res
    if hasattr(defaults, "shape") and hasattr(loaded, "shape"):
        if defaults.shape != loaded.shape:
            if defaults.ndim == 2 and loaded.ndim == 2 and defaults.shape[1] == loaded.shape[1] and defaults.shape[0] > loaded.shape[0]:
                new_param = np.array(defaults)
                new_param[:loaded.shape[0], :] = np.array(loaded)
                return jnp.array(new_param)
            return defaults
    return loaded


def _init_agent():
    global _INITIALIZED, _mz_params, _cfg, _network, _rng
    if _INITIALIZED:
        return

    _setup_paths()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from safetensors.numpy import load_file
    from cards.encoder import CardStaticFeatures
    from config import Config
    from env.encoding import encode_observation
    from export.hub import _unflatten_params
    from models.networks import MuZeroNetwork
    from search.ismcts import ismcts_action

    cfg_path = _resolve("config.json")
    mz_path = _resolve("muzero.safetensors")

    if not cfg_path or not os.path.exists(cfg_path):
        raise FileNotFoundError(f"FATAL ERROR: Configuration file 'config.json' not found (resolved: {cfg_path}). Silent fallback is not allowed.")
    if not mz_path or not os.path.exists(mz_path):
        raise FileNotFoundError(f"FATAL ERROR: Weight file 'muzero.safetensors' not found (resolved: {mz_path}). Silent fallback to random weights is not allowed.")

    _cfg = Config.from_json(open(cfg_path, encoding="utf-8").read())

    cards_csv = _find_cards_csv()
    if cards_csv and os.path.exists(cards_csv):
        _card_data = CardStaticFeatures(cards_csv)
        _n = max(_card_data.max_card_id + 1, _cfg.model.num_card_ids)
        _cfg.model.num_card_ids = _n
        _static = jnp.array(_card_data.feature_matrix(_n))
    else:
        _n = _cfg.model.num_card_ids
        _static = jnp.zeros((_n, 48), dtype=jnp.float32)

    _network = MuZeroNetwork(cfg=_cfg.model, static_features=_static)
    _rng = jax.random.PRNGKey(42)

    # Initialisation de base pour garantir la conformité des formes de paramètres
    dummy_enc = encode_observation({}, 0, _cfg.model)
    batch_obs = {k: jnp.array(v[None]) for k, v in dummy_enc.items()}
    fresh_params = _network.init(_rng, batch_obs, method=_network.init_all)

    # Chargement direct et adaptation automatique des poids (rétrocompatibilité)
    loaded_raw = _unflatten_params(load_file(mz_path))
    raw_params = loaded_raw.get("muzero", loaded_raw)
    _mz_params = _merge_params(fresh_params, raw_params)

    # AUDIT §2.1 — l'adversaire joue le même deck de référence : la
    # déterminisation ISMCTS doit y puiser, pas dans les 1268 IDs du pool.
    try:
        from search.ismcts import set_belief_deck
        set_belief_deck(read_deck_csv())
    except Exception as exc:
        logging.getLogger("ptcg_muzero.submit").warning(
            "set_belief_deck indisponible (%s) — repli sur le pool complet.", exc
        )

    # Warmup JIT-compilation pour que la 1ère action soit quasi-instantanée.
    # AUDIT §3.9 : ne plus avaler l'exception silencieusement — un échec de
    # compilation ne se manifestait qu'au premier coup joué, en pleine partie.
    try:
        dummy_mask = dummy_enc["option_mask"]
        dummy_mask[0] = True   # mctx exige au moins une action légale
        _rng, rng_warmup = jax.random.split(_rng)
        ismcts_action(_network, _mz_params, dummy_enc, dummy_mask, rng_warmup, _cfg)
    except Exception as exc:
        logging.getLogger("ptcg_muzero.submit").warning(
            "Warmup ISMCTS échoué (%s) — la première décision sera plus lente.", exc
        )

    _INITIALIZED = True


def agent(obs_dict) -> list[int]:
    _init_agent()

    if obs_dict is None:
        return read_deck_csv()

    # Extraction sécurisée de select et current
    if isinstance(obs_dict, dict):
        select = obs_dict.get("select")
        current = obs_dict.get("current") or {}
    else:
        select = getattr(obs_dict, "select", None)
        current = getattr(obs_dict, "current", None) or {}

    # Étape 1 : Demande du deck initial (select est None)
    if select is None:
        return read_deck_csv()

    # Étape 2+ : Actions de jeu
    import jax
    import numpy as np
    from env.encoding import encode_observation, _int_from
    from search.ismcts import ismcts_action

    global _rng
    _rng, rng_act = jax.random.split(_rng)

    your_idx = _int_from(current, "yourIndex", 0)
    enc = encode_observation(obs_dict, your_idx, _cfg.model)
    mask = enc["option_mask"]
    best, avg_policy, _ = ismcts_action(_network, _mz_params, enc, mask, rng_act, _cfg)

    raw_options = select.get("option", []) if isinstance(select, dict) else (getattr(select, "option", []) or getattr(select, "options", []) or [])
    min_cnt = int(select.get("minCount", 1) if isinstance(select, dict) else getattr(select, "minCount", 1))
    max_cnt = int(select.get("maxCount", 1) if isinstance(select, dict) else getattr(select, "maxCount", 1))

    # Filter all valid option indices (not None, within bounds, and legally masked if possible)
    valid_indices = [
        i for i in range(len(raw_options))
        if raw_options[i] is not None and (i < len(mask) and mask[i])
    ]
    if not valid_indices:
        valid_indices = [
            i for i in range(len(raw_options))
            if raw_options[i] is not None
        ]

    # Handle edge case where no options exist or maxCount is 0
    if not valid_indices or max_cnt == 0:
        return []

    # Sort valid options by policy score (highest first)
    scores = np.asarray(avg_policy)
    valid_scores = [float(scores[i]) if i < len(scores) else -1e9 for i in valid_indices]
    ranked_indices = [valid_indices[k] for k in np.argsort(-np.array(valid_scores))]

    # Determine exact count to return: must be within [min_cnt, max_cnt]
    desired_cnt = min(max_cnt, len(ranked_indices))
    desired_cnt = max(desired_cnt, min(min_cnt, len(ranked_indices)))

    if desired_cnt == 0:
        return []

    chosen = ranked_indices[:desired_cnt]
    return [int(x) for x in chosen]

