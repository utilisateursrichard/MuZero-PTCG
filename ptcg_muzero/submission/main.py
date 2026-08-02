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


def read_deck_csv() -> list[int]:
    """Lecture du fichier deck.csv (60 IDs de cartes)."""
    file_path = _resolve("deck.csv")
    with open(file_path, "r", encoding="utf-8") as f:
        csv = f.read().split("\n")
    deck = []
    for i in range(60):
        if csv[i].strip():
            deck.append(int(csv[i].strip()))
    return deck


_INITIALIZED = False
_mz_params = None
_cfg = None
_network = None
_rng = None


def _find_cards_csv() -> str:
    candidates = [
        "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv",
        "/kaggle/input/cards.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/input/**/EN_Card_Data.csv", recursive=True)
    if hits:
        return hits[0]
    return ""


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

    _cfg = Config.from_json(open(cfg_path, encoding="utf-8").read()) if os.path.exists(cfg_path) else Config()
    _mz_params = _unflatten_params(load_file(mz_path)) if os.path.exists(mz_path) else {}

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

    # Warmup JIT-compilation pour que la 1ère action soit quasi-instantanée
    try:
        dummy_enc = encode_observation({}, 0, _cfg.model)
        dummy_mask = dummy_enc["option_mask"]
        _rng, rng_warmup = jax.random.split(_rng)
        ismcts_action(_network, _mz_params, dummy_enc, dummy_mask, rng_warmup, _cfg)
    except Exception:
        pass

    _INITIALIZED = True


def agent(obs_dict) -> list[int]:
    _init_agent()

    # Extraction sécurisée de select
    if isinstance(obs_dict, dict):
        select = obs_dict.get("select")
    else:
        select = getattr(obs_dict, "select", None)

    # Étape 1 : Demande du deck initial (select est None)
    if select is None:
        return read_deck_csv()

    # Étape 2+ : Actions de jeu
    obs: Observation = to_observation_class(obs_dict) if isinstance(obs_dict, dict) else obs_dict

    import jax
    import numpy as np
    from env.encoding import encode_observation
    from search.ismcts import ismcts_action

    global _rng
    _rng, rng_act = jax.random.split(_rng)

    your_idx = obs.current.yourIndex if (obs.current is not None and hasattr(obs.current, "yourIndex")) else 0
    enc = encode_observation(obs_dict if isinstance(obs_dict, dict) else obs, your_idx, _cfg.model)
    mask = enc["option_mask"]
    best, avg_policy, _ = ismcts_action(_network, _mz_params, enc, mask, rng_act, _cfg)

    max_cnt = obs.select.maxCount if hasattr(obs.select, "maxCount") else 1
    if max_cnt > 1:
        scores = np.where(mask, np.asarray(avg_policy), -1e9)
        sorted_idx = sorted(np.argsort(-scores)[:max_cnt].tolist())
        return sorted_idx
    return [best]
