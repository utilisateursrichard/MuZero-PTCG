"""
PTCG MuZero — agent de soumission Kaggle.
Générée automatiquement par main.py submit.
"""
import glob, sys
# Replicate the exact path setup from the reference Kaggle notebook:
#   sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0])
_cg_hits = glob.glob('/kaggle/input/**/cg-lib', recursive=True)
if _cg_hits:
    sys.path.append(_cg_hits[0])

HF_REPO = "richard151111/muZero"

def _load():
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    import json, jax, jax.numpy as jnp
    from export.hub import get_hf_token

    token = get_hf_token()

    mz_path  = hf_hub_download(HF_REPO, "muzero.safetensors", token=token)
    cfg_path = hf_hub_download(HF_REPO, "config.json", token=token)
    dk_path  = hf_hub_download(HF_REPO, "deck_builder.safetensors", token=token)

    from export.hub import _unflatten_params
    from config import Config
    cfg = Config.from_json(open(cfg_path).read())

    mz_params  = _unflatten_params(load_file(mz_path))
    dk_params  = _unflatten_params(load_file(dk_path))
    return mz_params, dk_params, cfg

_mz_params, _dk_params, _cfg = _load()

from cards.encoder import CardStaticFeatures
from models.networks import MuZeroNetwork
from models.deck_builder import DeckBuilderNetwork, sample_deck, set_basic_pokemon_ids, set_energy_ids
import jax, jax.numpy as jnp

_card_data = CardStaticFeatures("/kaggle/input/cards.csv")
_n = max(_card_data.max_card_id + 1, _cfg.model.num_card_ids)
_cfg.model.num_card_ids = _n
_static = jnp.array(_card_data.feature_matrix(_n))
_energy_ids = [c for c in _card_data.card_ids
               if "Energy" in _card_data._cards[c].get("stage","")]
set_energy_ids(_energy_ids)
set_basic_pokemon_ids([
    cid for cid in _card_data.card_ids
    if _card_data._cards[cid].get("stage", "").strip().lower()
    in ("basic pokémon", "basic pokemon")
])

_network  = MuZeroNetwork(cfg=_cfg.model, static_features=_static)
_deck_net = DeckBuilderNetwork(cfg=_cfg.model, static_features=_static)

_rng = jax.random.PRNGKey(42)
_logits, _ = _deck_net.apply(_dk_params)
_deck, _   = sample_deck(_logits[0], _rng, _n, _energy_ids, temperature=0.1)

from cg.game import battle_start, battle_select, battle_finish
from search.ismcts import ismcts_action
from env.encoding import encode_observation

def agent(obs_dict):
    global _rng
    _rng, rng_act = jax.random.split(_rng)
    select  = obs_dict.get("select") or {}
    options = select.get("option", [])
    if not options:
        return []
    enc = encode_observation(obs_dict, obs_dict["current"]["yourIndex"], _cfg.model)
    mask = enc["option_mask"]
    best, _, _ = ismcts_action(_network, _mz_params, enc, mask, rng_act, _cfg)
    max_cnt = int(select.get("maxCount", 1))
    import numpy as np
    if max_cnt > 1:
        return sorted(np.argsort(-np.where(mask, enc["option_mask"].astype(float), -1e9))[:max_cnt].tolist())
    return [best]

def deck_builder():
    return _deck
