"""
ptcg_muzero/models/iree_agent.py
==================================
Autonomous agent wrapper using the IREE Runtime (Vulkan / CPU) for ultra-fast,
portable inference of the MuZero policy and value networks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config
from env.encoding import encode_observation, _int_from
from models.iree_engine import IREEMuZeroEngine

logger = logging.getLogger("iree_agent")


class IREEMuZeroAgent:
    """Agent running inference via IREE Vulkan/CPU compiled module (.vmfb)."""

    def __init__(
        self,
        vmfb_path: str | Path = "muzero_vulkan.vmfb",
        cfg: Optional[Config] = None,
        device_uri: str = "vulkan",
    ):
        self.cfg = cfg or Config()
        self.vmfb_path = Path(vmfb_path).resolve()
        
        # Expected input keys and dtypes in the compiled VMFB
        self.input_schema = {
            "global_feat": np.float32,
            "my_active_id": np.int32,
            "my_active_feat": np.float32,
            "my_bench_ids": np.int32,
            "my_bench_feat": np.float32,
            "my_bench_mask": np.float32,
            "my_hand_ids": np.int32,
            "my_hand_mask": np.float32,
            "my_discard_ids": np.int32,
            "my_discard_mask": np.float32,
            "opp_active_id": np.int32,
            "opp_active_feat": np.float32,
            "opp_bench_ids": np.int32,
            "opp_bench_feat": np.float32,
            "opp_bench_mask": np.float32,
            "opp_discard_ids": np.int32,
            "opp_discard_mask": np.float32,
            "opp_hand_ids": np.int32,
            "opp_hand_mask": np.float32,
            "option_ids": np.int32,
            "option_feat": np.float32,
            "option_mask": np.float32,
        }

        self.engine = IREEMuZeroEngine(self.vmfb_path, device_uri=device_uri)
        logger.info("IREEMuZeroAgent ready on device: %s", self.engine.config.device)

    def _prepare_inputs(self, enc_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Converts observation dict to conform to the exact VMFB signature and dtypes."""
        batch_inputs = {}
        for key in sorted(self.input_schema.keys()):
            arr = enc_obs.get(key)
            target_dtype = self.input_schema[key]
            if arr is None:
                # Fallback zero array with matching dimension
                arr = np.zeros((1,), dtype=target_dtype)
            else:
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr[None, :]
                elif arr.ndim == 2 and key in ("my_active_feat", "opp_active_feat", "my_bench_feat", "opp_bench_feat", "option_feat"):
                    arr = arr[None, :, :]
                if arr.dtype != target_dtype:
                    arr = arr.astype(target_dtype)
            batch_inputs[key] = arr
        return batch_inputs

    def evaluate(self, obs_dict: dict, player_idx: int) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Runs the forward pass for an observation.
        Returns:
            policy_probs: [max_actions] float array (softmax over legal actions)
            value: float in [-1, 1] representing game evaluation
            latent_z: [latent_dim] float array
        """
        enc = encode_observation(obs_dict, player_idx, self.cfg.model)
        batch_inputs = self._prepare_inputs(enc)
        
        z, pi_logits, v = self.engine.forward(batch_inputs)
        
        logits = np.asarray(pi_logits[0])
        val = float(v[0]) if hasattr(v, "__len__") else float(v)
        
        mask = enc["option_mask"]
        masked_logits = np.where(mask, logits, -1e9)
        
        # Softmax over legal actions
        max_l = np.max(masked_logits)
        if max_l > -1e8:
            exp_logits = np.exp(masked_logits - max_l) * mask
            sum_exp = np.sum(exp_logits)
            probs = exp_logits / (sum_exp + 1e-8) if sum_exp > 0 else np.zeros_like(logits)
        else:
            probs = np.zeros_like(logits)
            
        return probs, val, np.asarray(z[0])

    def choose_action(self, obs_dict: dict, player_idx: int) -> Tuple[List[int], Dict[str, Any]]:
        """
        Chooses the best action indices based on policy scores.
        Returns:
            chosen_indices: List[int] of selected option indices
            metadata: Dict with AI insights (predicted value, top ranked options, winrate)
        """
        select = obs_dict.get("select") if isinstance(obs_dict, dict) else getattr(obs_dict, "select", None)
        if select is None:
            return [], {"value": 0.0, "winrate": 50.0, "top_actions": []}

        raw_options = select.get("option", []) if isinstance(select, dict) else (getattr(select, "option", []) or [])
        min_cnt = int(select.get("minCount", 1) if isinstance(select, dict) else getattr(select, "minCount", 1))
        max_cnt = int(select.get("maxCount", 1) if isinstance(select, dict) else getattr(select, "maxCount", 1))

        if not raw_options or max_cnt == 0:
            return [], {"value": 0.0, "winrate": 50.0, "top_actions": []}

        probs, val, _ = self.evaluate(obs_dict, player_idx)
        
        enc = encode_observation(obs_dict, player_idx, self.cfg.model)
        mask = enc["option_mask"]

        valid_indices = [
            i for i in range(len(raw_options))
            if raw_options[i] is not None and (i < len(mask) and mask[i])
        ]
        if not valid_indices:
            valid_indices = [i for i in range(len(raw_options)) if raw_options[i] is not None]
        if not valid_indices:
            return [], {"value": val, "winrate": round(float((val + 1.0) / 2.0 * 100.0), 1), "top_actions": []}

        scores = [float(probs[i]) if i < len(probs) else 0.0 for i in valid_indices]
        ranked_order = np.argsort(-np.array(scores))
        ranked_indices = [valid_indices[k] for k in ranked_order]

        desired_cnt = min(max_cnt, len(ranked_indices))
        desired_cnt = max(desired_cnt, min(min_cnt, len(ranked_indices)))
        chosen = [int(x) for x in ranked_indices[:desired_cnt]]

        # Format top action explanations for the UI
        top_actions = []
        for k in ranked_order[:5]:
            idx = valid_indices[k]
            top_actions.append({
                "option_index": idx,
                "score": round(float(scores[k]) * 100.0, 1),
                "raw_option": raw_options[idx],
            })

        winrate = round(float(np.clip((val + 1.0) / 2.0, 0.0, 1.0) * 100.0), 1)

        return chosen, {
            "value": round(val, 3),
            "winrate": winrate,
            "top_actions": top_actions,
            "chosen_indices": chosen,
            "mode": "basic_iree",
        }


class ISMCTSMuZeroAgent:
    """Agent running full Gumbel MuZero ISMCTS search with belief sampling."""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        cards_csv: str = "competiton/EN_Card_Data.csv",
        repo_id: str = "richard151111/muzero-V2",
        step_prefix: str = "step_0198000",
        num_simulations: int = 50,
        num_belief_samples: int = 2,
    ):
        import jax
        import jax.numpy as jnp
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file
        from cards.encoder import CardStaticFeatures
        from export.hub import _unflatten_params
        from models.networks import MuZeroNetwork

        self.cfg = cfg or Config()
        self.cfg.search.num_simulations = num_simulations
        self.cfg.search.num_belief_samples = num_belief_samples

        sf_path = hf_hub_download(repo_id=repo_id, filename=f"{step_prefix}/muzero.safetensors")
        card_data = CardStaticFeatures(cards_csv)
        n_cards = max(card_data.max_card_id + 1, self.cfg.model.num_card_ids)
        self.cfg.model.num_card_ids = n_cards
        static_feats = jnp.array(card_data.feature_matrix(n_cards))

        self.network = MuZeroNetwork(cfg=self.cfg.model, static_features=static_feats)
        loaded_raw = _unflatten_params(load_file(sf_path))
        self.mz_params = loaded_raw.get("muzero", loaded_raw)
        self.rng = jax.random.PRNGKey(42)

        logger.info("ISMCTSMuZeroAgent ready (sims=%d, beliefs=%d)", num_simulations, num_belief_samples)

    def set_opponent_deck(self, deck_cards: List[int]) -> None:
        try:
            from search.ismcts import set_belief_deck
            set_belief_deck(deck_cards)
        except Exception as exc:
            logger.warning("Could not set belief deck: %s", exc)

    def choose_action(self, obs_dict: dict, player_idx: int) -> Tuple[List[int], Dict[str, Any]]:
        import jax
        from env.encoding import encode_observation
        from search.ismcts import ismcts_action

        select = obs_dict.get("select") if isinstance(obs_dict, dict) else getattr(obs_dict, "select", None)
        if select is None:
            return [], {"value": 0.0, "winrate": 50.0, "top_actions": []}

        raw_options = select.get("option", []) if isinstance(select, dict) else (getattr(select, "option", []) or [])
        min_cnt = int(select.get("minCount", 1) if isinstance(select, dict) else getattr(select, "minCount", 1))
        max_cnt = int(select.get("maxCount", 1) if isinstance(select, dict) else getattr(select, "maxCount", 1))

        if not raw_options or max_cnt == 0:
            return [], {"value": 0.0, "winrate": 50.0, "top_actions": []}

        enc = encode_observation(obs_dict, player_idx, self.cfg.model)
        mask = enc["option_mask"]

        self.rng, rng_act = jax.random.split(self.rng)
        try:
            best_act, avg_policy, avg_value = ismcts_action(
                self.network, self.mz_params, enc, mask, rng_act, self.cfg
            )
            probs = np.asarray(avg_policy)
            val = float(avg_value)
        except Exception as exc:
            logger.error("ISMCTS search error (%s), fallback to first legal option", exc)
            probs = np.zeros(len(mask), dtype=np.float32)
            val = 0.0

        valid_indices = [
            i for i in range(len(raw_options))
            if raw_options[i] is not None and (i < len(mask) and mask[i])
        ]
        if not valid_indices:
            valid_indices = [i for i in range(len(raw_options)) if raw_options[i] is not None]
        if not valid_indices:
            return [], {"value": val, "winrate": round(float((val + 1.0) / 2.0 * 100.0), 1), "top_actions": []}

        scores = [float(probs[i]) if i < len(probs) else 0.0 for i in valid_indices]
        ranked_order = np.argsort(-np.array(scores))
        ranked_indices = [valid_indices[k] for k in ranked_order]

        desired_cnt = min(max_cnt, len(ranked_indices))
        desired_cnt = max(desired_cnt, min(min_cnt, len(ranked_indices)))
        chosen = [int(x) for x in ranked_indices[:desired_cnt]]

        top_actions = []
        for k in ranked_order[:5]:
            idx = valid_indices[k]
            top_actions.append({
                "option_index": idx,
                "score": round(float(scores[k]) * 100.0, 1),
                "raw_option": raw_options[idx],
            })

        winrate = round(float(np.clip((val + 1.0) / 2.0, 0.0, 1.0) * 100.0), 1)

        return chosen, {
            "value": round(val, 3),
            "winrate": winrate,
            "top_actions": top_actions,
            "chosen_indices": chosen,
            "mode": "advanced_ismcts",
        }


def create_agent(
    mode: str = "basic",
    vmfb_path: str = "muzero_vulkan.vmfb",
    device_uri: str = "vulkan",
    num_simulations: int = 50,
    num_belief_samples: int = 2,
    cfg: Optional[Config] = None,
):
    """Factory creating either an IREE fast agent (basic) or an ISMCTS search agent (advanced)."""
    if mode.lower() == "advanced" or mode.lower() == "ismcts":
        return ISMCTSMuZeroAgent(
            cfg=cfg,
            num_simulations=num_simulations,
            num_belief_samples=num_belief_samples,
        )
    return IREEMuZeroAgent(vmfb_path=vmfb_path, device_uri=device_uri, cfg=cfg)


