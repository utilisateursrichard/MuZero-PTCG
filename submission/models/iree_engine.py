"""
ptcg_muzero/models/iree_engine.py
==================================
IREE Runtime execution engine for MuZero networks compiled to Vulkan / CPU.
Provides transparent inference for:
- represent(obs): [B, latent_dim]
- predict(z): policy logits [B, num_actions], value [B]
- dynamics(z, action): next_z [B, latent_dim], reward [B]
- forward(obs): z, policy logits, value
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("iree_engine")


class IREEMuZeroEngine:
    """IREE-backed execution engine for MuZero models running on Vulkan GPU or CPU."""

    def __init__(
        self,
        vmfb_path: Union[str, Path],
        device_uri: str = "vulkan",
        driver_name: Optional[str] = None,
    ):
        import iree.runtime as ireert

        self.vmfb_path = Path(vmfb_path).resolve()
        if not self.vmfb_path.exists():
            raise FileNotFoundError(f"VMFB module not found: {self.vmfb_path}")

        self.device_uri = device_uri
        try:
            self.config = ireert.Config(driver_name=driver_name or device_uri)
            logger.info("Initialized IREE runtime on device: %s", self.config.device)
        except Exception as e:
            logger.warning(
                "Failed to initialize IREE driver '%s' (%s). Falling back to local-task CPU.",
                device_uri,
                e,
            )
            self.config = ireert.Config(driver_name="local-task")

        self.system_context = ireert.SystemContext(config=self.config)
        self.vm_module = ireert.VmModule.mmap(self.config.vm_instance, str(self.vmfb_path))
        self.system_context.add_vm_module(self.vm_module)
        mod_name = self.vm_module.name
        self.module = self.system_context.modules[mod_name]

    def represent(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Runs representation network h(obs) -> z."""
        ordered_keys = sorted(obs.keys())
        args = [np.asarray(obs[k]) for k in ordered_keys]
        if hasattr(self.module, "represent"):
            res = self.module.represent(*args)
            return np.asarray(res)
        else:
            z, _, _ = self.forward(obs)
            return z

    def predict(self, z: np.ndarray, option_feat: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Runs prediction network f(z) -> policy_logits, value."""
        z_arr = np.asarray(z)
        if hasattr(self.module, "predict"):
            if option_feat is not None:
                pi, v = self.module.predict(z_arr, np.asarray(option_feat))
            else:
                pi, v = self.module.predict(z_arr)
            return np.asarray(pi), np.asarray(v)
        else:
            raise AttributeError("Compiled VMFB module has no 'predict' function.")

    def dynamics(self, z: np.ndarray, action_onehot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs dynamics network g(z, a) -> next_z, reward."""
        z_arr = np.asarray(z)
        a_arr = np.asarray(action_onehot)
        if hasattr(self.module, "dynamics"):
            z_next, r = self.module.dynamics(z_arr, a_arr)
            return np.asarray(z_next), np.asarray(r)
        else:
            raise AttributeError("Compiled VMFB module has no 'dynamics' function.")

    def forward(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs unified forward pass obs -> z, policy_logits, value."""
        ordered_keys = sorted(obs.keys())
        args = [np.asarray(obs[k]) for k in ordered_keys]
        if hasattr(self.module, "main"):
            z, pi, v = self.module.main(*args)
            return np.asarray(z), np.asarray(pi), np.asarray(v)
        elif hasattr(self.module, "forward"):
            z, pi, v = self.module.forward(*args)
            return np.asarray(z), np.asarray(pi), np.asarray(v)
        else:
            raise AttributeError("Compiled VMFB module has no 'main' or 'forward' function.")
