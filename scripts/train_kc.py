"""
HKM Training Entry Point (self-contained)
==========================================
1. Sets MuJoCo rendering environment variables for forked workers.
2. Registers Hydra resolvers (load_stats, eval).
3. Patches RobomimicLowdimWrapper at the CLASS level:
   - get_observation(): appends 42-D k(q) based on TRUE joint positions [used by __init__, reset]
   - step(): rewritten to call get_observation() [original bypasses it]
4. Delegates to the upstream diffusion_policy/train.py main().

File layout:  /workspace/scripts/train_kc.py
   2 x dirname -> /workspace  (PROJECT_ROOT)
"""

import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

# =====================================================================
# 0. Path & Rendering Setup (MUST be executed before any MuJoCo import)
# =====================================================================
# Ensure offscreen rendering works correctly in forked workers
os.environ.setdefault("MUJOCO_GL", "egl")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party", "diffusion_policy"))

# =====================================================================
# 1. Hydra Resolvers
# =====================================================================
def _load_stats(path: str, key: str):
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    if os.path.exists(path):
        try:
            return torch.load(path, map_location="cpu")[key]
        except Exception:
            pass
    return None

for _rname, _rfn in [("load_stats", _load_stats), ("eval", eval)]:
    try:
        OmegaConf.register_new_resolver(_rname, _rfn, replace=True)
    except Exception:
        pass

# =====================================================================
# 2. Class-level Observation Patch
# =====================================================================
_URDF_PATH = os.path.join(_PROJECT_ROOT, "data", "urdf", "franka_panda", "urdf", "panda.urdf")
_STATS_PATH = os.path.join(_PROJECT_ROOT, "data", "k_q_stats.pt")

_K_MEAN = np.zeros(42, dtype=np.float32)
_K_STD = np.ones(42, dtype=np.float32)

if os.path.exists(_STATS_PATH):
    _st = torch.load(_STATS_PATH, map_location="cpu")
    _K_MEAN = np.array(_st["mean"], dtype=np.float32)
    _K_STD = np.clip(np.array(_st["std"], dtype=np.float32), 1e-6, None)

_PID_CACHE: dict = {}

def _get_km():
    """
    Initializes and caches the Kinematic Module per process.
    """
    pid = os.getpid()
    if pid not in _PID_CACHE:
        from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule
        km = AnalyticKinematicModule(
            urdf_path=_URDF_PATH,
            ee_frame_name="panda_link8",
            max_dof=7,
        )
        idx = km._arm_q_indices
        q_min = np.where(
            np.isinf(km.model.lowerPositionLimit[idx]),
            -2 * np.pi, km.model.lowerPositionLimit[idx],
        ).astype(np.float64)
        q_max = np.where(
            np.isinf(km.model.upperPositionLimit[idx]),
            2 * np.pi, km.model.upperPositionLimit[idx],
        ).astype(np.float64)
        _PID_CACHE[pid] = (km, q_min, q_max)
    return _PID_CACHE[pid]

def _compute_kq(q_pos: np.ndarray) -> np.ndarray:
    """
    Computes the normalized k(q) feature using true joint positions.
    """
    km, q_min, q_max = _get_km()
    n = len(q_min)
    q = q_pos[:n].astype(np.float64)
    raw = km.compute_k_q_with_custom_limits(q[np.newaxis], q_min, q_max)[0]
    return ((raw - _K_MEAN) / _K_STD).astype(np.float32)


# ---------- Apply Environment Patches ----------
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
    RobomimicLowdimWrapper,
)

# Patch 1: get_observation (invoked during __init__ and reset)
_orig_get_obs = RobomimicLowdimWrapper.get_observation

def _patched_get_observation(self):
    # Retrieve the 1D concatenated base observation (e.g., 23D)
    obs = _orig_get_obs(self)
    
    # Fetch the native raw dictionary to extract the authentic joint angles
    raw_obs_dict = self.env.get_observation()
    actual_q = raw_obs_dict['robot0_joint_pos']
    
    # Compute k(q) correctly using the actual joint angles
    kq = _compute_kq(actual_q)
    return np.concatenate([obs, kq])

RobomimicLowdimWrapper.get_observation = _patched_get_observation

# Patch 2: step (the original bypasses get_observation and performs inline concatenation)
def _patched_step(self, action):
    raw_obs, reward, done, info = self.env.step(action)
    obs = self.get_observation()  # Returns the patched 65D observation
    return obs, reward, done, info

RobomimicLowdimWrapper.step = _patched_step

print(f"[HKM] Successfully patched get_observation and step with TRUE JOINT ANGLES (PID={os.getpid()})")

# =====================================================================
# 3. Execute Upstream Training
# =====================================================================
from train import main

if __name__ == "__main__":
    main()