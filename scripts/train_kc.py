"""
HKM Training Entry Point (self-contained)
==========================================
1. Sets MuJoCo rendering env vars for forked workers.
2. Registers Hydra resolvers (load_stats, eval).
3. Patches RobomimicLowdimWrapper at the CLASS level:
   - get_observation(): appends 42-D k(q)  [used by __init__, reset]
   - step(): rewritten to call get_observation() [original bypasses it]
4. Delegates to the upstream diffusion_policy/train.py main().

File layout:  /workspace/scripts/train_kc.py
   2 x dirname -> /workspace  (PROJECT_ROOT)
"""

import os
import sys
import numpy as np

# =====================================================================
# 0. Path & rendering setup (MUST be before any MuJoCo import)
# =====================================================================
# Ensure offscreen rendering works in forked workers
os.environ.setdefault("MUJOCO_GL", "egl")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party", "diffusion_policy"))

# =====================================================================
# 1. Hydra resolvers
# =====================================================================
import torch
from omegaconf import OmegaConf


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
# 2. Class-level observation patch
# =====================================================================
_URDF_PATH = os.path.join(
    _PROJECT_ROOT, "data", "urdf", "franka_panda", "urdf", "panda.urdf"
)
_STATS_PATH = os.path.join(_PROJECT_ROOT, "data", "k_q_stats.pt")

_K_MEAN = np.zeros(42, dtype=np.float32)
_K_STD = np.ones(42, dtype=np.float32)
if os.path.exists(_STATS_PATH):
    _st = torch.load(_STATS_PATH, map_location="cpu")
    _K_MEAN = np.array(_st["mean"], dtype=np.float32)
    _K_STD = np.clip(np.array(_st["std"], dtype=np.float32), 1e-6, None)

_PID_CACHE: dict = {}


def _get_km():
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


def _compute_kq(obs_flat: np.ndarray) -> np.ndarray:
    km, q_min, q_max = _get_km()
    n = len(q_min)
    q = obs_flat[:n].astype(np.float64)
    raw = km.compute_k_q_with_custom_limits(q[np.newaxis], q_min, q_max)[0]
    return ((raw - _K_MEAN) / _K_STD).astype(np.float32)


# ---------- Apply patches ----------
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
    RobomimicLowdimWrapper,
)

# Patch 1: get_observation (called by __init__ and reset)
_orig_get_obs = RobomimicLowdimWrapper.get_observation


def _patched_get_observation(self):
    obs = _orig_get_obs(self)
    kq = _compute_kq(obs)
    return np.concatenate([obs, kq])


RobomimicLowdimWrapper.get_observation = _patched_get_observation

# Patch 2: step (original bypasses get_observation, does inline concat)
def _patched_step(self, action):
    raw_obs, reward, done, info = self.env.step(action)
    obs = self.get_observation()  # patched -> 65D
    return obs, reward, done, info


RobomimicLowdimWrapper.step = _patched_step

print(f"[HKM] Patched get_observation + step (PID={os.getpid()})")

# =====================================================================
# 3. Run upstream training
# =====================================================================
from train import main

if __name__ == "__main__":
    main()
