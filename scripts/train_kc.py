"""
HKM Training Entry Point (self-contained)
==========================================
1. Sets MuJoCo rendering environment variables for forked workers.
2. Registers Hydra resolvers (load_stats, eval).
3. Patches RobomimicLowdimWrapper at the CLASS level:
   - get_observation(): appends 42-D k(q) based on TRUE joint positions [used by __init__, reset]
   - step(): rewritten to call get_observation() [original bypasses it]
4. Patches EnvUtils to strictly enable offscreen rendering for WandB videos.
5. Delegates to the upstream diffusion_policy/train.py main().

File layout:  /workspace/scripts/train_kc.py
   2 x dirname -> /workspace  (PROJECT_ROOT)
"""

import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
import multiprocessing
from kc_dp.kinematics.kq_feature import get_pid_cached_kq_computer

# =====================================================================
# 0. Path & Rendering Setup (MUST be executed before any MuJoCo import)
# =====================================================================
# Ensure offscreen rendering works correctly in forked workers
os.environ.setdefault("MUJOCO_GL", "egl")

# Force 'spawn' method to prevent EGL context corruption in multiprocess Rollouts
try:
    multiprocessing.set_start_method('spawn', force=True)
except Exception:
    pass

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

def _compute_kq(q_pos: np.ndarray) -> np.ndarray:
    comp = get_pid_cached_kq_computer(
        urdf_path=_URDF_PATH,
        ee_frame_name="panda_link8",
        max_dof=7,
        stats_path=_STATS_PATH,
    )
    n = len(comp.q_min_base)
    q = q_pos[:n].astype(np.float64)
    return comp.compute_normalized(q)


from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
    RobomimicLowdimWrapper,
)

_orig_get_obs = RobomimicLowdimWrapper.get_observation

def _patched_get_observation(self):
    obs = _orig_get_obs(self)
    raw_obs_dict = self.env.get_observation()
    actual_q = raw_obs_dict['robot0_joint_pos']
    kq = _compute_kq(actual_q)
    return np.concatenate([obs, kq])

RobomimicLowdimWrapper.get_observation = _patched_get_observation

def _patched_step(self, action):
    raw_obs, reward, done, info = self.env.step(action)
    obs = self.get_observation()  
    return obs, reward, done, info

RobomimicLowdimWrapper.step = _patched_step

# =====================================================================
# 2.5 Force Offscreen Rendering for Video Rollouts (Vis)
# =====================================================================
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

_orig_get_env_meta = FileUtils.get_env_metadata_from_dataset

def patched_get_env_meta(dataset_path):
    meta = _orig_get_env_meta(dataset_path)
    meta['env_kwargs']['has_offscreen_renderer'] = True
    meta['env_kwargs']['has_renderer'] = False
    return meta

FileUtils.get_env_metadata_from_dataset = patched_get_env_meta

_orig_create_env = EnvUtils.create_env_from_metadata

def patched_create_env(env_meta, env_name=None, render=False, render_offscreen=False, use_image_obs=False, **kwargs):
    return _orig_create_env(
        env_meta=env_meta, 
        env_name=env_name, 
        render=render, 
        render_offscreen=True, # Unconditionally force offscreen rendering
        use_image_obs=use_image_obs, 
        **kwargs
    )

EnvUtils.create_env_from_metadata = patched_create_env

# =====================================================================
# 3. Prevent EGL Rendering Crashes during Vis Rollouts
# =====================================================================
import diffusion_policy.env_runner.robomimic_lowdim_runner as runner_mod
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
runner_mod.AsyncVectorEnv = SyncVectorEnv

# =====================================================================
# 4. Execute Upstream Training
# =====================================================================
from train import main

if __name__ == "__main__":
    main()