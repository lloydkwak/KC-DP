"""
RemoteBot Training Entry Point
==============================
- Keeps MuJoCo offscreen-safe runtime patches.
- Does NOT inject k(q) features.
- Delegates to diffusion_policy/train.py main().
"""

import os
import sys
import multiprocessing

import torch
from omegaconf import OmegaConf


os.environ.setdefault("MUJOCO_GL", "egl")

try:
    multiprocessing.set_start_method("spawn", force=True)
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party", "diffusion_policy"))


def _load_stats(path: str, key: str):
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    if os.path.exists(path):
        try:
            try:
                st = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                st = torch.load(path, map_location="cpu")
            return st[key]
        except Exception:
            return None
    return None


for _rname, _rfn in [("load_stats", _load_stats), ("eval", eval)]:
    try:
        OmegaConf.register_new_resolver(_rname, _rfn, replace=True)
    except Exception:
        pass


import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

_orig_get_env_meta = FileUtils.get_env_metadata_from_dataset


def patched_get_env_meta(dataset_path):
    meta = _orig_get_env_meta(dataset_path)
    meta["env_kwargs"]["has_offscreen_renderer"] = True
    meta["env_kwargs"]["has_renderer"] = False
    return meta


FileUtils.get_env_metadata_from_dataset = patched_get_env_meta

_orig_create_env = EnvUtils.create_env_from_metadata


def patched_create_env(
    env_meta,
    env_name=None,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
    **kwargs,
):
    return _orig_create_env(
        env_meta=env_meta,
        env_name=env_name,
        render=render,
        render_offscreen=True,
        use_image_obs=use_image_obs,
        **kwargs,
    )


EnvUtils.create_env_from_metadata = patched_create_env

import diffusion_policy.env_runner.robomimic_lowdim_runner as runner_mod
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv

runner_mod.AsyncVectorEnv = SyncVectorEnv

from train import main


if __name__ == "__main__":
    main()
