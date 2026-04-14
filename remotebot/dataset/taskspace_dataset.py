import copy
import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.normalize_util import array_to_stats, get_identity_normalizer_from_stat


def _quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat must be [T,4], got shape={q.shape}")
    eps = 1e-8
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)
    xyz = q[:, :3]
    w = np.clip(q[:, 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = xyz / np.maximum(sin_half[:, None], eps)
    aa = axis * angle[:, None]
    small = sin_half < 1e-5
    if np.any(small):
        aa[small] = 2.0 * xyz[small]
    return aa.astype(np.float32)


def _build_taskspace_raw_action(raw_obs, action_mode: str = "obs_abs_axis_angle") -> np.ndarray:
    pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
    quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)
    grip = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)

    if grip.ndim > 1:
        g = grip.mean(axis=-1, keepdims=True)
    else:
        g = grip.reshape(-1, 1)

    if action_mode == "obs_abs_axis_angle":
        rot = _quat_xyzw_to_axis_angle(quat)
        return np.concatenate([pos, rot, g.astype(np.float32)], axis=-1).astype(np.float32)

    if action_mode == "obs_abs_quat":
        return np.concatenate([pos, quat, g.astype(np.float32)], axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported action_mode: {action_mode}")


def _data_to_obs_taskspace(
    raw_obs,
    raw_actions,
    obs_keys,
    abs_action,
    rotation_transformer,
    action_mode: str,
):
    obs_list = []
    for key in obs_keys:
        if key not in raw_obs:
            raise KeyError(f"obs key '{key}' missing in dataset episode")
        obs_list.append(raw_obs[key])
    obs = np.concatenate(obs_list, axis=-1).astype(np.float32)

    if action_mode.startswith("obs_abs"):
        raw_actions = _build_taskspace_raw_action(raw_obs, action_mode=action_mode)
    else:
        raw_actions = np.asarray(raw_actions, dtype=np.float32)

    if abs_action and raw_actions.shape[-1] == 7:
        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

    data = {
        "obs": obs,
        "action": raw_actions.astype(np.float32),
    }
    return data


class TaskSpaceLowdimDataset(BaseLowdimDataset):
    """
    RemoteBot task-space dataset.

    Default behavior builds actions from observation trajectory to enforce
    task-space action semantics consistently across raw and augmented datasets.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys: List[str] = [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        abs_action=True,
        rotation_rep="rotation_6d",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        action_mode: str = "obs_abs_axis_angle",
    ):
        dataset_path = self._resolve_dataset_path(str(dataset_path))

        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file["data"]
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f"demo_{i}"]
                episode = _data_to_obs_taskspace(
                    raw_obs=demo["obs"],
                    raw_actions=demo["actions"][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    action_mode=action_mode,
                )
                replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    @staticmethod
    def _resolve_dataset_path(dataset_path: str) -> str:
        if os.path.exists(dataset_path):
            return dataset_path

        if dataset_path.startswith("data/robomimic/"):
            cand = dataset_path.replace(
                "data/robomimic/",
                "third_party/diffusion_policy/data/robomimic/",
                1,
            )
            if os.path.exists(cand):
                return cand

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        cand2 = os.path.join(project_root, dataset_path)
        if os.path.exists(cand2):
            return cand2

        if "data/robomimic/" in dataset_path:
            rel = dataset_path.split("data/robomimic/", 1)[1]
            cand3 = os.path.join(
                project_root,
                "third_party",
                "diffusion_policy",
                "data",
                "robomimic",
                rel,
            )
            if os.path.exists(cand3):
                return cand3

        raise FileNotFoundError(
            f"Dataset not found: '{dataset_path}'."
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        action_stat = array_to_stats(self.replay_buffer["action"])
        # Task-space action is explicitly normalized by dataset stats,
        # independent from robomimic's abs_action axis-angle assumptions.
        normalizer["action"] = normalizer_from_stat(action_stat)

        obs_stat = array_to_stats(self.replay_buffer["obs"])
        normalizer["obs"] = get_identity_normalizer_from_stat(obs_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
