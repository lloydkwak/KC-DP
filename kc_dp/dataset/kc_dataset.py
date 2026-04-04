import numpy as np
import torch
import torch.nn as nn
import h5py
from typing import Dict, Any, Optional, Tuple
import copy

import pinocchio as pin

from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import (
    RobomimicReplayLowdimDataset,
)
from diffusion_policy.common.sampler import SequenceSampler
from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule
from kc_dp.kinematics.virtual_sampler import VirtualRobotSampler
from kc_dp.dataset.density_sampler import DensityWeightedSampler


class HKMLowdimDataset(RobomimicReplayLowdimDataset):
    """
    Hierarchical Kinematic Modulation (HKM) Dataset Wrapper.

    The parent class concatenates all obs_keys into a single 'obs' array and
    stores only 'obs' and 'action' in the ReplayBuffer — individual keys like
    'robot0_joint_pos' are *not* preserved.  This class re-reads the HDF5 to
    inject 'robot0_joint_pos' as a separate ReplayBuffer column so that k(q)
    can be computed per-timestep.
    """

    def __init__(
        self,
        *args,
        virtual_sampler: Optional[VirtualRobotSampler] = None,
        base_kinematic_module: Optional[AnalyticKinematicModule] = None,
        k_feature_mode: str = "base_physical",
        use_density_sampler: bool = False,
        density_k_neighbors: int = 5,
        density_temperature: float = 1.0,
        density_max_weight_ratio: float = 5.0,
        stats_path: str = "data/k_q_stats.pt",
        k_mean: Optional[Any] = None,
        k_std: Optional[Any] = None,
        k_clip_value: float = 3.0,
        **kwargs,
    ):
        kwargs.pop("zarr_path", None)

        # ── 1. Build the parent dataset (creates self.replay_buffer) ──
        # We need dataset_path before super().__init__ consumes it.
        # It is either the first positional arg or a keyword arg.
        if args:
            dataset_path = args[0]
        else:
            dataset_path = kwargs.get("dataset_path")
            
        if dataset_path is None:
            raise ValueError("dataset_path must be provided.")

        super().__init__(*args, **kwargs)

        # ── 2. Re-read robot0_joint_pos from HDF5 and inject into buffer ──
        self._inject_joint_pos(dataset_path)

        # ── 3. HKM-specific attributes ──
        self.virtual_sampler = virtual_sampler
        self.base_kinematic_module = base_kinematic_module
        self.k_feature_mode = k_feature_mode
        self.use_density_sampler = use_density_sampler
        self.density_k_neighbors = density_k_neighbors
        self.density_temperature = density_temperature
        self.density_max_weight_ratio = density_max_weight_ratio
        self.k_clip_value = k_clip_value

        if self.virtual_sampler is None and self.base_kinematic_module is None:
            raise ValueError(
                "Either virtual_sampler or base_kinematic_module must be provided."
            )
        if self.k_feature_mode not in ("base_physical", "virtual"):
            raise ValueError("k_feature_mode must be either 'base_physical' or 'virtual'.")

        # Type-safe tensor init (accepts list / ListConfig from Hydra)
        stats = None
        if (k_mean is None) or (k_std is None):
            stats = torch.load(stats_path, map_location='cpu')

        if k_mean is not None:
            self.k_mean = torch.from_numpy(np.array(k_mean, dtype=np.float32)).float()
        else:
            self.k_mean = torch.as_tensor(stats['mean']).float()

        if k_std is not None:
            self.k_std = torch.from_numpy(np.array(k_std, dtype=np.float32)).float()
        else:
            self.k_std = torch.as_tensor(stats['std']).float()
        self.k_std = torch.clamp(self.k_std, min=1e-6)

        if self.base_kinematic_module is not None:
            idx = self.base_kinematic_module._arm_q_indices
            self._base_q_min = np.where(
                np.isinf(self.base_kinematic_module.model.lowerPositionLimit[idx]),
                -2 * np.pi,
                self.base_kinematic_module.model.lowerPositionLimit[idx],
            ).astype(np.float64)
            self._base_q_max = np.where(
                np.isinf(self.base_kinematic_module.model.upperPositionLimit[idx]),
                2 * np.pi,
                self.base_kinematic_module.model.upperPositionLimit[idx],
            ).astype(np.float64)

        # Virtual-module cache (cleared every epoch)
        self._vmodule_cache: Dict[
            int, Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]
        ] = {}
        self._build_episode_mapping()

        self._getitem_count = 0
        self._density_sampler = None

    # ------------------------------------------------------------------
    # HDF5 re-read
    # ------------------------------------------------------------------

    def _inject_joint_pos(self, dataset_path: str):
        """
        Reads 'robot0_joint_pos' from every demo in the HDF5 and appends it
        as a new column in self.replay_buffer so that downstream code can
        access it via self.replay_buffer['robot0_joint_pos'].
        """
        all_joint_pos = []
        with h5py.File(dataset_path, "r") as f:
            demos = f["data"]
            for i in range(len(demos)):
                jp = demos[f"demo_{i}"]["obs"]["robot0_joint_pos"][:]
                all_joint_pos.append(jp.astype(np.float32))
        joint_pos_all = np.concatenate(all_joint_pos, axis=0)

        # Sanity: length must match the existing buffer
        assert joint_pos_all.shape[0] == self.replay_buffer.n_steps, (
            f"joint_pos length {joint_pos_all.shape[0]} != "
            f"replay_buffer n_steps {self.replay_buffer.n_steps}"
        )

        # ReplayBuffer has no __setitem__; write directly into the backing store.
        self.replay_buffer.root['data']['robot0_joint_pos'] = joint_pos_all

    # ------------------------------------------------------------------
    # Episode mapping
    # ------------------------------------------------------------------

    def _build_episode_mapping(self):
        episode_ends = self.replay_buffer.episode_ends[:]
        total_raw_steps = int(episode_ends[-1])

        self.raw_to_ep = np.zeros(total_raw_steps, dtype=int)
        starts = np.insert(episode_ends[:-1], 0, 0)
        for ep_idx, (start, end) in enumerate(zip(starts, episode_ends)):
            self.raw_to_ep[int(start) : int(end)] = ep_idx

    def clear_epoch_cache(self):
        self._vmodule_cache.clear()
        self._density_sampler = None

    def get_train_sampler(self):
        """
        Stage 3 density-weighted sampling.
        Returns None when disabled.
        """
        if not self.use_density_sampler:
            return None
        if self._density_sampler is None:
            self._density_sampler = DensityWeightedSampler(
                dataset=self,
                kinematic_module=self.base_kinematic_module,
                k_neighbors=self.density_k_neighbors,
                temperature=self.density_temperature,
                max_weight_ratio=self.density_max_weight_ratio,
            )
        return self._density_sampler.get_sampler()

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
        val_set.use_density_sampler = False
        val_set._density_sampler = None
        return val_set

    # ------------------------------------------------------------------
    # Virtual module creation (Stage 1 + 2)
    # ------------------------------------------------------------------

    def _get_or_create_virtual_module(
        self, ep_idx: int
    ) -> Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]:
        if ep_idx in self._vmodule_cache:
            return self._vmodule_cache[ep_idx]

        n_dof = self.replay_buffer["robot0_joint_pos"].shape[1]

        if self.virtual_sampler is None:
            model = self.base_kinematic_module.model
            idx = self.base_kinematic_module._arm_q_indices[:n_dof]
            q_min = np.where(
                np.isinf(model.lowerPositionLimit[idx]),
                -2 * np.pi,
                model.lowerPositionLimit[idx],
            )
            q_max = np.where(
                np.isinf(model.upperPositionLimit[idx]),
                2 * np.pi,
                model.upperPositionLimit[idx],
            )
            result = (self.base_kinematic_module, q_min, q_max)
            self._vmodule_cache[ep_idx] = result
            return result

        # Full episode joint trajectory
        episode_ends = self.replay_buffer.episode_ends[:]
        start_idx = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end_idx = int(episode_ends[ep_idx])

        q_traj_full = self.replay_buffer["robot0_joint_pos"][start_idx:end_idx]
        T = len(q_traj_full)

        # Finite-difference joint velocity
        delta_q = np.zeros_like(q_traj_full)
        if T > 1:
            delta_q[:-1] = q_traj_full[1:] - q_traj_full[:-1]
            delta_q[-1] = delta_q[-2]

        delta_pose_actual = np.zeros((T, 6))
        # Worker-local data to avoid multi-process race conditions
        local_data = self.base_kinematic_module.model.createData()

        for t in range(T):
            q_arm = q_traj_full[t]
            # Pad arm-only q (e.g. 7D) to full model config (e.g. 9D)
            q_full = self.base_kinematic_module._pad_q(q_arm)
            pin.framesForwardKinematics(
                self.base_kinematic_module.model, local_data, q_full
            )
            J_full = pin.computeFrameJacobian(
                self.base_kinematic_module.model,
                local_data,
                q_full,
                self.base_kinematic_module.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            # Use only arm-chain Jacobian columns
            n_eff = min(q_traj_full.shape[1], len(self.base_kinematic_module._arm_v_indices))
            J_arm = J_full[:, self.base_kinematic_module._arm_v_indices[:n_eff]]
            delta_pose_actual[t] = J_arm @ delta_q[t][:n_eff]

        # Stage 1 + 2
        q_min_v, q_max_v = self.virtual_sampler.sample_stage1_limits(
            np.min(q_traj_full, axis=0), np.max(q_traj_full, axis=0)
        )
        v_module, q_min_v, q_max_v = self.virtual_sampler.sample_stage2_module(
            q_traj_full, delta_pose_actual, q_min_v, q_max_v
        )

        result = (v_module, q_min_v, q_max_v)
        self._vmodule_cache[ep_idx] = result
        return result

    # ------------------------------------------------------------------
    # Normalizer expansion
    # ------------------------------------------------------------------

    def get_normalizer(self, mode="limits", **kwargs):
        """
        Expands the parent's obs normalizer by 42 identity-scaled dims for k(q).
        """
        normalizer = super().get_normalizer(mode=mode, **kwargs)
        obs_params = normalizer["obs"].params_dict

        device = obs_params["offset"].device
        dtype = obs_params["offset"].dtype

        obs_params["offset"] = nn.Parameter(
            torch.cat(
                [obs_params["offset"], torch.zeros(42, dtype=dtype, device=device)]
            )
        )
        obs_params["scale"] = nn.Parameter(
            torch.cat(
                [obs_params["scale"], torch.ones(42, dtype=dtype, device=device)]
            )
        )

        if "input_stats" in obs_params:
            stats = obs_params["input_stats"]
            for key in list(stats.keys()):
                pad = torch.zeros(42, dtype=dtype, device=device)
                if key in ("std", "max"):
                    pad = torch.ones(42, dtype=dtype, device=device)
                stats[key] = nn.Parameter(torch.cat([stats[key], pad]))

        return normalizer

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._getitem_count += 1

        # In a multi-worker environment (num_workers > 0), each worker has its own copy of the dataset.
        # We refresh the cache roughly once per epoch per worker.
        # Using 1/10th of len(self) is a safe heuristic for most multi-worker setups.
        if self._getitem_count % max(1, len(self) // 10) == 0: #
            self.clear_epoch_cache()

        item = super().__getitem__(idx)

        # ── Extract joint positions with proper padding ──
        buf_start, buf_end, samp_start, samp_end = self.sampler.indices[idx]
        seq_len = self.sampler.sequence_length

        q_raw = self.replay_buffer["robot0_joint_pos"][buf_start:buf_end]

        # Replicate the same padding logic as SequenceSampler.sample_sequence
        if samp_start > 0 or samp_end < seq_len:
            q_window = np.zeros(
                (seq_len, q_raw.shape[1]), dtype=q_raw.dtype
            )
            if samp_start > 0:
                q_window[:samp_start] = q_raw[0]
            if samp_end < seq_len:
                q_window[samp_end:] = q_raw[-1]
            q_window[samp_start:samp_end] = q_raw
        else:
            q_window = q_raw

        # ── Resolve episode index ──
        raw_start_idx = int(buf_start)
        # Clamp to valid range (buf_start can be 0 which is always valid)
        ep_idx = self.raw_to_ep[min(raw_start_idx, len(self.raw_to_ep) - 1)]

        # ── Compute k(q) ──
        if self.k_feature_mode == "base_physical":
            n = q_window.shape[1]
            k_q_np = self.base_kinematic_module.compute_k_q_with_custom_limits(
                q_window, self._base_q_min[:n], self._base_q_max[:n]
            )
        else:
            v_module, q_min_v, q_max_v = self._get_or_create_virtual_module(ep_idx)
            k_q_np = v_module.compute_k_q_with_custom_limits(
                q_window, q_min_v, q_max_v
            )

        k_q = torch.from_numpy(k_q_np).float()
        k_q_normalized = (k_q - self.k_mean) / self.k_std
        if self.k_clip_value is not None:
            k_q_normalized = torch.clamp(
                k_q_normalized,
                min=-float(self.k_clip_value),
                max=float(self.k_clip_value),
            )

        # ── Concatenate onto the unified obs tensor ──
        item["obs"] = torch.cat([item["obs"], k_q_normalized], dim=-1)

        item.pop("robot0_joint_pos", None)

        return item
