import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

import pinocchio as pin

from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule
from kc_dp.kinematics.virtual_sampler import VirtualRobotSampler

class HKMLowdimDataset(RobomimicReplayLowdimDataset):
    """
    Hierarchical Kinematic Modulation (HKM) Dataset Wrapper.
    
    Dynamically generates virtual robots, computes normalized k(q) features, 
    and seamlessly expands the core U-Net observation tensor and normalizer.
    """

    def __init__(self,
                 *args,
                 virtual_sampler: Optional[VirtualRobotSampler] = None,
                 base_kinematic_module: Optional[AnalyticKinematicModule] = None,
                 k_mean: Optional[Any] = None, # Typed to accept Hydra ListConfig
                 k_std: Optional[Any] = None,  # Typed to accept Hydra ListConfig
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.virtual_sampler = virtual_sampler
        self.base_kinematic_module = base_kinematic_module
        
        if self.virtual_sampler is None and self.base_kinematic_module is None:
            raise ValueError("Either virtual_sampler or base_kinematic_module must be provided.")
            
        # Type-Safe Tensor Initialization:
        # Casts Python lists/ListConfigs to numpy arrays, then to PyTorch tensors.
        if k_mean is not None:
            k_mean_np = np.array(k_mean, dtype=np.float32)
            self.k_mean = torch.from_numpy(k_mean_np).float()
        else:
            self.k_mean = torch.zeros(42)
            
        if k_std is not None:
            k_std_np = np.array(k_std, dtype=np.float32)
            self.k_std = torch.from_numpy(k_std_np).float()
        else:
            self.k_std = torch.ones(42)
            
        # Prevent division-by-zero instability during normalization
        self.k_std = torch.clamp(self.k_std, min=1e-6)
        
        # Epoch-level cache for virtual kinematic modules
        self._vmodule_cache: Dict[int, Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]] = {}
        self._build_episode_mapping()

    def _build_episode_mapping(self):
        episode_ends = self.replay_buffer.episode_ends[:]
        total_raw_steps = episode_ends[-1]
        
        self.raw_to_ep = np.zeros(total_raw_steps, dtype=int)
        starts = np.insert(episode_ends[:-1], 0, 0)
        for ep_idx, (start, end) in enumerate(zip(starts, episode_ends)):
            self.raw_to_ep[start:end] = ep_idx
            
    def clear_epoch_cache(self):
        self._vmodule_cache.clear()

    def _get_or_create_virtual_module(self, ep_idx: int) -> Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]:
        if ep_idx in self._vmodule_cache:
            return self._vmodule_cache[ep_idx]
            
        if self.virtual_sampler is None:
            model = self.base_kinematic_module.model
            n_dof = self.replay_buffer['robot0_joint_pos'].shape[1]
            q_min = np.where(np.isinf(model.lowerPositionLimit[:n_dof]), -2 * np.pi, model.lowerPositionLimit[:n_dof])
            q_max = np.where(np.isinf(model.upperPositionLimit[:n_dof]), 2 * np.pi, model.upperPositionLimit[:n_dof])
            
            result = (self.base_kinematic_module, q_min, q_max)
            self._vmodule_cache[ep_idx] = result
            return result
            
        episode_ends = self.replay_buffer.episode_ends[:]
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        end_idx = episode_ends[ep_idx]
        
        q_traj_full = self.replay_buffer['robot0_joint_pos'][start_idx:end_idx]
        T = len(q_traj_full)
        
        # Calculate finite difference velocity
        delta_q = np.zeros_like(q_traj_full)
        if T > 1:
            delta_q[:-1] = q_traj_full[1:] - q_traj_full[:-1]
            delta_q[-1] = delta_q[-2]
            
        delta_pose_actual = np.zeros((T, 6))
        
        # Worker-local data object prevents race conditions in PyTorch DataLoader
        local_base_data = self.base_kinematic_module.model.createData()
        
        for t in range(T):
            q_t = q_traj_full[t]
            pin.framesForwardKinematics(self.base_kinematic_module.model, local_base_data, q_t)
            J_base = pin.computeFrameJacobian(
                self.base_kinematic_module.model, local_base_data, q_t, 
                self.base_kinematic_module.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            delta_pose_actual[t] = J_base @ delta_q[t]
        
        q_min_v, q_max_v = self.virtual_sampler.sample_stage1_limits(
            np.min(q_traj_full, axis=0), np.max(q_traj_full, axis=0)
        )
        
        # Stage 2 morphological augmentation ensuring physical feasibility
        v_module, q_min_v, q_max_v = self.virtual_sampler.sample_stage2_module(
            q_traj_full, delta_pose_actual, q_min_v, q_max_v
        )
        
        result = (v_module, q_min_v, q_max_v)
        self._vmodule_cache[ep_idx] = result
        return result

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Dynamically expands the parent normalizer's ParameterDict to accommodate 
        the pre-normalized 42D k(q) tensor using an identity mapping.
        """
        normalizer = super().get_normalizer(mode=mode, **kwargs)
        obs_norm = normalizer['obs'] 
        obs_params = obs_norm.params_dict 
        
        device = obs_params['offset'].device
        dtype = obs_params['offset'].dtype
        
        identity_offset = torch.zeros(42, dtype=dtype, device=device)
        identity_scale = torch.ones(42, dtype=dtype, device=device)
        
        obs_params['offset'] = nn.Parameter(torch.cat([obs_params['offset'], identity_offset], dim=-1))
        obs_params['scale'] = nn.Parameter(torch.cat([obs_params['scale'], identity_scale], dim=-1))
        
        if 'input_stats' in obs_params:
            stats = obs_params['input_stats']
            for stat_key in stats.keys():
                pad = torch.zeros(42, dtype=dtype, device=device)
                if stat_key in ['std', 'max']:
                    pad = torch.ones(42, dtype=dtype, device=device)
                stats[stat_key] = nn.Parameter(torch.cat([stats[stat_key], pad], dim=-1))
                
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        
        raw_sample = self.sampler.sample_sequence(idx)
        q_window_np = raw_sample['robot0_joint_pos']
        
        idx_info = self.sampler.indices[idx]
        raw_start_idx = int(idx_info[0]) if isinstance(idx_info, (tuple, list, np.ndarray)) else int(idx_info)
        ep_idx = self.raw_to_ep[raw_start_idx]
        
        v_module, q_min_v, q_max_v = self._get_or_create_virtual_module(ep_idx)
        k_q_np = v_module.compute_k_q_with_custom_limits(q_window_np, q_min_v, q_max_v)
        
        k_q_tensor = torch.from_numpy(k_q_np).float()
        k_q_normalized = (k_q_tensor - self.k_mean) / self.k_std
        
        # Ultimate Concatenation: Append to the unified observation tensor
        item['obs'] = torch.cat([item['obs'], k_q_normalized], dim=-1)
        
        return item