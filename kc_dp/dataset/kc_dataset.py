import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

import pinocchio as pin

# Import the base dataset from the original diffusion_policy submodule
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.common.sampler import SequenceSampler

from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule
from kc_dp.kinematics.virtual_sampler import VirtualRobotSampler

class HKMLowdimDataset(RobomimicReplayLowdimDataset):
    """
    Hierarchical Kinematic Modulation (HKM) Dataset Wrapper.
    
    Inherits from the baseline RobomimicReplayLowdimDataset. It intercepts the 
    data loading pipeline to dynamically generate virtual robots (Stage 1 & 2), 
    compute the 42-dimensional kinematic feature vector k(q), and inject it into 
    the observation dictionary under the key 'robot0_k_q'.
    """

    def __init__(self,
                 *args,
                 virtual_sampler: Optional[VirtualRobotSampler] = None,
                 base_kinematic_module: Optional[AnalyticKinematicModule] = None,
                 k_mean: Optional[np.ndarray] = None,
                 k_std: Optional[np.ndarray] = None,
                 **kwargs):
        """
        Initializes the HKM dataset.
        
        Args:
            virtual_sampler: Instance of VirtualRobotSampler for topology augmentation.
            base_kinematic_module: Fallback module used if virtual_sampler is None, 
                                   and utilized to compute the required task-space velocity.
            k_mean: Precomputed mean of k(q) across the dataset for normalization (42D).
            k_std: Precomputed standard deviation of k(q) for normalization (42D).
            *args, **kwargs: Arguments passed to the parent RobomimicReplayLowdimDataset.
        """
        super().__init__(*args, **kwargs)
        
        self.virtual_sampler = virtual_sampler
        self.base_kinematic_module = base_kinematic_module
        
        if self.virtual_sampler is None and self.base_kinematic_module is None:
            raise ValueError("Either virtual_sampler or base_kinematic_module must be provided.")
            
        # Initialize normalization statistics (defaulting to identity if not provided)
        self.k_mean = torch.from_numpy(k_mean).float() if k_mean is not None else torch.zeros(42)
        self.k_std = torch.from_numpy(k_std).float() if k_std is not None else torch.ones(42)
        # Prevent zero-division instability during normalization
        self.k_std = torch.clamp(self.k_std, min=1e-6)
        
        # Cache for virtual modules to prevent redundant URDF parsing per step.
        # Structure: {episode_index: (virtual_module, q_min_v, q_max_v)}
        self._vmodule_cache: Dict[int, Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]] = {}
        
        # Precompute the O(1) lookup mapping from raw timestep to episode index
        self._build_episode_mapping()

    def _build_episode_mapping(self):
        """
        Constructs an array mapping every raw timestep index in the replay buffer 
        to its corresponding episode index.
        """
        episode_ends = self.replay_buffer.episode_ends[:]
        total_raw_steps = episode_ends[-1]
        
        self.raw_to_ep = np.zeros(total_raw_steps, dtype=int)
        starts = np.insert(episode_ends[:-1], 0, 0)
        
        for ep_idx, (start, end) in enumerate(zip(starts, episode_ends)):
            self.raw_to_ep[start:end] = ep_idx
            
    def clear_epoch_cache(self):
        """
        Clears the virtual robot cache. Must be explicitly called at the boundary 
        of each training epoch to ensure a novel set of virtual morphologies is sampled.
        """
        self._vmodule_cache.clear()

    def _get_or_create_virtual_module(self, ep_idx: int) -> Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]:
        """
        Retrieves the cached virtual robot for the given episode, or dynamically 
        samples a new one if it does not exist in the cache.
        
        Args:
            ep_idx: The index of the current episode.
            
        Returns:
            A tuple containing the AnalyticKinematicModule, virtual minimum limits, 
            and virtual maximum limits.
        """
        # Return cached module if it already exists for this epoch
        if ep_idx in self._vmodule_cache:
            return self._vmodule_cache[ep_idx]
            
        if self.virtual_sampler is None:
            # Fallback: Utilize the base robot geometry and its actual physical limits
            model = self.base_kinematic_module.model
            n_dof = self.replay_buffer['robot0_joint_pos'].shape[1]
            q_min = model.lowerPositionLimit[:n_dof]
            q_max = model.upperPositionLimit[:n_dof]
            
            # Sanity check: Handle undefined limits (inf) by defaulting to [-2pi, 2pi]
            q_min = np.where(np.isinf(q_min), -2 * np.pi, q_min)
            q_max = np.where(np.isinf(q_max), 2 * np.pi, q_max)
            
            result = (self.base_kinematic_module, q_min, q_max)
            self._vmodule_cache[ep_idx] = result
            return result
            
        # 1. Extract the full trajectory for the current episode
        episode_ends = self.replay_buffer.episode_ends[:]
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        end_idx = episode_ends[ep_idx]
        
        q_traj_full = self.replay_buffer['robot0_joint_pos'][start_idx:end_idx]
        T = len(q_traj_full)
        
        # 2. Accurately compute the required task-space velocity (delta_pose)
        # Handle T=1 edge case to prevent IndexError during finite difference
        delta_q = np.zeros_like(q_traj_full)
        if T > 1:
            delta_q[:-1] = q_traj_full[1:] - q_traj_full[:-1]
            delta_q[-1] = delta_q[-2] # Forward fill the final step
            
        delta_pose_actual = np.zeros((T, 6))
        
        # Create a worker-local data buffer to guarantee multiprocessing safety.
        # This prevents race conditions and pickling errors in PyTorch spawn contexts.
        local_base_data = self.base_kinematic_module.model.createData()
        
        for t in range(T):
            q_t = q_traj_full[t]
            dq_t = delta_q[t]
            
            pin.framesForwardKinematics(self.base_kinematic_module.model, local_base_data, q_t)
            J_base = pin.computeFrameJacobian(
                self.base_kinematic_module.model,
                local_base_data,
                q_t,
                self.base_kinematic_module.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            # Approximate the spatial velocity: v = J * dq
            delta_pose_actual[t] = J_base @ dq_t
        
        q_traj_min = np.min(q_traj_full, axis=0)
        q_traj_max = np.max(q_traj_full, axis=0)
        
        # Stage 1: Sample virtual joint limits
        q_min_v, q_max_v = self.virtual_sampler.sample_stage1_limits(q_traj_min, q_traj_max)
        
        # Stage 2: Sample virtual topology ensuring the required delta_pose is achievable
        v_module, q_min_v, q_max_v = self.virtual_sampler.sample_stage2_module(
            q_traj_full, delta_pose_actual, q_min_v, q_max_v
        )
        
        result = (v_module, q_min_v, q_max_v)
        self._vmodule_cache[ep_idx] = result
        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Intercepts the standard dataset item retrieval to dynamically compute 
        and inject the k(q) kinematic feature vector.
        """
        # 1. Retrieve the base item dictionary from the parent class
        item = super().__getitem__(idx)
        
        # 2. Determine the episode index corresponding to the current sample
        idx_info = self.sampler.indices[idx]
        
        # Robust extraction of the raw buffer start index (handling both integer and tuple formats)
        raw_start_idx = int(idx_info[0]) if isinstance(idx_info, (tuple, list, np.ndarray)) else int(idx_info)
        ep_idx = self.raw_to_ep[raw_start_idx]
        
        # 3. Retrieve or generate the virtual robot configuration for this episode
        v_module, q_min_v, q_max_v = self._get_or_create_virtual_module(ep_idx)
        
        # 4. Extract the joint position sequence from the current observation window
        # The shape is expected to be (T_obs, n_dof)
        q_window_tensor = item['obs']['robot0_joint_pos']
        q_window_np = q_window_tensor.numpy()
        
        # 5. Compute the 42D kinematic feature vector utilizing the virtual module
        k_q_np = v_module.compute_k_q_with_custom_limits(q_window_np, q_min_v, q_max_v)
        k_q_tensor = torch.from_numpy(k_q_np).float()
        
        # 6. Normalize the feature vector using dataset statistics
        k_q_normalized = (k_q_tensor - self.k_mean) / self.k_std
        
        # 7. Inject the finalized feature into the observation dictionary
        item['obs']['robot0_k_q'] = k_q_normalized
        
        return item
