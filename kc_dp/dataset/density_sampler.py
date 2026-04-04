import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from typing import Any, Union

from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule

class DensityWeightedSampler:
    """
    Stage 3: k-space Density-based Episode Sampler for HKM.
    
    This class analyzes the kinematic feature space (k-space) of all episodes in a 
    Zarr/HDF5 ReplayBuffer. It utilizes a pure PyTorch k-Nearest Neighbors (kNN) 
    approach to assign higher sampling probabilities to episodes located in sparse 
    regions, mitigating data imbalance in cross-embodiment learning.
    """

    def __init__(self, 
                 dataset: Any, 
                 kinematic_module: AnalyticKinematicModule, 
                 k_neighbors: int = 5, 
                 temperature: float = 1.0):
        """
        Initializes the density-weighted sampler and computes sampling weights.
        """
        self.dataset = dataset
        self.kinematic_module = kinematic_module
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        
        self.episode_means = self._compute_episode_means()
        self.episode_weights = self._compute_gap_weights()

    def _compute_episode_means(self) -> np.ndarray:
        """
        Computes the average 42D k(q) feature vector for every episode.
        """
        # Standard access path for diffusion_policy's ReplayBuffer
        q_all = self.dataset.replay_buffer['robot0_joint_pos'][:]
        episode_ends = self.dataset.replay_buffer.episode_ends[:]
        starts = np.insert(episode_ends[:-1], 0, 0)
        
        n_dof = q_all.shape[1]
        
        # Robust arm-chain joint limit extraction from Pinocchio model
        arm_idx = self.kinematic_module._arm_q_indices[:n_dof]
        q_min_base = self.kinematic_module.model.lowerPositionLimit[arm_idx]
        q_max_base = self.kinematic_module.model.upperPositionLimit[arm_idx]
        
        # Sanity check: Handle undefined limits (inf) by defaulting to [-2pi, 2pi]
        q_min_base = np.where(np.isinf(q_min_base), -2 * np.pi, q_min_base)
        q_max_base = np.where(np.isinf(q_max_base), 2 * np.pi, q_max_base)
        
        episode_means = []
        for ep_idx, (start, end) in enumerate(zip(starts, episode_ends)):
            q_traj = q_all[start:end]

            # In virtual mode, use the same virtual module/limits as training feature generation.
            if getattr(self.dataset, 'k_feature_mode', 'base_physical') == 'virtual' \
                    and hasattr(self.dataset, '_get_or_create_virtual_module'):
                v_module, q_min_v, q_max_v = self.dataset._get_or_create_virtual_module(ep_idx)
                k_q_traj = v_module.compute_k_q_with_custom_limits(q_traj, q_min_v, q_max_v)
            else:
                # base_physical mode
                k_q_traj = self.kinematic_module.compute_k_q_with_custom_limits(
                    q_traj, q_min_base, q_max_base
                )
            
            mean_k_q = np.mean(k_q_traj, axis=0)
            episode_means.append(mean_k_q)
            
        return np.vstack(episode_means)

    def _compute_gap_weights(self) -> np.ndarray:
        """
        Estimates k-space density using PyTorch-based kNN.
        """
        n_episodes = len(self.episode_means)
        if n_episodes <= self.k_neighbors:
            return np.ones(n_episodes) / n_episodes
            
        features_tensor = torch.from_numpy(self.episode_means).float()
        dist_matrix = torch.cdist(features_tensor, features_tensor, p=2.0)
        
        # Select k+1 neighbors and discard the self-distance (index 0)
        topk_dists, _ = torch.topk(dist_matrix, k=self.k_neighbors + 1, dim=1, largest=False)
        neighbor_dists = topk_dists[:, 1:]
        mean_distances = neighbor_dists.mean(dim=1).numpy()
        
        scaled_gaps = mean_distances * self.temperature
        exp_gaps = np.exp(scaled_gaps - np.max(scaled_gaps))
        probabilities = exp_gaps / np.sum(exp_gaps)
        
        return probabilities

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Constructs a PyTorch WeightedRandomSampler by mapping episode-level weights
        to sequence-window indices.
        """
        dataset_len = len(self.dataset)
        step_weights = np.zeros(dataset_len, dtype=np.float32)
        
        episode_ends = self.dataset.replay_buffer.episode_ends[:]
        total_raw_steps = episode_ends[-1]
        
        # 1. Reverse lookup: raw timestep index -> episode index
        raw_to_ep = np.zeros(total_raw_steps, dtype=int)
        starts = np.insert(episode_ends[:-1], 0, 0)
        for ep_idx, (start, end) in enumerate(zip(starts, episode_ends)):
            raw_to_ep[start:end] = ep_idx
            
        # 2. Map dataset window indices to corresponding episode weights
        # Handle SequenceSampler.indices which is a list of tuples:
        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        if hasattr(self.dataset, 'sampler') and hasattr(self.dataset.sampler, 'indices'):
            valid_indices = self.dataset.sampler.indices
        else:
            raise AttributeError("Dataset sampler lacks 'indices'. Cannot perform window mapping.")
            
        for dataset_idx in range(dataset_len):
            idx_info = valid_indices[dataset_idx]
            
            # Robust extraction of raw buffer start index from tuple or integer
            if isinstance(idx_info, (tuple, list, np.ndarray)):
                raw_idx = int(idx_info[0])
            else:
                raw_idx = int(idx_info)
                
            ep_idx = raw_to_ep[raw_idx]
            step_weights[dataset_idx] = self.episode_weights[ep_idx]
            
        weights_tensor = torch.from_numpy(step_weights).double()
        
        return WeightedRandomSampler(
            weights=weights_tensor, 
            num_samples=dataset_len, 
            replacement=True
        )
