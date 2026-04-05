from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy


class KCDiffusionUnetLowdimPolicy(DiffusionUnetLowdimPolicy):
	"""
	KC policy variant:
	- keeps full observation normalization at input obs_dim
	- compresses k(+mask) sub-vector into a small embedding
	- uses [base_obs_flat, k_embed] as global condition

	This leverages the same FiLM-conditioned U-Net path as upstream
	(global_cond -> cond_encoder), while reducing k dominance.
	"""

	def __init__(
		self,
		*args,
		base_obs_dim: int = 23,
		k_embed_dim: int = 16,
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.base_obs_dim = int(base_obs_dim)
		self.k_embed_dim = int(k_embed_dim)

		# Expect obs_as_global_cond in this project. Still keep graceful fallback.
		k_cond_dim = max(0, self.obs_dim - self.base_obs_dim)
		self.k_cond_dim = k_cond_dim

		if k_cond_dim > 0:
			self.k_encoder = nn.Sequential(
				nn.Linear(k_cond_dim * self.n_obs_steps, 64),
				nn.Mish(),
				nn.Linear(64, self.k_embed_dim),
			)
		else:
			self.k_encoder = None

	def _build_global_cond(self, obs: torch.Tensor) -> torch.Tensor:
		"""
		obs: (B, T, Do) normalized obs
		returns global_cond: (B, base_obs_dim*T + k_embed_dim)
		"""
		B = obs.shape[0]
		To = self.n_obs_steps
		obs_to = obs[:, :To, :]

		if (self.k_encoder is None) or (self.k_cond_dim <= 0) or (self.base_obs_dim >= self.obs_dim):
			return obs_to.reshape(B, -1)

		base = obs_to[:, :, : self.base_obs_dim].reshape(B, -1)
		k_part = obs_to[:, :, self.base_obs_dim :].reshape(B, -1)
		k_embed = self.k_encoder(k_part)
		return torch.cat([base, k_embed], dim=-1)

	# ========= inference  =========
	def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		assert 'obs' in obs_dict
		assert 'past_action' not in obs_dict
		nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
		B, _, Do = nobs.shape
		To = self.n_obs_steps
		assert Do == self.obs_dim
		T = self.horizon
		Da = self.action_dim

		device = self.device
		dtype = self.dtype

		local_cond = None
		global_cond = None
		if self.obs_as_local_cond:
			local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
			local_cond[:, :To] = nobs[:, :To]
			shape = (B, T, Da)
			cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
			cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
		elif self.obs_as_global_cond:
			global_cond = self._build_global_cond(nobs)
			shape = (B, T, Da)
			if self.pred_action_steps_only:
				shape = (B, self.n_action_steps, Da)
			cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
			cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
		else:
			shape = (B, T, Da + Do)
			cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
			cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
			cond_data[:, :To, Da:] = nobs[:, :To]
			cond_mask[:, :To, Da:] = True

		nsample = self.conditional_sample(
			cond_data,
			cond_mask,
			local_cond=local_cond,
			global_cond=global_cond,
			**self.kwargs,
		)

		naction_pred = nsample[..., :Da]
		action_pred = self.normalizer['action'].unnormalize(naction_pred)

		if self.pred_action_steps_only:
			action = action_pred
		else:
			start = To
			if self.oa_step_convention:
				start = To - 1
			end = start + self.n_action_steps
			action = action_pred[:, start:end]

		result = {
			'action': action,
			'action_pred': action_pred,
		}
		if not (self.obs_as_local_cond or self.obs_as_global_cond):
			nobs_pred = nsample[..., Da:]
			obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
			action_obs_pred = obs_pred[:, start:end]
			result['action_obs_pred'] = action_obs_pred
			result['obs_pred'] = obs_pred
		return result

	# ========= training =========
	def compute_loss(self, batch):
		assert 'valid_mask' not in batch
		nbatch = self.normalizer.normalize(batch)
		obs = nbatch['obs']
		action = nbatch['action']

		local_cond = None
		global_cond = None
		trajectory = action
		if self.obs_as_local_cond:
			local_cond = obs
			local_cond[:, self.n_obs_steps:, :] = 0
		elif self.obs_as_global_cond:
			global_cond = self._build_global_cond(obs)
			if self.pred_action_steps_only:
				To = self.n_obs_steps
				start = To
				if self.oa_step_convention:
					start = To - 1
				end = start + self.n_action_steps
				trajectory = action[:, start:end]
		else:
			trajectory = torch.cat([action, obs], dim=-1)

		if self.pred_action_steps_only:
			condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
		else:
			condition_mask = self.mask_generator(trajectory.shape)

		noise = torch.randn(trajectory.shape, device=trajectory.device)
		bsz = trajectory.shape[0]
		timesteps = torch.randint(
			0,
			self.noise_scheduler.config.num_train_timesteps,
			(bsz,),
			device=trajectory.device,
		).long()
		noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

		loss_mask = ~condition_mask
		noisy_trajectory[condition_mask] = trajectory[condition_mask]

		pred = self.model(
			noisy_trajectory,
			timesteps,
			local_cond=local_cond,
			global_cond=global_cond,
		)

		pred_type = self.noise_scheduler.config.prediction_type
		if pred_type == 'epsilon':
			target = noise
		elif pred_type == 'sample':
			target = trajectory
		else:
			raise ValueError(f"Unsupported prediction type {pred_type}")

		loss = F.mse_loss(pred, target, reduction='none')
		loss = loss * loss_mask.type(loss.dtype)
		loss = reduce(loss, 'b ... -> b (...)', 'mean')
		loss = loss.mean()
		return loss

