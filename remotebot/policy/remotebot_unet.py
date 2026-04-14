from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy


class RemoteBotDiffusionUnetLowdimPolicy(DiffusionUnetLowdimPolicy):
    """
    Task-space Diffusion Policy with optional feasibility-guided denoising.
    """

    def __init__(
        self,
        *args,
        guidance_weight_base: float = 0.0,
        guidance_power: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.guidance_weight_base = float(guidance_weight_base)
        self.guidance_power = float(guidance_power)
        self.feasibility_oracle = None

    def set_feasibility_oracle(self, oracle):
        self.feasibility_oracle = oracle

    def _guidance_weight(self, t_scalar: int) -> float:
        max_t = float(max(1, self.noise_scheduler.config.num_train_timesteps - 1))
        ratio = float(t_scalar) / max_t
        return self.guidance_weight_base * ((1.0 - ratio) ** self.guidance_power)

    def conditional_sample(self,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        use_guidance = (
            self.feasibility_oracle is not None
            and self.guidance_weight_base > 0.0
        )

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            if not use_guidance:
                model_output = model(
                    trajectory,
                    t,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
                trajectory = scheduler.step(
                    model_output,
                    t,
                    trajectory,
                    generator=generator,
                    **kwargs,
                ).prev_sample
                continue

            t_scalar = int(t.item()) if hasattr(t, "item") else int(t)
            weight = self._guidance_weight(t_scalar)
            if weight <= 0.0:
                model_output = model(
                    trajectory,
                    t,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
                trajectory = scheduler.step(
                    model_output,
                    t,
                    trajectory,
                    generator=generator,
                    **kwargs,
                ).prev_sample
                continue

            traj_req = trajectory.detach().requires_grad_(True)
            with torch.enable_grad():
                model_output = model(
                    traj_req,
                    t,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
                step_out = scheduler.step(
                    model_output,
                    t,
                    traj_req,
                    generator=generator,
                    **kwargs,
                )
                pred_x0 = getattr(step_out, "pred_original_sample", None)
                if pred_x0 is None:
                    pred_x0 = traj_req - model_output
                logf = self.feasibility_oracle.log_feasibility(pred_x0[..., : self.action_dim])
                grad = torch.autograd.grad(logf.sum(), traj_req, retain_graph=False)[0]

            guided_eps = model_output - (weight * grad)
            trajectory = scheduler.step(
                guided_eps,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

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
            global_cond = obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1 if self.oa_step_convention else To
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
        return loss.mean()

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().predict_action(obs_dict)
