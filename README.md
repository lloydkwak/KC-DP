# RemoteBot

Task-space trajectory diversification and feasibility-guided diffusion for cross-embodiment robot manipulation.

## What this repo contains

- **Offline augmentation**: TSTD pipeline to generate diverse task-space trajectories
- **Policy training**: diffusion policy in task space
- **Inference support**: feasibility oracle and guided denoising hooks
- **Cross-robot evaluation**: Robosuite/Robomimic evaluation entrypoint

## Repository layout

- `remotebot/tstd/` — TSTD modules (keypoints, grasp, approach, path)
- `remotebot/dataset/` — task-space dataset wrapper
- `remotebot/policy/` — RemoteBot diffusion policy
- `remotebot/robotics/` — feasibility/oracle and robot utilities
- `remotebot/training/` — training workspace
- `scripts/augment_tstd.py` — offline dataset augmentation
- `scripts/train_remotebot.py` — training entrypoint
- `scripts/eval_remotebot.py` — evaluation entrypoint
- `configs/train_remotebot_dp.yaml` — main training config

## Quick start

1. Install dependencies from `requirements.txt`.
2. Prepare a Robomimic low-dim HDF5 dataset.
3. Run in order:
	- `augment_tstd.py` (optional but recommended)
	- `train_remotebot.py`
	- `eval_remotebot.py`

## Notes

- The codebase was cleaned from legacy `k(q)`-conditioned KC-DP paths.
- Current package root is `remotebot`.

## Practical limitations (current scope)

- **Oracle gradient semantics**: feasibility guidance gradients are primarily driven by
	workspace and IK residual terms. Joint-limit and joint smoothness terms are tracked
	as monitoring costs in the current stable implementation.
- **TSTD object-state assumption**: augmented trajectories keep non-EE observation keys
	(e.g., `obs/object`) from source demos. This assumption is most reliable for pick-place
	free-space phases and may be less accurate for strongly contact-rich tasks.

