from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class FeasibilityOracle:
    """
    Differentiable feasibility proxy used for guided denoising.

    This oracle is intentionally lightweight and robust:
    - If `pytorch_kinematics` is available, URDF metadata can be loaded for
      future IK/FK extensions.
    - Current scoring is fully differentiable in task-space trajectory and
      combines workspace and smoothness penalties.
    """

    def __init__(
        self,
        workspace_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        lambda_workspace: float = 5.0,
        lambda_smooth: float = 1.0,
        lambda_jerk: float = 0.5,
        urdf_path: Optional[str] = None,
        ee_frame_name: Optional[str] = None,
        device: str = "cpu",
    ):
        self.lambda_workspace = float(lambda_workspace)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_jerk = float(lambda_jerk)
        self.device = torch.device(device)

        if workspace_bounds is None:
            workspace_bounds = ((-0.8, -0.8, 0.0), (0.8, 0.8, 1.2))
        ws_min, ws_max = workspace_bounds
        self.ws_min = torch.tensor(ws_min, dtype=torch.float32, device=self.device)
        self.ws_max = torch.tensor(ws_max, dtype=torch.float32, device=self.device)

        self.urdf_path = urdf_path
        self.ee_frame_name = ee_frame_name
        self.pk_chain = None
        self._try_load_pk_chain()

    def _try_load_pk_chain(self):
        if (self.urdf_path is None) or (self.ee_frame_name is None):
            return
        try:
            import pytorch_kinematics as pk  # type: ignore

            with open(self.urdf_path, "r", encoding="utf-8") as f:
                urdf_xml = f.read()
            self.pk_chain = pk.build_serial_chain_from_urdf(urdf_xml, self.ee_frame_name)
        except Exception:
            self.pk_chain = None

    def log_feasibility(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        trajectory: (B, H, D), where first 3 dims are EE xyz.
        returns: (B,) differentiable log feasibility.
        """
        if trajectory.ndim != 3:
            raise ValueError(f"trajectory must be 3D [B,H,D], got shape={tuple(trajectory.shape)}")
        if trajectory.shape[-1] < 3:
            raise ValueError("trajectory last dim must be >=3 with xyz at [:3].")

        pos = trajectory[..., :3]

        ws_under = F.relu(self.ws_min.view(1, 1, 3) - pos)
        ws_over = F.relu(pos - self.ws_max.view(1, 1, 3))
        c_workspace = (ws_under + ws_over).pow(2).mean(dim=(1, 2))

        if pos.shape[1] > 1:
            vel = pos[:, 1:] - pos[:, :-1]
            c_smooth = vel.pow(2).mean(dim=(1, 2))
        else:
            c_smooth = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)

        if pos.shape[1] > 2:
            acc = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]
            c_jerk = acc.pow(2).mean(dim=(1, 2))
        else:
            c_jerk = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)

        total_cost = (
            self.lambda_workspace * c_workspace
            + self.lambda_smooth * c_smooth
            + self.lambda_jerk * c_jerk
        )
        return -total_cost

    def score_dict(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        logf = self.log_feasibility(trajectory)
        return {
            "log_feasibility": logf,
            "feasibility": torch.exp(logf),
        }
