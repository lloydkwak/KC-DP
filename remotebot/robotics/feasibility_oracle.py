from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class FeasibilityOracle:
    """
    Feasibility oracle for guided denoising.

    If URDF + EE frame are provided and pytorch-kinematics is available,
    this computes a batched damped-least-squares IK proxy and uses:
      - FK residual (task-space target mismatch)
      - joint-limit violation penalty
      - joint smoothness/jerk penalty

    The IK solve is intentionally run in `no_grad` for stability, while the
    final residual term remains differentiable w.r.t. target trajectory.

        NOTE (gradient semantics):
        - `c_ik` and `c_workspace` provide direct gradients to denoising guidance.
        - `c_joint_limit` and joint `c_smooth/c_jerk` are computed from IK output
            solved under `no_grad`, so they mainly act as monitoring/regularization
            indicators in the current implementation.
        - For full end-to-end joint-cost gradients, implicit differentiation over
            the IK solver would be required.
    """

    def __init__(
        self,
        workspace_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        lambda_workspace: float = 5.0,
        lambda_ik_residual: float = 10.0,
        lambda_joint_limit: float = 2.0,
        lambda_smooth: float = 0.2,
        lambda_jerk: float = 0.05,
        urdf_path: Optional[str] = None,
        ee_frame_name: Optional[str] = None,
        num_ik_iters: int = 12,
        ik_step_size: float = 0.7,
        ik_damping: float = 1e-3,
        device: str = "cpu",
    ):
        self.lambda_workspace = float(lambda_workspace)
        self.lambda_ik_residual = float(lambda_ik_residual)
        self.lambda_joint_limit = float(lambda_joint_limit)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_jerk = float(lambda_jerk)

        self.num_ik_iters = int(num_ik_iters)
        self.ik_step_size = float(ik_step_size)
        self.ik_damping = float(ik_damping)

        self.device = torch.device(device)

        if workspace_bounds is None:
            workspace_bounds = ((-0.8, -0.8, 0.0), (0.8, 0.8, 1.2))
        ws_min, ws_max = workspace_bounds
        self.ws_min = torch.tensor(ws_min, dtype=torch.float32, device=self.device)
        self.ws_max = torch.tensor(ws_max, dtype=torch.float32, device=self.device)

        self.urdf_path = urdf_path
        self.ee_frame_name = ee_frame_name

        self.pk_chain = None
        self.n_dof = 0
        self.q_lower = None
        self.q_upper = None
        self._joint_faults: Dict[int, float] = {}

        self._try_load_pk_chain()

    def _try_load_pk_chain(self):
        if (self.urdf_path is None) or (self.ee_frame_name is None):
            return
        try:
            import pytorch_kinematics as pk  # type: ignore

            with open(self.urdf_path, "r", encoding="utf-8") as f:
                urdf_xml = f.read()

            chain = pk.build_serial_chain_from_urdf(urdf_xml, self.ee_frame_name)
            self.pk_chain = chain.to(device=self.device, dtype=torch.float32)
            self.n_dof = int(self.pk_chain.get_joint_parameter_names().__len__())

            q_lower = torch.full((self.n_dof,), -3.1416, device=self.device, dtype=torch.float32)
            q_upper = torch.full((self.n_dof,), 3.1416, device=self.device, dtype=torch.float32)
            try:
                limits = self.pk_chain.get_joint_limits()
                if isinstance(limits, tuple) and len(limits) == 2:
                    ql, qu = limits
                    ql = torch.as_tensor(ql, dtype=torch.float32, device=self.device).reshape(-1)
                    qu = torch.as_tensor(qu, dtype=torch.float32, device=self.device).reshape(-1)
                    if ql.numel() == self.n_dof and qu.numel() == self.n_dof:
                        q_lower = ql
                        q_upper = qu
            except Exception:
                pass

            self.q_lower = q_lower
            self.q_upper = q_upper
        except Exception:
            self.pk_chain = None
            self.n_dof = 0
            self.q_lower = None
            self.q_upper = None

    def set_joint_fault(self, joint_idx: int, locked_value: float) -> None:
        if joint_idx < 0:
            raise ValueError("joint_idx must be >= 0")
        self._joint_faults[int(joint_idx)] = float(locked_value)

    def clear_faults(self) -> None:
        self._joint_faults.clear()

    def _forward_kinematics_pos(self, q: torch.Tensor) -> torch.Tensor:
        tf = self.pk_chain.forward_kinematics(q)
        if hasattr(tf, "get_matrix"):
            mat = tf.get_matrix()
        elif isinstance(tf, dict):
            first_key = next(iter(tf))
            obj = tf[first_key]
            mat = obj.get_matrix() if hasattr(obj, "get_matrix") else obj
        else:
            mat = tf
        return mat[..., :3, 3]

    def _jacobian_pos(self, q: torch.Tensor) -> torch.Tensor:
        jac = self.pk_chain.jacobian(q)
        if jac.shape[-2] >= 3:
            return jac[..., :3, :]
        raise RuntimeError(f"Unexpected jacobian shape: {tuple(jac.shape)}")

    def _apply_joint_faults(self, q: torch.Tensor) -> torch.Tensor:
        if not self._joint_faults:
            return q
        q_locked = q
        for idx, value in self._joint_faults.items():
            if 0 <= idx < q_locked.shape[-1]:
                q_locked[:, idx] = value
        return q_locked

    def _solve_ik_batched(self, target_pos: torch.Tensor) -> torch.Tensor:
        """
        target_pos: (N,3)
        returns q_seq: (N,DoF)
        """
        n = target_pos.shape[0]
        q = torch.zeros((n, self.n_dof), device=target_pos.device, dtype=target_pos.dtype)

        with torch.no_grad():
            for _ in range(self.num_ik_iters):
                q = self._apply_joint_faults(q)
                if self.q_lower is not None and self.q_upper is not None:
                    q = torch.max(torch.min(q, self.q_upper.view(1, -1)), self.q_lower.view(1, -1))

                cur_pos = self._forward_kinematics_pos(q)
                err = (target_pos - cur_pos).unsqueeze(-1)
                j_pos = self._jacobian_pos(q)
                j_t = j_pos.transpose(-1, -2)

                eye = torch.eye(3, device=q.device, dtype=q.dtype).view(1, 3, 3).expand(n, -1, -1)
                jj_t = j_pos @ j_t
                inv = torch.linalg.inv(jj_t + self.ik_damping * eye)
                dq = (j_t @ (inv @ err)).squeeze(-1)
                q = q + self.ik_step_size * dq

        q = self._apply_joint_faults(q)
        if self.q_lower is not None and self.q_upper is not None:
            q = torch.max(torch.min(q, self.q_upper.view(1, -1)), self.q_lower.view(1, -1))
        return q

    def _workspace_cost(self, pos: torch.Tensor) -> torch.Tensor:
        ws_under = F.relu(self.ws_min.view(1, 1, 3) - pos)
        ws_over = F.relu(pos - self.ws_max.view(1, 1, 3))
        return (ws_under + ws_over).pow(2).mean(dim=(1, 2))

    def log_feasibility(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        trajectory: (B,H,D), where first 3 dims are target EE xyz.
        returns: (B,) differentiable log feasibility.
        """
        if trajectory.ndim != 3:
            raise ValueError(f"trajectory must be 3D [B,H,D], got shape={tuple(trajectory.shape)}")
        if trajectory.shape[-1] < 3:
            raise ValueError("trajectory last dim must be >=3 with xyz at [:3].")

        pos = trajectory[..., :3]
        batch_size, horizon, _ = pos.shape

        c_workspace = self._workspace_cost(pos)

        if self.pk_chain is None or self.n_dof <= 0:
            if horizon > 1:
                vel = pos[:, 1:] - pos[:, :-1]
                c_smooth = vel.pow(2).mean(dim=(1, 2))
            else:
                c_smooth = torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)

            if horizon > 2:
                acc = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]
                c_jerk = acc.pow(2).mean(dim=(1, 2))
            else:
                c_jerk = torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)

            total_cost = (
                self.lambda_workspace * c_workspace
                + self.lambda_smooth * c_smooth
                + self.lambda_jerk * c_jerk
            )
            return -total_cost

        target_pos = pos.reshape(-1, 3)
        q_flat = self._solve_ik_batched(target_pos)

        fk_pos = self._forward_kinematics_pos(q_flat)
        c_ik = (fk_pos - target_pos).pow(2).sum(dim=-1).view(batch_size, horizon).mean(dim=1)

        q_seq = q_flat.view(batch_size, horizon, self.n_dof)

        if (self.q_lower is not None) and (self.q_upper is not None):
            ql = self.q_lower.view(1, 1, -1)
            qu = self.q_upper.view(1, 1, -1)
            v_low = F.relu(ql - q_seq)
            v_high = F.relu(q_seq - qu)
            c_joint_limit = (v_low + v_high).pow(2).mean(dim=(1, 2))
        else:
            c_joint_limit = torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)

        if horizon > 1:
            q_vel = q_seq[:, 1:] - q_seq[:, :-1]
            c_smooth = q_vel.pow(2).mean(dim=(1, 2))
        else:
            c_smooth = torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)

        if horizon > 2:
            q_acc = q_seq[:, 2:] - 2 * q_seq[:, 1:-1] + q_seq[:, :-2]
            c_jerk = q_acc.pow(2).mean(dim=(1, 2))
        else:
            c_jerk = torch.zeros(batch_size, device=pos.device, dtype=pos.dtype)

        total_cost = (
            self.lambda_workspace * c_workspace
            + self.lambda_ik_residual * c_ik
            + self.lambda_joint_limit * c_joint_limit
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
