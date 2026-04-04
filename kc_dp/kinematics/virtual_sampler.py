import numpy as np
import xml.etree.ElementTree as ET
from typing import Tuple

import pinocchio as pin
from kc_dp.kinematics.feature_extractor import (
    AnalyticKinematicModule,
    _build_model_from_xml_string,
)


class VirtualRobotSampler:
    """
    Virtual Robot Sampler for Hierarchical Kinematic Modulation (HKM).

    Handles Stage 1 (Virtual Joint Limits) and Stage 2 (Virtual Link Lengths).
    """

    def __init__(
        self,
        base_urdf_path: str,
        ee_frame_name: str,
        max_dof: int = 7,
        max_retries: int = 20,
        safety_margin: float = 0.05,
        violation_threshold: float = 0.05,
        stage1_min_range_deg: float = 60.0,
        stage1_max_range_deg: float = 300.0,
        stage2_log_sigma: float = 0.15,
        stage2_scale_min: float = 0.7,
        stage2_scale_max: float = 1.4,
    ):
        self.base_urdf_path = base_urdf_path
        self.ee_frame_name = ee_frame_name
        self.max_dof = max_dof
        self.max_retries = max_retries
        self.safety_margin = safety_margin
        self.violation_threshold = violation_threshold
        self.stage1_min_range_rad = np.radians(stage1_min_range_deg)
        self.stage1_max_range_rad = np.radians(stage1_max_range_deg)
        self.stage2_log_sigma = stage2_log_sigma
        self.stage2_scale_min = stage2_scale_min
        self.stage2_scale_max = stage2_scale_max

        with open(base_urdf_path, "r", encoding="utf-8") as f:
            self.base_urdf_str = f.read()

        base_model = _build_model_from_xml_string(self.base_urdf_str)

        if not base_model.existFrame(self.ee_frame_name):
            raise ValueError(f"Frame '{self.ee_frame_name}' not found in URDF.")

        ee_frame_id = base_model.getFrameId(self.ee_frame_name)

        current_joint_id = base_model.frames[ee_frame_id].parentJoint
        chain_joint_ids = []
        while current_joint_id > 0:
            chain_joint_ids.append(current_joint_id)
            current_joint_id = base_model.parents[current_joint_id]
        chain_joint_ids.reverse()

        self.arm_chain_joint_names = [
            base_model.names[j_id]
            for j_id in chain_joint_ids
            if base_model.joints[j_id].nq > 0
        ]

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def sample_stage1_limits(
        self, q_traj_min: np.ndarray, q_traj_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Sample virtual limits as supersets of demonstrated limits.
        # Keep a floor (for numerical stability in M*) and a ceiling
        # (to reduce extreme over-expansion tails).
        r_actual = np.maximum(q_traj_max - q_traj_min, 1e-6)

        q_min_v = np.zeros_like(q_traj_min)
        q_max_v = np.zeros_like(q_traj_max)

        for i in range(len(r_actual)):
            lower_bound_r = max(r_actual[i], self.stage1_min_range_rad)
            upper_bound_r = max(lower_bound_r, self.stage1_max_range_rad)
            r_v = np.random.uniform(lower_bound_r, upper_bound_r)
            slack = r_v - r_actual[i]
            slack_below = np.random.uniform(0.0, max(slack, 0.0))
            q_min_v[i] = q_traj_min[i] - slack_below
            q_max_v[i] = q_min_v[i] + r_v

        return q_min_v, q_max_v

    # ------------------------------------------------------------------
    # URDF modification
    # ------------------------------------------------------------------

    def _modify_urdf_in_memory(self, scales: np.ndarray) -> str:
        root = ET.fromstring(self.base_urdf_str)
        num_actuated = min(len(scales), len(self.arm_chain_joint_names))
        scale_map = {
            self.arm_chain_joint_names[i]: scales[i] for i in range(num_actuated)
        }
        for joint in root.findall("joint"):
            j_name = joint.get("name")
            if j_name in scale_map:
                origin = joint.find("origin")
                if origin is not None and "xyz" in origin.attrib:
                    xyz = list(map(float, origin.attrib["xyz"].split()))
                    s = scale_map[j_name]
                    origin.attrib["xyz"] = (
                        f"{xyz[0]*s:.6f} {xyz[1]*s:.6f} {xyz[2]*s:.6f}"
                    )
        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def check_action_feasibility(
        self,
        virtual_module: AnalyticKinematicModule,
        q_traj: np.ndarray,
        delta_pose: np.ndarray,
    ) -> bool:
        """
        Linearized tau condition using the virtual module's _pad_q to handle
        the arm-only → full-model DoF mismatch.
        """
        T = q_traj.shape[0]
        n_data = q_traj.shape[1]
        violations = 0

        for t in range(T):
            q_arm = q_traj[t]
            dp_t = delta_pose[t]

            # Pad arm-only q to full model config
            q_full = virtual_module._pad_q(q_arm)

            pin.framesForwardKinematics(
                virtual_module.model, virtual_module.data, q_full
            )
            J_full = pin.computeFrameJacobian(
                virtual_module.model,
                virtual_module.data,
                q_full,
                virtual_module.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            # Use only arm-chain columns
            n_eff = min(n_data, len(virtual_module._arm_v_indices))
            J = J_full[:, virtual_module._arm_v_indices[:n_eff]]

            J_norm = virtual_module.S_matrix @ J
            dp_norm = virtual_module.S_matrix @ dp_t

            J_frob = np.linalg.norm(J_norm, ord="fro")
            required = np.linalg.norm(dp_norm)

            if required > (J_frob + self.safety_margin):
                violations += 1

        return (violations / float(T)) < self.violation_threshold

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def sample_stage2_module(
        self,
        q_traj: np.ndarray,
        delta_pose: np.ndarray,
        q_min_v: np.ndarray,
        q_max_v: np.ndarray,
    ) -> Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]:
        n_dof = q_traj.shape[1]

        for _ in range(self.max_retries):
            scales = np.random.lognormal(mean=0.0, sigma=self.stage2_log_sigma, size=n_dof)
            scales = np.clip(scales, self.stage2_scale_min, self.stage2_scale_max)
            modified_urdf_str = self._modify_urdf_in_memory(scales)

            try:
                virtual_module = AnalyticKinematicModule(
                    urdf_xml=modified_urdf_str,
                    ee_frame_name=self.ee_frame_name,
                    max_dof=self.max_dof,
                )
                if self.check_action_feasibility(
                    virtual_module, q_traj, delta_pose
                ):
                    return virtual_module, q_min_v, q_max_v
            except Exception:
                pass

        fallback_module = AnalyticKinematicModule(
            urdf_xml=self.base_urdf_str,
            ee_frame_name=self.ee_frame_name,
            max_dof=self.max_dof,
        )
        return fallback_module, q_min_v, q_max_v
