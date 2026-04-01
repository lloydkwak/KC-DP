import os
import tempfile
import numpy as np
import pinocchio as pin


def _build_model_from_xml_string(urdf_xml: str) -> pin.Model:
    """
    Builds a Pinocchio Model from a URDF XML string.
    Falls back to tempfile when pin.buildModelFromXML is unavailable.
    """
    if hasattr(pin, "buildModelFromXML"):
        return pin.buildModelFromXML(urdf_xml)

    fd, tmp_path = tempfile.mkstemp(suffix=".urdf", prefix="hkm_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(urdf_xml)
        model = pin.buildModelFromUrdf(tmp_path)
    finally:
        os.remove(tmp_path)
    return model


class AnalyticKinematicModule:
    """
    Computes the 42-dimensional k(q) kinematic feature vector.

    k(q) = [M*_vech(21), sin_enc(7), cos_enc(7), p_norm(3), quat(4)] = 42D

    Handles the common case where demo data contains only arm joints
    (e.g. 7D for Panda) while the URDF model includes extra joints like
    gripper fingers (model.nq = 9).  Arm-chain joint indices are extracted
    automatically from the EE kinematic chain.
    """

    def __init__(self, urdf_path=None, urdf_xml=None, ee_frame_name="", max_dof=7):
        # ── 1. Load robot model ──
        if urdf_xml is not None:
            self.model = _build_model_from_xml_string(urdf_xml)
        elif urdf_path is not None:
            self.model = pin.buildModelFromUrdf(urdf_path)
        else:
            raise ValueError("Either urdf_path or urdf_xml must be provided.")

        self.data = self.model.createData()

        # ── 2. EE frame ──
        if not self.model.existFrame(ee_frame_name):
            raise ValueError(f"Frame '{ee_frame_name}' not found in the URDF.")
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)

        # ── 3. Identify arm-chain q-vector indices ──
        parent_jid = self.model.frames[self.ee_frame_id].parentJoint
        chain_jids = []
        while parent_jid > 0:
            chain_jids.append(parent_jid)
            parent_jid = self.model.parents[parent_jid]
        chain_jids.reverse()

        self._arm_q_indices = []
        for jid in chain_jids:
            jnt = self.model.joints[jid]
            if jnt.nq > 0:
                for k in range(jnt.nq):
                    self._arm_q_indices.append(jnt.idx_q + k)
        self._arm_q_indices = np.array(self._arm_q_indices, dtype=int)

        self.n_arm_dof = len(self._arm_q_indices)
        self.n_model_q = self.model.nq
        self.max_dof = max_dof

        # Neutral config used as padding template
        self._q_neutral = pin.neutral(self.model).copy()

        # ── 4. Characteristic length & scaling matrix ──
        pin.framesForwardKinematics(self.model, self.data, self._q_neutral)
        ee_pos = self.data.oMf[self.ee_frame_id].translation
        self.L_char = max(float(np.linalg.norm(ee_pos)), 1e-3)
        self.S_matrix = np.diag([1.0 / self.L_char] * 3 + [1.0] * 3)

    # ------------------------------------------------------------------

    def _pad_q(self, q_arm: np.ndarray) -> np.ndarray:
        """Expand arm-only (n_arm_dof,) → full model config (n_model_q,)."""
        q_full = self._q_neutral.copy()
        q_full[self._arm_q_indices[: len(q_arm)]] = q_arm
        return q_full

    # ------------------------------------------------------------------

    def compute_k_q_with_custom_limits(
        self, q_sequence: np.ndarray, q_min: np.ndarray, q_max: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            q_sequence: (T, n_data) — may be shorter than model.nq.
            q_min, q_max: Virtual limits matching n_data.
        Returns:
            (T, 42) feature array.
        """
        T = q_sequence.shape[0]
        n_data = q_sequence.shape[1]
        k_q_out = np.zeros((T, 42))

        range_q = np.maximum(q_max[:n_data] - q_min[:n_data], 1e-6)

        for t in range(T):
            q_arm = q_sequence[t]
            q_full = self._pad_q(q_arm)

            pin.framesForwardKinematics(self.model, self.data, q_full)
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q_full,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            # Keep only arm-chain Jacobian columns
            J = J_full[:, self._arm_q_indices[: n_data]]

            # ── 1. Range-weighted, L_char-normalized M* — 21D ──
            J_norm = self.S_matrix @ J
            d_vec = 1.0 / (np.square(range_q) + 1e-6)
            M_star = J_norm @ np.diag(d_vec) @ J_norm.T
            m_star_vech = M_star[np.triu_indices(6)]

            # ── 2. Periodic joint encoding — 14D ──
            q_normalized = 2.0 * np.pi * (q_arm - q_min[:n_data]) / range_q - np.pi
            q_pad = np.zeros(self.max_dof)
            q_pad[:n_data] = q_normalized
            periodic = np.concatenate([np.sin(q_pad), np.cos(q_pad)])

            # ── 3. L_char-normalized EE pose — 7D ──
            pose = self.data.oMf[self.ee_frame_id]
            p_norm = pose.translation / self.L_char
            quat = pin.Quaternion(pose.rotation).coeffs()
            if quat[3] < 0:
                quat = -quat
            ee_pose = np.concatenate([p_norm, quat])

            k_q_out[t] = np.concatenate([m_star_vech, periodic, ee_pose])

        return k_q_out
