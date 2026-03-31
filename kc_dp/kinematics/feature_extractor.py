import numpy as np
import pinocchio as pin

class AnalyticKinematicModule:
    """
    Analytic Kinematic Module for Hierarchical Kinematic Modulation (HKM).
    
    Computes a 42-dimensional, embodiment-agnostic kinematic feature vector k(q).
    The vector consists of:
        - 21D: Upper-triangular elements of the range-weighted manipulability matrix (M*).
        - 7D: sin(pi * q_norm), zero-padded to `max_dof` to ensure fixed dimensionality.
        - 7D: cos(pi * q_norm), zero-padded to `max_dof`.
        - 7D: Normalized EE position (3D) and continuous Quaternion (4D).
    """

    def __init__(self, 
                 urdf_path: str, 
                 ee_frame_name: str, 
                 max_dof: int = 7):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.max_dof = max_dof
        
        assert self.model.nq == self.model.nv, \
            f"Mismatch between position DoF ({self.model.nq}) and velocity DoF ({self.model.nv}). Floating base robots are not supported."
        
        if self.model.existFrame(ee_frame_name):
            self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        else:
            raise ValueError(f"Frame '{ee_frame_name}' not found in URDF.")
            
        q_zero = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_zero)
        ee_pos_zero = self.data.oMf[self.ee_frame_id].translation
        self.L_char = max(np.linalg.norm(ee_pos_zero), 1e-3)

        # Scale matrix S for spatial normalization (Translation: 1/L_char, Rotation: 1.0)
        self.S_matrix = np.diag([1.0 / self.L_char] * 3 + [1.0] * 3)

    def compute_k_q_with_custom_limits(self, 
                                       q_window: np.ndarray, 
                                       q_min_v: np.ndarray, 
                                       q_max_v: np.ndarray) -> np.ndarray:
        is_single_step = q_window.ndim == 1
        if is_single_step:
            q_window = q_window[np.newaxis, :]
            
        T, n_dof = q_window.shape
        if n_dof > self.max_dof:
            raise ValueError(f"Robot DoF ({n_dof}) exceeds max_dof ({self.max_dof}).")

        q_center = (q_max_v + q_min_v) / 2.0
        q_half = (q_max_v - q_min_v) / 2.0
        q_range = q_max_v - q_min_v
        
        d_vec = 1.0 / (np.square(q_range) + 1e-6)
        D_matrix = np.diag(d_vec)

        k_q_list = []
        pad_width = self.max_dof - n_dof
        
        for t in range(T):
            q_t = q_window[t]
            
            # 1. Forward Kinematics & Jacobian Computation
            pin.framesForwardKinematics(self.model, self.data, q_t)
            J = pin.computeFrameJacobian(
                self.model, self.data, q_t, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            
            # 2. Normalized Range-Weighted Manipulability (M*) -> 21D
            J_norm = self.S_matrix @ J
            M_star = J_norm @ D_matrix @ J_norm.T
            m_star_vech = M_star[np.triu_indices(6)]
            
            # 3. Periodic Joint Encoding (sin/cos of normalized q)
            q_norm_pi = np.pi * (q_t - q_center) / (q_half + 1e-6)
            sin_q = np.sin(q_norm_pi)
            cos_q = np.cos(q_norm_pi)
            
            sin_q_padded = np.pad(sin_q, (0, pad_width), mode='constant', constant_values=0.0)
            cos_q_padded = np.pad(cos_q, (0, pad_width), mode='constant', constant_values=0.0)
            
            # 4. Normalized End-Effector Pose -> 7D
            ee_pose = self.data.oMf[self.ee_frame_id]
            p_norm = ee_pose.translation / self.L_char
            
            quat = pin.Quaternion(ee_pose.rotation).coeffs() # [x, y, z, w]
            if quat[3] < 0:
                quat = -quat
            
            # 5. Feature Concatenation (Total 42D)
            k_q_t = np.concatenate([
                m_star_vech,    # 21
                sin_q_padded,   # 7
                cos_q_padded,   # 7
                p_norm,         # 3
                quat            # 4
            ])
            
            k_q_list.append(k_q_t)

        k_q_array = np.vstack(k_q_list)
        return k_q_array[0] if is_single_step else k_q_array
