import numpy as np
import pinocchio as pin

class AnalyticKinematicModule:
    """
    Computes kinematics and extracts the 42-dimensional k(q) feature vector 
    using Pinocchio for high-performance rigid body algorithms.
    """
    def __init__(self, urdf_path=None, urdf_xml=None, ee_frame_name='', max_dof=7):
        """
        Initializes the kinematic module from either a file path or an XML string.
        
        Args:
            urdf_path: Path to the URDF file (used for the base robot).
            urdf_xml: String containing the URDF XML (used for virtual robots).
            ee_frame_name: Name of the end-effector frame in the URDF.
            max_dof: Maximum degrees of freedom to pad the feature vector (default: 7).
        """
        # 1. Load the robot model safely
        if urdf_xml is not None:
            # Parse directly from the modified XML string (used by VirtualRobotSampler)
            self.model = pin.buildModelFromXML(urdf_xml)
        elif urdf_path is not None:
            # Parse from an existing URDF file (used for the base robot)
            self.model = pin.buildModelFromUrdf(urdf_path)
        else:
            raise ValueError("Either urdf_path or urdf_xml must be provided.")

        self.data = self.model.createData()
        
        # 2. Resolve the End-Effector (EE) frame ID
        if not self.model.existFrame(ee_frame_name):
            raise ValueError(f"Frame '{ee_frame_name}' not found in the URDF.")
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        
        # 3. Cache Degree of Freedom (DoF) settings
        self.n_dof = self.model.nq
        self.max_dof = max_dof

        # 4. Calculate Characteristic Length (L_char) and Scaling Matrix (S_matrix)
        # This ensures scale invariance across different virtual link lengths (Stage 2).
        q_zero = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q_zero)
        ee_pos_zero = self.data.oMf[self.ee_frame_id].translation
        
        # Extract the scalar characteristic length, bounded to prevent division by zero
        self.L_char = max(float(np.linalg.norm(ee_pos_zero)), 1e-3)
        
        # Construct the scaling matrix S: 
        # Normalizes translational components by L_char, leaves rotational components intact.
        self.S_matrix = np.diag([1.0 / self.L_char] * 3 + [1.0] * 3)

    def compute_k_q_with_custom_limits(self, q_sequence: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
        """
        Computes the 42D kinematic feature vector k(q) for a sequence of joint positions,
        incorporating range-weighting and scale invariance.
        
        Args:
            q_sequence: Sequence of joint positions, shape (T, n_dof)
            q_min: Virtual lower joint limits
            q_max: Virtual upper joint limits
            
        Returns:
            k_q_sequence: The 42D kinematic feature sequence, shape (T, 42)
        """
        T = q_sequence.shape[0]
        k_q_sequence = np.zeros((T, 42))
        
        # Compute the valid joint range, bounded to prevent numerical instability
        range_q = np.maximum(q_max - q_min, 1e-6)
        
        for t in range(T):
            q_t = q_sequence[t]
            
            # Forward kinematics and spatial Jacobian computation
            pin.framesForwardKinematics(self.model, self.data, q_t)
            J = pin.computeFrameJacobian(
                self.model, self.data, q_t, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            
            # 1. Range-weighted & L_char-normalized Manipulability (M*) - 21D
            # Apply the scaling matrix to the Jacobian to ensure scale invariance
            J_norm = self.S_matrix @ J
            
            # Construct the diagonal penalty matrix D based on the active DoF limits
            d_vec_t = 1.0 / (np.square(range_q[:self.n_dof]) + 1e-6)
            D_matrix_t = np.diag(d_vec_t)
            
            # Calculate the manipulability matrix M* = J_norm * D * J_norm^T
            # This matrix is symmetric positive definite (SPD)
            M_star = J_norm @ D_matrix_t @ J_norm.T
            m_star_vech = M_star[np.triu_indices(6)] # 21 dimensions
            
            # 2. Periodic Joint Encoding - 14D
            # Map the joint position to [-pi, pi] relative to the virtual limits
            q_normalized = 2 * np.pi * (q_t - q_min[:self.n_dof]) / range_q[:self.n_dof] - np.pi
            
            # Pad the encoding to the maximum specified degrees of freedom (e.g., 7)
            q_norm_padded = np.zeros(self.max_dof)
            q_norm_padded[:self.n_dof] = q_normalized
            periodic_encoding = np.concatenate([np.sin(q_norm_padded), np.cos(q_norm_padded)]) # 14 dimensions
            
            # 3. L_char-normalized End-Effector Pose - 7D
            pose = self.data.oMf[self.ee_frame_id]
            
            # Normalize the positional translation vector by the characteristic length
            p_norm = pose.translation / self.L_char
            
            # Extract quaternion coefficients as an isolated numpy array [x, y, z, w]
            quat = pin.Quaternion(pose.rotation).coeffs() 
            
            # Safely resolve quaternion sign ambiguity to ensure a continuous representation
            if quat[3] < 0:
                quat = -quat
                
            ee_pose = np.concatenate([p_norm, quat]) # 7 dimensions
            
            # Assemble the final 42D kinematic feature vector for the current timestep
            k_q_sequence[t] = np.concatenate([m_star_vech, periodic_encoding, ee_pose])
            
        return k_q_sequence