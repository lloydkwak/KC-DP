import numpy as np
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, List

import pinocchio as pin
from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule

class VirtualRobotSampler:
    """
    Virtual Robot Sampler for Hierarchical Kinematic Modulation (HKM).
    
    This class handles Stage 1 (Virtual Joint Limits) and Stage 2 (Virtual Link Lengths)
    augmentations. It dynamically modifies the URDF in memory via XML string parsing,
    generates virtual kinematics modules, and ensures the feasibility of the original
    trajectory on the virtual robot using the linearized tau condition.
    """

    def __init__(self, 
                 base_urdf_path: str, 
                 ee_frame_name: str, 
                 max_dof: int = 7,
                 max_retries: int = 20,
                 safety_margin: float = 0.05,
                 violation_threshold: float = 0.05):
        """
        Initializes the virtual robot sampler.
        
        Args:
            base_urdf_path: Path to the original robot URDF.
            ee_frame_name: Name of the end-effector frame.
            max_dof: Maximum degrees of freedom for padding.
            max_retries: Maximum attempts to find a feasible virtual robot topology.
            safety_margin: Margin added to the Jacobian Frobenius norm for feasibility checks.
            violation_threshold: Maximum allowed ratio of infeasible steps in a trajectory.
        """
        self.base_urdf_path = base_urdf_path
        self.ee_frame_name = ee_frame_name
        self.max_dof = max_dof
        self.max_retries = max_retries
        self.safety_margin = safety_margin
        self.violation_threshold = violation_threshold
        
        # Cache the original URDF as a string to prevent in-place mutation accumulation
        # and to eliminate temporary file I/O overhead during resampling.
        with open(base_urdf_path, 'r', encoding='utf-8') as f:
            self.base_urdf_str = f.read()
            
        # Parse base model to extract the exact kinematic chain driving the end-effector
        base_model = pin.buildModelFromXML(self.base_urdf_str)
        
        if not base_model.existFrame(self.ee_frame_name):
            raise ValueError(f"Frame '{self.ee_frame_name}' not found in URDF.")
            
        ee_frame_id = base_model.getFrameId(self.ee_frame_name)
        
        # Trace back from End-Effector to Universe (Base) to find the main kinematic chain
        current_joint_id = base_model.frames[ee_frame_id].parentJoint
        chain_joint_ids = []
        
        while current_joint_id > 0:  # 0 is the universe joint
            chain_joint_ids.append(current_joint_id)
            current_joint_id = base_model.parents[current_joint_id]
            
        # Reverse to establish a proximal-to-distal order (Base -> EE)
        chain_joint_ids.reverse()
        
        # Filter strictly for actuated joints (nq > 0) within this main chain.
        # This safely ignores parallel branches such as gripper fingers or head cameras.
        self.arm_chain_joint_names = []
        for j_id in chain_joint_ids:
            if base_model.joints[j_id].nq > 0:
                self.arm_chain_joint_names.append(base_model.names[j_id])

    def sample_stage1_limits(self, 
                             q_traj_min: np.ndarray, 
                             q_traj_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 1: Samples virtual joint limits that rigorously encompass the original trajectory.
        
        Args:
            q_traj_min: Minimum joint positions observed in the current trajectory window.
            q_traj_max: Maximum joint positions observed in the current trajectory window.
            
        Returns:
            q_min_v, q_max_v: Virtual minimum and maximum joint limits.
        """
        r_actual = q_traj_max - q_traj_min
        
        # Define bounds for virtual joint range (60 degrees to 315 degrees in radians)
        min_range_rad = np.radians(60.0)
        max_range_rad = np.radians(315.0)
        
        q_min_v = np.zeros_like(q_traj_min)
        q_max_v = np.zeros_like(q_traj_max)
        
        for i in range(len(r_actual)):
            lower_bound_r = max(r_actual[i], min_range_rad)
            upper_bound_r = max(lower_bound_r, max_range_rad)
            
            # Sample virtual range ensuring it accommodates the actual trajectory
            r_v = np.random.uniform(lower_bound_r, upper_bound_r)
            
            # Distribute the slack uniformly below the trajectory minimum
            slack_total = r_v - r_actual[i]
            slack_below = np.random.uniform(0.0, slack_total)
            
            q_min_v[i] = q_traj_min[i] - slack_below
            q_max_v[i] = q_min_v[i] + r_v
            
        return q_min_v, q_max_v

    def _modify_urdf_in_memory(self, scales: np.ndarray) -> str:
        """
        Modifies the URDF XML by scaling the XYZ translation of joint origins.
        
        Args:
            scales: Array of scaling factors corresponding to the actuated arm joints.
            
        Returns:
            A string representation of the dynamically scaled URDF.
        """
        # Parse a fresh XML tree from the cached string to guarantee independent mutations
        root = ET.fromstring(self.base_urdf_str)
        
        # Map scales precisely to the extracted arm chain joint names
        num_actuated = min(len(scales), len(self.arm_chain_joint_names))
        scale_map = {self.arm_chain_joint_names[i]: scales[i] for i in range(num_actuated)}
        
        for joint in root.findall('joint'):
            j_name = joint.get('name')
            if j_name in scale_map:
                scale_factor = scale_map[j_name]
                origin = joint.find('origin')
                if origin is not None and 'xyz' in origin.attrib:
                    xyz = list(map(float, origin.attrib['xyz'].split()))
                    # Scale the translation vector for the joint origin
                    scaled_xyz = [val * scale_factor for val in xyz]
                    origin.attrib['xyz'] = f"{scaled_xyz[0]:.6f} {scaled_xyz[1]:.6f} {scaled_xyz[2]:.6f}"
                    
        return ET.tostring(root, encoding='unicode')

    def check_action_feasibility(self, 
                                 virtual_module: AnalyticKinematicModule, 
                                 q_traj: np.ndarray, 
                                 delta_pose: np.ndarray) -> bool:
        """
        Validates whether the virtual robot can physically execute the required trajectory.
        
        Args:
            virtual_module: The proposed virtual kinematic module.
            q_traj: Joint trajectory array of shape (T, n_dof).
            delta_pose: Required task-space spatial velocity (dx/dt) array of shape (T, 6).
            
        Returns:
            True if the trajectory is kinematically feasible, False otherwise.
        """
        T = q_traj.shape[0]
        violations = 0
        
        for t in range(T):
            q_t = q_traj[t]
            dp_t = delta_pose[t] 
            
            pin.framesForwardKinematics(virtual_module.model, virtual_module.data, q_t)
            # Compute Jacobian relative to the LOCAL_WORLD_ALIGNED frame
            J = pin.computeFrameJacobian(
                virtual_module.model, 
                virtual_module.data, 
                q_t, 
                virtual_module.ee_frame_id, 
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            
            # Normalize the Jacobian and the required task-space velocity
            J_norm = virtual_module.S_matrix @ J
            dp_norm = virtual_module.S_matrix @ dp_t
            
            # The Frobenius norm of the normalized Jacobian approximates the upper bound
            # of the achievable scaled task-space velocity.
            J_frob = np.linalg.norm(J_norm, ord='fro')
            required_mobility = np.linalg.norm(dp_norm)
            
            if required_mobility > (J_frob + self.safety_margin):
                violations += 1
                
        violation_rate = violations / float(T)
        return violation_rate < self.violation_threshold

    def sample_stage2_module(self, 
                             q_traj: np.ndarray, 
                             delta_pose: np.ndarray,
                             q_min_v: np.ndarray, 
                             q_max_v: np.ndarray) -> Tuple[AnalyticKinematicModule, np.ndarray, np.ndarray]:
        """
        Stage 2: Iteratively samples virtual link scales until a kinematically feasible
        robot topology is discovered.
        
        Args:
            q_traj: Original joint trajectory of shape (T, n_dof).
            delta_pose: Original task-space spatial velocity of shape (T, 6).
            q_min_v: Virtual minimum joint limits from Stage 1.
            q_max_v: Virtual maximum joint limits from Stage 1.
            
        Returns:
            A tuple containing the feasible AnalyticKinematicModule and the virtual joint limits.
        """
        n_dof = q_traj.shape[1]
        
        for attempt in range(self.max_retries):
            # Sample link scales from a LogNormal distribution to favor variations around 1.0
            scales = np.random.lognormal(mean=0.0, sigma=0.4, size=n_dof)
            scales = np.clip(scales, 0.3, 3.0)
            
            modified_urdf_str = self._modify_urdf_in_memory(scales)
            
            try:
                # Construct the kinematic module directly from the scaled XML string
                virtual_module = AnalyticKinematicModule(
                    urdf_xml=modified_urdf_str, 
                    ee_frame_name=self.ee_frame_name, 
                    max_dof=self.max_dof
                )
                
                is_feasible = self.check_action_feasibility(virtual_module, q_traj, delta_pose)
                if is_feasible:
                    return virtual_module, q_min_v, q_max_v
                    
            except Exception:
                # Silently catch potential Pinocchio build failures caused by extreme topological deformations
                pass
                
        # Fallback Strategy: Revert to the base robot geometry while retaining the Stage 1 virtual limits.
        fallback_module = AnalyticKinematicModule(
            urdf_xml=self.base_urdf_str,
            ee_frame_name=self.ee_frame_name,
            max_dof=self.max_dof
        )
        return fallback_module, q_min_v, q_max_v
