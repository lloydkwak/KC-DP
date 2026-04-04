import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule


class KQFeatureComputer:
    """
    Shared k(q) feature computer used by dataset/train/eval.

    This enforces a single, identical definition of:
    - robot model (URDF + EE frame)
    - joint limits used in k(q)
    - normalization (k_mean, k_std)
    """

    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str,
        max_dof: int = 7,
        stats_path: Optional[str] = None,
        k_mean: Optional[np.ndarray] = None,
        k_std: Optional[np.ndarray] = None,
        use_physical_limits: bool = True,
    ):
        self.urdf_path = urdf_path
        self.ee_frame_name = ee_frame_name
        self.max_dof = max_dof
        self.use_physical_limits = use_physical_limits

        self.km = AnalyticKinematicModule(
            urdf_path=self.urdf_path,
            ee_frame_name=self.ee_frame_name,
            max_dof=self.max_dof,
        )

        if k_mean is None or k_std is None:
            if stats_path is None or (not os.path.exists(stats_path)):
                self.k_mean = np.zeros(42, dtype=np.float32)
                self.k_std = np.ones(42, dtype=np.float32)
            else:
                st = torch.load(stats_path, map_location="cpu")
                self.k_mean = np.asarray(st["mean"], dtype=np.float32)
                self.k_std = np.clip(np.asarray(st["std"], dtype=np.float32), 1e-6, None)
        else:
            self.k_mean = np.asarray(k_mean, dtype=np.float32)
            self.k_std = np.clip(np.asarray(k_std, dtype=np.float32), 1e-6, None)

        idx = self.km._arm_q_indices
        self.q_min_base = np.where(
            np.isinf(self.km.model.lowerPositionLimit[idx]),
            -2 * np.pi,
            self.km.model.lowerPositionLimit[idx],
        ).astype(np.float64)
        self.q_max_base = np.where(
            np.isinf(self.km.model.upperPositionLimit[idx]),
            2 * np.pi,
            self.km.model.upperPositionLimit[idx],
        ).astype(np.float64)

    def get_limits(self, n_joints: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_physical_limits:
            return self.q_min_base[:n_joints], self.q_max_base[:n_joints]
        return self.q_min_base[:n_joints], self.q_max_base[:n_joints]

    def compute_raw(self, q_pos: np.ndarray) -> np.ndarray:
        q = np.asarray(q_pos, dtype=np.float64)
        n = q.shape[-1]
        q_min, q_max = self.get_limits(n)
        return self.km.compute_k_q_with_custom_limits(q[np.newaxis], q_min, q_max)[0]

    def compute_normalized(self, q_pos: np.ndarray) -> np.ndarray:
        raw = self.compute_raw(q_pos)
        return ((raw - self.k_mean) / self.k_std).astype(np.float32)


_PID_CACHE: Dict[int, KQFeatureComputer] = {}


def get_pid_cached_kq_computer(
    urdf_path: str,
    ee_frame_name: str,
    max_dof: int,
    stats_path: Optional[str],
) -> KQFeatureComputer:
    pid = os.getpid()
    comp = _PID_CACHE.get(pid)
    if comp is None:
        comp = KQFeatureComputer(
            urdf_path=urdf_path,
            ee_frame_name=ee_frame_name,
            max_dof=max_dof,
            stats_path=stats_path,
            use_physical_limits=True,
        )
        _PID_CACHE[pid] = comp
    return comp
