from typing import Optional

import numpy as np


def sample_diverse_grasps(
    object_points: np.ndarray,
    n: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Lightweight antipodal-inspired grasp center/orientation sampling from point cloud.

    Returns shape: (N, 7) => xyz + quaternion(xyzw)
    """
    if rng is None:
        rng = np.random.default_rng()
    pts = np.asarray(object_points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("object_points must be shape [M,3]")
    if len(pts) < 2:
        raise ValueError("object_points needs at least two points")

    out = []
    cand = max(n * 6, n)
    for _ in range(cand):
        i, j = rng.integers(0, len(pts), size=2)
        if i == j:
            continue
        p1 = pts[i]
        p2 = pts[j]
        center = 0.5 * (p1 + p2)
        approach = p2 - p1
        norm = np.linalg.norm(approach) + 1e-8
        approach = approach / norm
        yaw = float(np.arctan2(approach[1], approach[0]))
        qw = np.cos(yaw * 0.5)
        qz = np.sin(yaw * 0.5)
        quat = np.array([0.0, 0.0, qz, qw], dtype=np.float32)
        out.append(np.concatenate([center.astype(np.float32), quat], axis=0))
        if len(out) >= n:
            break
    if len(out) == 0:
        raise RuntimeError("failed to sample grasps")
    return np.stack(out, axis=0)
