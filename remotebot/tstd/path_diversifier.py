from typing import Optional

import numpy as np


def _bezier(start: np.ndarray, via: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return ((1 - t)[:, None] ** 2) * start + 2 * ((1 - t)[:, None] * t[:, None]) * via + (t[:, None] ** 2) * end


def diversify_path(
    start: np.ndarray,
    end: np.ndarray,
    n: int = 10,
    steps: int = 20,
    lateral: float = 0.15,
    height: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate free-space path variants via bezier via-points.
    Returns shape: (N, steps, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    base = 0.5 * (start + end)
    d = end - start
    d_norm = np.linalg.norm(d) + 1e-8
    tangent = d / d_norm

    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    lat = np.cross(tangent, z)
    if np.linalg.norm(lat) < 1e-6:
        lat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    lat = lat / (np.linalg.norm(lat) + 1e-8)

    paths = []
    for _ in range(n):
        lat_scale = rng.uniform(-lateral, lateral)
        h_scale = rng.uniform(0.0, height)
        via = base + lat_scale * lat + np.array([0.0, 0.0, h_scale], dtype=np.float32)
        paths.append(_bezier(start, via, end, steps=steps))

    return np.stack(paths, axis=0)
