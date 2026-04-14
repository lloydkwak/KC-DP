from typing import Optional

import numpy as np


def diversify_approach(
    grasp_pos: np.ndarray,
    theta_bins: int = 7,
    phi_bins: int = 8,
    offset: float = 0.08,
    max_theta_ratio: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample approach start points in spherical coordinates around grasp point.
    Returns (K, 3).
    """
    if rng is None:
        rng = np.random.default_rng()

    grasp_pos = np.asarray(grasp_pos, dtype=np.float32)
    thetas = np.linspace(0.0, max_theta_ratio * np.pi, theta_bins)
    phis = np.linspace(0.0, 2.0 * np.pi, phi_bins, endpoint=False)

    starts = []
    for th in thetas:
        for ph in phis:
            direction = np.array(
                [
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th),
                ],
                dtype=np.float32,
            )
            jitter = rng.normal(scale=0.003, size=(3,)).astype(np.float32)
            starts.append(grasp_pos + (offset * direction) + jitter)
    return np.stack(starts, axis=0)
