from typing import Dict

import numpy as np


def _find_gripper_transition(gripper: np.ndarray, rising: bool) -> int:
    if gripper.ndim > 1:
        g = gripper.mean(axis=-1)
    else:
        g = gripper
    dg = np.diff(g, prepend=g[0])
    if rising:
        idx = int(np.argmax(dg))
    else:
        idx = int(np.argmin(dg))
    return idx


def extract_pickplace_keypoints(eef_pos: np.ndarray, gripper: np.ndarray) -> Dict[str, int]:
    """
    Heuristic phase split for pick-place style trajectories.
    """
    T = int(eef_pos.shape[0])

    def _fallback_split(tt: int) -> Dict[str, int]:
        return {
            "approach_start": 0,
            "grasp": max(1, tt // 4),
            "lift": max(2, tt // 2),
            "transport": max(3, (3 * tt) // 4),
            "release": tt - 1,
        }

    if T < 6:
        return _fallback_split(T)

    vel = np.linalg.norm(np.diff(eef_pos, axis=0, prepend=eef_pos[:1]), axis=-1)
    grasp = _find_gripper_transition(gripper, rising=False)
    release = _find_gripper_transition(gripper, rising=True)
    lift = int(min(T - 1, grasp + np.argmax(vel[grasp: max(grasp + 2, T // 2)])))
    approach_start = int(max(0, grasp - max(3, T // 8)))
    transport = int(max(lift + 1, (lift + release) // 2))
    out = {
        "approach_start": approach_start,
        "grasp": int(np.clip(grasp, 0, T - 1)),
        "lift": int(np.clip(lift, 0, T - 1)),
        "transport": int(np.clip(transport, 0, T - 1)),
        "release": int(np.clip(release, 0, T - 1)),
    }

    # Enforce strict temporal ordering expected by augmentation logic.
    if not (out["approach_start"] < out["grasp"] < out["lift"] < out["transport"] < out["release"]):
        return _fallback_split(T)
    return out
