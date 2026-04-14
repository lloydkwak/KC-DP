"""
TSTD Offline Augmentation (RemoteBot)
=====================================

Input : RoboMimic low_dim_abs.hdf5
Output: augmented hdf5 with diversified task-space trajectories
"""

import argparse
import os
from typing import Dict, List

import h5py
import numpy as np

from remotebot.tstd import (
    diversify_approach,
    extract_pickplace_keypoints,
    sample_diverse_grasps,
)


def _read_demo(demo_grp) -> Dict[str, np.ndarray]:
    obs = demo_grp["obs"]
    obs_dict = {k: np.asarray(obs[k][:], dtype=np.float32) for k in obs.keys()}

    out = {
        "obs": obs_dict,
        "eef_pos": np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32),
        "eef_quat": np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float32),
        "gripper": np.asarray(obs_dict["robot0_gripper_qpos"], dtype=np.float32),
        "actions": np.asarray(demo_grp["actions"][:], dtype=np.float32),
    }
    for k in ["states", "rewards", "dones"]:
        if k in demo_grp:
            out[k] = np.asarray(demo_grp[k][:])
    return out


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / max(1e-8, float(np.linalg.norm(q)))


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _quat_normalize(q0.astype(np.float32))
    q1 = _quat_normalize(q1.astype(np.float32))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize(q0 + t * (q1 - q0))
    th0 = float(np.arccos(dot))
    sth0 = float(np.sin(th0))
    s0 = float(np.sin((1.0 - t) * th0) / max(sth0, 1e-8))
    s1 = float(np.sin(t * th0) / max(sth0, 1e-8))
    return _quat_normalize(s0 * q0 + s1 * q1)


def _quat_from_approach_direction(direction: np.ndarray) -> np.ndarray:
    z = direction.astype(np.float32)
    nz = float(np.linalg.norm(z))
    if nz < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    z = z / nz
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(z, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = np.cross(up, z)
    x = x / max(1e-8, float(np.linalg.norm(x)))
    y = np.cross(z, x)
    y = y / max(1e-8, float(np.linalg.norm(y)))
    R = np.stack([x, y, z], axis=-1)

    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return _quat_normalize(np.array([qx, qy, qz, qw], dtype=np.float32))


def _quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat must be [T,4], got shape={q.shape}")
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    xyz = q[:, :3]
    w = np.clip(q[:, 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = xyz / np.maximum(sin_half[:, None], 1e-8)
    aa = axis * angle[:, None]
    small = sin_half < 1e-5
    if np.any(small):
        aa[small] = 2.0 * xyz[small]
    return aa.astype(np.float32)


def _compose_action_from_pose(pos: np.ndarray, quat: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    # Task-space raw action layout (robomimic abs-compatible):
    # [x, y, z, axis_angle(3), grip_scalar] => 7D
    if gripper.ndim > 1:
        g = gripper.mean(axis=-1, keepdims=True)
    else:
        g = gripper.reshape(-1, 1)
    aa = _quat_xyzw_to_axis_angle(quat)
    return np.concatenate([pos, aa, g.astype(np.float32)], axis=-1)


def interpolate_segment_pos(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    n_steps: int,
    mode: str = "linear",
    via_point: np.ndarray = None,
) -> np.ndarray:
    """
    Interpolate segment positions with n_steps samples (including endpoints).
    """
    n_steps = int(max(2, n_steps))
    t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)

    if mode == "bezier":
        if via_point is None:
            via_point = 0.5 * (start_pos + end_pos)
        p0 = start_pos.astype(np.float32)
        p1 = via_point.astype(np.float32)
        p2 = end_pos.astype(np.float32)
        return (
            ((1.0 - t) ** 2)[:, None] * p0[None, :]
            + (2.0 * (1.0 - t) * t)[:, None] * p1[None, :]
            + (t**2)[:, None] * p2[None, :]
        ).astype(np.float32)

    return np.linspace(start_pos, end_pos, n_steps, dtype=np.float32)


def interpolate_segment_quat(start_quat: np.ndarray, end_quat: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Interpolate segment orientations with Slerp using n_steps samples (including endpoints).
    """
    n_steps = int(max(2, n_steps))
    out = np.zeros((n_steps, 4), dtype=np.float32)
    for i in range(n_steps):
        alpha = float(i) / float(max(1, n_steps - 1))
        out[i] = _quat_slerp(start_quat, end_quat, alpha)
    return out


def _allocate_segment_steps_from_ratios(T: int, anchor_indices: List[int]) -> List[int]:
    """
    Allocate segment lengths (in intervals) by preserving original keypoint interval ratios.
    Sum(seg_steps) == T-1.
    """
    total_intervals = int(max(1, T - 1))
    spans = np.diff(np.asarray(anchor_indices, dtype=np.int64))
    spans = np.maximum(spans, 1)

    weights = spans.astype(np.float64) / float(np.sum(spans))
    raw = weights * float(total_intervals)
    seg_steps = np.floor(raw).astype(np.int64)
    seg_steps = np.maximum(seg_steps, 1)

    diff = int(total_intervals - int(np.sum(seg_steps)))
    if diff != 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac if diff > 0 else frac)
        idx = 0
        while diff != 0 and idx < len(order) * 4:
            j = int(order[idx % len(order)])
            if diff > 0:
                seg_steps[j] += 1
                diff -= 1
            else:
                if seg_steps[j] > 1:
                    seg_steps[j] -= 1
                    diff += 1
            idx += 1

    return seg_steps.tolist()


def build_full_trajectory(
    keypoints: Dict[str, Dict[str, np.ndarray]],
    step_allocation: Dict[str, int],
    T: int,
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    Build a full T-step trajectory by stitching keypoint-to-keypoint segments.
    """
    segment_order = [
        ("start", "approach", "start_to_approach"),
        ("approach", "grasp", "approach_to_grasp"),
        ("grasp", "lift_peak", "grasp_to_lift"),
        ("lift_peak", "place", "lift_to_place"),
        ("place", "end", "place_to_end"),
    ]

    pos_chunks = []
    quat_chunks = []

    for i, (k0, k1, seg_name) in enumerate(segment_order):
        n_interval = int(max(1, step_allocation.get(seg_name, 1)))
        n_samples = n_interval + 1

        i0 = int(keypoints[k0]["idx"])
        i1 = int(keypoints[k1]["idx"])

        if seg_name == "start_to_approach":
            p0 = keypoints[k0]["pos"]
            p1 = keypoints[k1]["pos"]
            via = 0.5 * (p0 + p1)
            via[2] += 0.02
            seg_pos = interpolate_segment_pos(p0, p1, n_samples, mode="bezier", via_point=via)
            seg_quat = interpolate_segment_quat(keypoints[k0]["quat"], keypoints[k1]["quat"], n_samples)
        elif seg_name == "approach_to_grasp":
            seg_pos = interpolate_segment_pos(
                keypoints[k0]["pos"],
                keypoints[k1]["pos"],
                n_samples,
                mode="linear",
            )
            seg_quat = interpolate_segment_quat(keypoints[k0]["quat"], keypoints[k1]["quat"], n_samples)
        elif seg_name == "grasp_to_lift":
            seg_pos = interpolate_segment_pos(
                keypoints[k0]["pos"],
                keypoints[k1]["pos"],
                n_samples,
                mode="linear",
            )
            # Keep original orientation trend in this segment.
            if (i1 - i0 + 1) == n_samples:
                seg_quat = interpolate_segment_quat(keypoints[k0]["quat"], quat_ref[i1], n_samples)
            else:
                seg_quat = interpolate_segment_quat(keypoints[k0]["quat"], quat_ref[i1], n_samples)
        elif seg_name == "lift_to_place":
            # Preserve original carried-object-consistent path/orientation.
            if (i1 - i0 + 1) == n_samples:
                seg_pos = pos_ref[i0 : i1 + 1].copy()
                seg_quat = quat_ref[i0 : i1 + 1].copy()
            else:
                seg_pos = interpolate_segment_pos(pos_ref[i0], pos_ref[i1], n_samples, mode="linear")
                seg_quat = interpolate_segment_quat(quat_ref[i0], quat_ref[i1], n_samples)
        else:  # place_to_end
            seg_pos = interpolate_segment_pos(
                keypoints[k0]["pos"],
                keypoints[k1]["pos"],
                n_samples,
                mode="linear",
            )
            seg_quat = interpolate_segment_quat(keypoints[k0]["quat"], keypoints[k1]["quat"], n_samples)

        if i > 0:
            seg_pos = seg_pos[1:]
            seg_quat = seg_quat[1:]

        pos_chunks.append(seg_pos)
        quat_chunks.append(seg_quat)

    pos_aug = np.concatenate(pos_chunks, axis=0)
    quat_aug = np.concatenate(quat_chunks, axis=0)

    if pos_aug.shape[0] != T:
        idx = np.linspace(0, pos_aug.shape[0] - 1, T).astype(np.int64)
        pos_aug = pos_aug[idx]
        quat_aug = quat_aug[idx]

    return pos_aug.astype(np.float32), quat_aug.astype(np.float32)


def _augment_single_demo(
    demo: Dict[str, np.ndarray],
    grasp_variants: int,
    approach_variants: int,
    path_variants: int,
    rng: np.random.Generator,
) -> List[Dict[str, np.ndarray]]:
    pos = demo["eef_pos"]
    quat = demo["eef_quat"]
    grip = demo["gripper"]
    obs_base = demo["obs"]

    kps = extract_pickplace_keypoints(pos, grip)
    T = int(pos.shape[0])
    a0 = int(kps["approach_start"])
    g = int(kps["grasp"])
    lift = int(kps["lift"])
    tr = int(kps["transport"])
    rel = int(kps["release"])

    # Keep ordering robust even under rare noisy boundary cases.
    a0 = max(0, min(a0, T - 5))
    g = max(a0 + 1, min(g, T - 4))
    lift = max(g + 1, min(lift, T - 3))
    tr = max(lift + 1, min(tr, T - 2))
    rel = max(tr + 1, min(rel, T - 1))

    grasp_pos = pos[g]
    local_scale = max(1e-3, float(np.std(pos, axis=0).mean()) * 0.05)
    pseudo_points = grasp_pos[None, :] + rng.normal(
        loc=0.0,
        scale=local_scale,
        size=(max(64, grasp_variants * 12), 3),
    ).astype(np.float32)

    grasp_targets = [grasp_pos.astype(np.float32)]
    try:
        sampled = sample_diverse_grasps(
            pseudo_points,
            n=max(grasp_variants * 3, grasp_variants),
            rng=rng,
        )
        grasp_targets.extend(sampled[:, :3].astype(np.float32))
    except Exception:
        grasp_targets.extend(
            (grasp_pos[None, :] + rng.normal(0.0, local_scale, size=(max(grasp_variants * 3, 6), 3))).astype(np.float32)
        )

    uniq_targets = np.unique(np.stack(grasp_targets, axis=0), axis=0)
    if len(uniq_targets) > grasp_variants:
        keep_idx = rng.choice(len(uniq_targets), size=grasp_variants, replace=False)
        uniq_targets = uniq_targets[keep_idx]

    # path_variants is intentionally not used in this regeneration mode for now.
    # We preserve lift->place from original trajectory to keep object-state consistency.
    _ = path_variants

    anchor_indices = [0, a0, g, lift, rel, T - 1]
    seg_intervals = _allocate_segment_steps_from_ratios(T, anchor_indices)
    step_allocation = {
        "start_to_approach": int(seg_intervals[0]),
        "approach_to_grasp": int(seg_intervals[1]),
        "grasp_to_lift": int(seg_intervals[2]),
        "lift_to_place": int(seg_intervals[3]),
        "place_to_end": int(seg_intervals[4]),
    }

    out = []
    for grasp_target in uniq_targets:
        approach_starts = diversify_approach(
            grasp_target,
            theta_bins=7,
            phi_bins=8,
            offset=0.08,
            rng=rng,
        )
        if len(approach_starts) > approach_variants:
            sel = rng.choice(len(approach_starts), size=approach_variants, replace=False)
            approach_starts = approach_starts[sel]

        for ap in approach_starts:
            q_approach = _quat_from_approach_direction(grasp_target - ap)
            # Keep grasp orientation aligned with approach direction in regenerated path.
            q_grasp = q_approach.copy()

            kp_poses = {
                "start": {"idx": 0, "pos": pos[0], "quat": quat[0]},
                "approach": {"idx": a0, "pos": ap.astype(np.float32), "quat": q_approach},
                "grasp": {"idx": g, "pos": grasp_target.astype(np.float32), "quat": q_grasp},
                "lift_peak": {"idx": lift, "pos": pos[lift], "quat": quat[lift]},
                "place": {"idx": rel, "pos": pos[rel], "quat": quat[rel]},
                "end": {"idx": T - 1, "pos": pos[T - 1], "quat": quat[T - 1]},
            }

            pos_aug, quat_aug = build_full_trajectory(
                keypoints=kp_poses,
                step_allocation=step_allocation,
                T=T,
                pos_ref=pos,
                quat_ref=quat,
            )

            obs_aug = {k: v.copy() for k, v in obs_base.items()}
            obs_aug["robot0_eef_pos"] = pos_aug
            obs_aug["robot0_eef_quat"] = quat_aug
            obs_aug["robot0_gripper_qpos"] = grip

            action_aug = _compose_action_from_pose(pos_aug, quat_aug, grip)
            out.append(
                {
                    "obs": obs_aug,
                    "actions": action_aug,
                    "states": demo.get("states", None),
                    "rewards": demo.get("rewards", None),
                    "dones": demo.get("dones", None),
                }
            )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input low_dim_abs.hdf5")
    parser.add_argument("--output", required=True, help="Output augmented hdf5")
    parser.add_argument("--grasp_variants", type=int, default=5)
    parser.add_argument("--approach_variants", type=int, default=30)
    parser.add_argument("--path_variants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        g_data_in = fin["data"]
        g_data_out = fout.create_group("data")

        demo_keys = sorted(list(g_data_in.keys()))
        demo_out_idx = 0

        for key in demo_keys:
            demo = _read_demo(g_data_in[key])
            augmented = _augment_single_demo(
                demo,
                grasp_variants=args.grasp_variants,
                approach_variants=args.approach_variants,
                path_variants=args.path_variants,
                rng=rng,
            )

            for sample in augmented:
                grp = g_data_out.create_group(f"demo_{demo_out_idx}")
                obs_grp = grp.create_group("obs")
                for obs_key, obs_val in sample["obs"].items():
                    obs_grp.create_dataset(obs_key, data=obs_val, compression="gzip")

                grp.create_dataset("actions", data=sample["actions"], compression="gzip")
                if sample.get("states", None) is not None:
                    grp.create_dataset("states", data=sample["states"], compression="gzip")
                if sample.get("rewards", None) is not None:
                    grp.create_dataset("rewards", data=sample["rewards"], compression="gzip")
                if sample.get("dones", None) is not None:
                    grp.create_dataset("dones", data=sample["dones"], compression="gzip")
                demo_out_idx += 1

        fout.attrs["augmentation"] = "TSTD"
        fout.attrs["source_file"] = args.input
        fout.attrs["grasp_variants"] = int(args.grasp_variants)
        fout.attrs["approach_variants"] = int(args.approach_variants)
        fout.attrs["path_variants"] = int(args.path_variants)
        fout.attrs["num_demos"] = demo_out_idx

    print(f"[TSTD] done: {args.output} | demos={demo_out_idx}")


if __name__ == "__main__":
    main()
