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
    diversify_path,
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
    a0 = kps["approach_start"]
    g = kps["grasp"]
    tr = kps["transport"]
    rel = kps["release"]

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

    transport_start = pos[max(g + 1, tr - 1)]
    place_pos = pos[rel]
    transport_paths = diversify_path(
        transport_start,
        place_pos,
        n=path_variants,
        steps=max(4, rel - tr + 1),
        rng=rng,
    )

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
            for pth in transport_paths:
                pos_aug = pos.copy()
                quat_aug = quat.copy()

                approach_len = max(2, g - a0 + 1)
                alpha = np.linspace(0.0, 1.0, approach_len, dtype=np.float32)[:, None]
                approach_curve = (1 - alpha) * ap[None, :] + alpha * grasp_target[None, :]
                pos_aug[a0 : g + 1] = approach_curve

                q_start = _quat_from_approach_direction(grasp_target - ap)
                q_end = quat[g]
                for i_local, t_global in enumerate(range(a0, g + 1)):
                    blend = float(i_local) / float(max(1, (g - a0)))
                    quat_aug[t_global] = _quat_slerp(q_start, q_end, blend)

                if rel >= tr:
                    nseg = rel - tr + 1
                    if pth.shape[0] != nseg:
                        idx = np.linspace(0, pth.shape[0] - 1, nseg).astype(np.int64)
                        pos_aug[tr : rel + 1] = pth[idx]
                    else:
                        pos_aug[tr : rel + 1] = pth

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
