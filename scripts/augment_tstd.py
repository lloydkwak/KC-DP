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
)


def _read_demo(demo_grp) -> Dict[str, np.ndarray]:
    obs = demo_grp["obs"]
    eef_pos = np.asarray(obs["robot0_eef_pos"][:], dtype=np.float32)
    eef_quat = np.asarray(obs["robot0_eef_quat"][:], dtype=np.float32)
    grip = np.asarray(obs["robot0_gripper_qpos"][:], dtype=np.float32)
    actions = np.asarray(demo_grp["actions"][:], dtype=np.float32)
    return {
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "gripper": grip,
        "actions": actions,
    }


def _compose_action_from_pose(pos: np.ndarray, quat: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    # Task-space action layout: [x,y,z, quat(xyzw), grip_scalar]
    if gripper.ndim > 1:
        g = gripper.mean(axis=-1, keepdims=True)
    else:
        g = gripper.reshape(-1, 1)
    return np.concatenate([pos, quat, g.astype(np.float32)], axis=-1)


def _augment_single_demo(
    demo: Dict[str, np.ndarray],
    approach_variants: int,
    path_variants: int,
    rng: np.random.Generator,
) -> List[Dict[str, np.ndarray]]:
    pos = demo["eef_pos"]
    quat = demo["eef_quat"]
    grip = demo["gripper"]

    kps = extract_pickplace_keypoints(pos, grip)
    a0 = kps["approach_start"]
    g = kps["grasp"]
    tr = kps["transport"]
    rel = kps["release"]

    grasp_pos = pos[g]
    approach_starts = diversify_approach(grasp_pos, theta_bins=7, phi_bins=8, offset=0.08, rng=rng)
    if len(approach_starts) > approach_variants:
        sel = rng.choice(len(approach_starts), size=approach_variants, replace=False)
        approach_starts = approach_starts[sel]

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
    for ap in approach_starts:
        for pth in transport_paths:
            pos_aug = pos.copy()

            approach_len = max(2, g - a0 + 1)
            alpha = np.linspace(0.0, 1.0, approach_len, dtype=np.float32)[:, None]
            approach_curve = (1 - alpha) * ap[None, :] + alpha * grasp_pos[None, :]
            pos_aug[a0 : g + 1] = approach_curve

            if rel >= tr:
                nseg = rel - tr + 1
                if pth.shape[0] != nseg:
                    idx = np.linspace(0, pth.shape[0] - 1, nseg).astype(np.int64)
                    pos_aug[tr : rel + 1] = pth[idx]
                else:
                    pos_aug[tr : rel + 1] = pth

            action_aug = _compose_action_from_pose(pos_aug, quat, grip)
            out.append(
                {
                    "eef_pos": pos_aug,
                    "eef_quat": quat,
                    "gripper": grip,
                    "actions": action_aug,
                }
            )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input low_dim_abs.hdf5")
    parser.add_argument("--output", required=True, help="Output augmented hdf5")
    parser.add_argument("--approach_variants", type=int, default=30)
    parser.add_argument("--path_variants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        g_data_in = fin["data"]
        g_data_out = fout.create_group("data")

        demo_keys = sorted(list(g_data_in.keys()))
        demo_out_idx = 0

        for key in demo_keys:
            demo = _read_demo(g_data_in[key])
            augmented = _augment_single_demo(
                demo,
                approach_variants=args.approach_variants,
                path_variants=args.path_variants,
                rng=rng,
            )

            for sample in augmented:
                grp = g_data_out.create_group(f"demo_{demo_out_idx}")
                obs_grp = grp.create_group("obs")
                obs_grp.create_dataset("robot0_eef_pos", data=sample["eef_pos"], compression="gzip")
                obs_grp.create_dataset("robot0_eef_quat", data=sample["eef_quat"], compression="gzip")
                obs_grp.create_dataset("robot0_gripper_qpos", data=sample["gripper"], compression="gzip")
                grp.create_dataset("actions", data=sample["actions"], compression="gzip")
                demo_out_idx += 1

        fout.attrs["augmentation"] = "TSTD"
        fout.attrs["source_file"] = args.input
        fout.attrs["num_demos"] = demo_out_idx

    print(f"[TSTD] done: {args.output} | demos={demo_out_idx}")


if __name__ == "__main__":
    main()
