#!/usr/bin/env python3
"""
Compute normalization stats (mean/std) for virtual HKM k(q) distribution.

This should be used when training with k_feature_mode=virtual to avoid
base-vs-virtual normalization mismatch.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
import torch

from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule
from kc_dp.kinematics.virtual_sampler import VirtualRobotSampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/robomimic/datasets/can/ph/low_dim_abs.hdf5")
    ap.add_argument("--urdf", default="data/urdf/franka_panda/urdf/panda.urdf")
    ap.add_argument("--ee", default="panda_link8")
    ap.add_argument("--max_dof", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_episodes", type=int, default=0, help="0 means all")
    ap.add_argument("--repeats", type=int, default=5, help="number of virtual re-samples per episode")
    ap.add_argument("--out", default="data/k_q_stats_virtual.pt")
    args = ap.parse_args()

    np.random.seed(args.seed)

    base = AnalyticKinematicModule(
        urdf_path=args.urdf,
        ee_frame_name=args.ee,
        max_dof=args.max_dof,
    )
    sampler = VirtualRobotSampler(
        base_urdf_path=args.urdf,
        ee_frame_name=args.ee,
        max_dof=args.max_dof,
    )

    all_k = []

    with h5py.File(args.dataset, "r") as f:
        demos = f["data"]
        n_ep = len(demos)
        if args.max_episodes > 0:
            n_ep = min(n_ep, args.max_episodes)

        for ep in range(n_ep):
            q = demos[f"demo_{ep}"]["obs"]["robot0_joint_pos"][:].astype(np.float64)
            T, n_dof = q.shape

            dq = np.zeros_like(q)
            if T > 1:
                dq[:-1] = q[1:] - q[:-1]
                dq[-1] = dq[-2]

            dpose = np.zeros((T, 6), dtype=np.float64)
            local_data = base.model.createData()
            for t in range(T):
                q_full = base._pad_q(q[t])
                pin.framesForwardKinematics(base.model, local_data, q_full)
                J_full = pin.computeFrameJacobian(
                    base.model,
                    local_data,
                    q_full,
                    base.ee_frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                n_eff = min(n_dof, len(base._arm_v_indices))
                J_arm = J_full[:, base._arm_v_indices[:n_eff]]
                dpose[t] = J_arm @ dq[t][:n_eff]

            q_lo = np.min(q, axis=0)
            q_hi = np.max(q, axis=0)
            for _ in range(max(1, args.repeats)):
                q_min_v, q_max_v = sampler.sample_stage1_limits(q_lo, q_hi)
                vmod, q_min_v, q_max_v = sampler.sample_stage2_module(q, dpose, q_min_v, q_max_v)
                kv = vmod.compute_k_q_with_custom_limits(q, q_min_v, q_max_v)
                all_k.append(kv)

    k = np.concatenate(all_k, axis=0)
    mean = k.mean(axis=0).astype(np.float32)
    std = np.clip(k.std(axis=0), 1e-6, None).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, out)

    print(f"saved: {out}")
    print(f"shape: {k.shape}, std min/med/max: {std.min():.6f}/{np.median(std):.6f}/{std.max():.6f}")


if __name__ == "__main__":
    main()
