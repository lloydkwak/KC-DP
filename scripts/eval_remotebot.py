"""
Cross-Embodiment Evaluation Framework (RemoteBot)
==================================================
- Uses task-space policy input directly (no k(q) injection).
- Keeps robot swap + controller safety logic from KC eval script.
"""

import copy
import json
import os
import sys
import multiprocessing
from typing import Optional, Tuple

import click
import numpy as np
import torch

try:
    multiprocessing.set_start_method("spawn", force=True)
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party", "diffusion_policy"))

os.environ.setdefault("MUJOCO_GL", "egl")

from omegaconf import OmegaConf
from remotebot.robotics.feasibility_oracle import FeasibilityOracle

try:
    OmegaConf.register_new_resolver(
        "load_stats",
        lambda path, key: torch.load(path, map_location="cpu")[key] if os.path.exists(path) else None,
        replace=True,
    )
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass


ROBOT_REGISTRY = {
    "Panda": {
        "grip_open": np.array([0.001, -0.001]),
        "grip_closed": np.array([0.039, -0.039]),
        "ee_frame": "gripper0_grip_site",
        "workspace_bounds": ((-0.8, -0.8, 0.0), (0.8, 0.8, 1.2)),
        "urdf_candidates": [
            "third_party/robosuite/robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm.urdf",
            "third_party/robosuite/robosuite/models/assets/robots/panda/robot.xml",
        ],
    },
    "IIWA": {
        "grip_open": np.array([0.49, 0.31, -0.36, -0.47, 0.04, -0.32]),
        "grip_closed": np.array([0.01, 0.04, 0.06, -0.01, 0.02, 0.07]),
        "ee_frame": "right_hand",
        "workspace_bounds": ((-0.9, -0.9, 0.0), (0.9, 0.9, 1.3)),
        "urdf_candidates": [
            "third_party/robosuite/robosuite/models/assets/bullet_data/iiwa_description/urdf/iiwa14.urdf",
        ],
    },
    "Jaco": {
        "grip_open": np.array([0.60, 0.05, 0.60, 0.05, 0.60, 0.05]),
        "grip_closed": np.array([0.75, 0.14, 0.75, 0.14, 0.75, 0.14]),
        "ee_frame": "j2s7s300_ee_link",
        "workspace_bounds": ((-0.9, -0.9, 0.0), (0.9, 0.9, 1.3)),
        "urdf_candidates": [
            "third_party/robosuite/robosuite/models/assets/bullet_data/jaco_description/urdf/j2s7s300.urdf",
        ],
    },
    "UR5e": {
        "grip_open": np.array([0.52, -0.10, 0.25, 0.51, -0.15, 0.26]),
        "grip_closed": np.array([-0.02, -0.25, -0.21, -0.02, -0.25, -0.21]),
        "ee_frame": "ee_link",
        "workspace_bounds": ((-1.0, -1.0, 0.0), (1.0, 1.0, 1.4)),
        "urdf_candidates": [
            "third_party/robosuite/robosuite/models/assets/bullet_data/ur5e_description/urdf/ur5e.urdf",
        ],
    },
}


TARGET_ROBOT = "Panda"
TARGET_TASK = None
_GRIPPER_DEBUG_PRINTED = False
_ROBOT_OSC_CONFIG_CACHE = {}
_OSC_DEBUG_PRINTED = set()
_DATASET_PATH_DEBUG_PRINTED = False


def _resolve_existing_path(path_candidates) -> Optional[str]:
    for rel in path_candidates:
        if os.path.isabs(rel) and os.path.exists(rel):
            return rel
        abs_path = os.path.join(_PROJECT_ROOT, rel)
        if os.path.exists(abs_path):
            return abs_path
    return None


def _build_oracle_for_robot(robot_name: str, device: torch.device) -> FeasibilityOracle:
    robot_cfg = ROBOT_REGISTRY.get(robot_name, {})
    ws_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = robot_cfg.get(
        "workspace_bounds",
        ((-0.8, -0.8, 0.0), (0.8, 0.8, 1.2)),
    )
    urdf_path = _resolve_existing_path(robot_cfg.get("urdf_candidates", []))
    ee_frame = robot_cfg.get("ee_frame", None)

    oracle = FeasibilityOracle(
        workspace_bounds=ws_bounds,
        urdf_path=urdf_path,
        ee_frame_name=ee_frame,
        device=str(device),
    )

    if urdf_path is None:
        print(
            f"[WARN] No URDF found for {robot_name}. "
            "Oracle will run in workspace/smoothness fallback mode."
        )
    elif oracle.pk_chain is None:
        print(
            f"[WARN] URDF resolved for {robot_name} but PK chain init failed. "
            "Falling back to workspace/smoothness oracle."
        )
    else:
        print(f"[INFO] Oracle loaded PK chain for {robot_name}: urdf={urdf_path}, ee_frame={ee_frame}")

    return oracle


def _apply_joint_lock(oracle: FeasibilityOracle, lock_joint: Optional[str]):
    if not lock_joint:
        return
    try:
        idx_str, val_str = lock_joint.split(":", 1)
        jidx = int(idx_str.strip())
        jval = float(val_str.strip())
        oracle.set_joint_fault(jidx, jval)
        print(f"[INFO] Joint fault lock applied: joint[{jidx}]={jval}")
    except Exception as e:
        print(f"[WARN] Invalid --lock_joint format '{lock_joint}'. Expected 'index:value'. error={e}")


_ROBOT_OSC_PRESETS = {
    name: {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "input_type": "delta",
        "input_ref_frame": "base",
        "interpolation": None,
        "ramp_ratio": 0.2,
    }
    for name in ["Panda", "IIWA", "Jaco", "UR5e"]
}


def _resolve_dataset_path(dataset_path: str) -> str:
    if dataset_path is None:
        raise FileNotFoundError("dataset_path is None")
    path = str(dataset_path)
    if os.path.exists(path):
        return path

    candidates = []
    if path.startswith("data/robomimic/"):
        candidates.append(
            path.replace(
                "data/robomimic/",
                "third_party/diffusion_policy/data/robomimic/",
                1,
            )
        )

    candidates.append(os.path.join(_PROJECT_ROOT, path))

    if "data/robomimic/" in path:
        rel = path.split("data/robomimic/", 1)[1]
        candidates.append(
            os.path.join(
                _PROJECT_ROOT,
                "third_party",
                "diffusion_policy",
                "data",
                "robomimic",
                rel,
            )
        )

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(f"Dataset not found: '{path}'. Tried: {[path] + candidates}")


def normalize_gripper(grip_qpos, grip_open, grip_closed):
    range_q = grip_open - grip_closed
    safe_range = np.where(np.abs(range_q) < 1e-6, 1.0, range_q)
    per_joint = np.clip((grip_qpos - grip_closed) / safe_range, 0.0, 1.0)
    openness = np.mean(per_joint)
    panda_open, panda_closed = np.array([0.001, -0.001]), np.array([0.039, -0.039])
    return panda_closed + openness * (panda_open - panda_closed)


def _extract_arm_controller_config(controller_cfg):
    if not isinstance(controller_cfg, dict):
        return None
    if "type" in controller_cfg:
        return copy.deepcopy(controller_cfg)

    body_parts = controller_cfg.get("body_parts", {})
    if isinstance(body_parts, dict):
        right = body_parts.get("right")
        if isinstance(right, dict) and "type" in right:
            return copy.deepcopy(right)
        arms = body_parts.get("arms", {})
        if isinstance(arms, dict):
            right = arms.get("right")
            if isinstance(right, dict) and "type" in right:
                return copy.deepcopy(right)
    return None


def _build_robot_specific_osc_config(robot_name: str, dataset_controller_cfg: dict) -> dict:
    if robot_name in _ROBOT_OSC_CONFIG_CACHE:
        return copy.deepcopy(_ROBOT_OSC_CONFIG_CACHE[robot_name])

    cfg = None
    source = "unknown"

    try:
        from robosuite.controllers import load_composite_controller_config

        loaded = load_composite_controller_config(controller=None, robot=robot_name)
        cfg = _extract_arm_controller_config(loaded)
        if cfg is not None:
            source = "robosuite_default"
    except Exception:
        cfg = None

    if cfg is None and robot_name in _ROBOT_OSC_PRESETS:
        cfg = copy.deepcopy(_ROBOT_OSC_PRESETS[robot_name])
        source = "repo_fallback_preset"

    if cfg is None and isinstance(dataset_controller_cfg, dict):
        cfg = copy.deepcopy(dataset_controller_cfg)
        source = "dataset_controller"

    if cfg is None:
        try:
            from robosuite.controllers import load_part_controller_config

            cfg = load_part_controller_config(default_controller="OSC_POSE")
            source = "robosuite_default_osc_pose"
        except Exception:
            cfg = {}
            source = "empty_default"

    if not isinstance(cfg, dict):
        cfg = {}

    cfg["type"] = "OSC_POSE"
    if isinstance(dataset_controller_cfg, dict):
        for key in [
            "input_max",
            "input_min",
            "output_max",
            "output_min",
            "control_delta",
            "interpolation",
            "ramp_ratio",
        ]:
            if key in dataset_controller_cfg and key not in cfg:
                cfg[key] = copy.deepcopy(dataset_controller_cfg[key])

    cfg["_remotebot_source"] = source
    _ROBOT_OSC_CONFIG_CACHE[robot_name] = copy.deepcopy(cfg)
    return cfg


import robomimic.utils.file_utils as FileUtils

_orig_get_env_meta = FileUtils.get_env_metadata_from_dataset


def patched_get_env_meta(dataset_path):
    global _DATASET_PATH_DEBUG_PRINTED
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    if (not _DATASET_PATH_DEBUG_PRINTED) and (resolved_dataset_path != dataset_path):
        print(f"[INFO] Resolved dataset_path: {dataset_path} -> {resolved_dataset_path}")
        _DATASET_PATH_DEBUG_PRINTED = True

    meta = _orig_get_env_meta(resolved_dataset_path)
    env_kwargs = meta.setdefault("env_kwargs", {})
    dataset_cc = env_kwargs.get("controller_configs", {})

    env_kwargs["robots"] = [TARGET_ROBOT]
    env_kwargs["gripper_types"] = "default"
    env_kwargs["controller_configs"] = _build_robot_specific_osc_config(TARGET_ROBOT, dataset_cc)

    if TARGET_ROBOT not in _OSC_DEBUG_PRINTED:
        cc = env_kwargs.get("controller_configs", {})
        print(
            f"[INFO] Controller config for {TARGET_ROBOT}: "
            f"type={cc.get('type')}, control_delta={cc.get('control_delta', None)}, "
            f"kp={cc.get('kp', None)}, source={cc.get('_remotebot_source', 'unknown')}"
        )
        _OSC_DEBUG_PRINTED.add(TARGET_ROBOT)

    if TARGET_TASK is not None:
        task_map = {"can": "PickPlaceCan", "lift": "Lift", "square": "NutAssemblySquare"}
        mapped_task = task_map.get(TARGET_TASK.lower())
        if mapped_task:
            meta["env_name"] = mapped_task

    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["has_renderer"] = False
    return meta


FileUtils.get_env_metadata_from_dataset = patched_get_env_meta

import robomimic.utils.env_utils as EnvUtils

_orig_create_env = EnvUtils.create_env_from_metadata


def patched_create_env(
    env_meta,
    env_name=None,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
    **kwargs,
):
    return _orig_create_env(
        env_meta=env_meta,
        env_name=env_name,
        render=render,
        render_offscreen=True,
        use_image_obs=use_image_obs,
        **kwargs,
    )


EnvUtils.create_env_from_metadata = patched_create_env

from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper


def patched_get_obs(self):
    global _GRIPPER_DEBUG_PRINTED
    cfg = ROBOT_REGISTRY[TARGET_ROBOT]
    raw_obs = self.env.get_observation()
    parts = []

    for key in self.obs_keys:
        val = raw_obs[key]
        if key == "robot0_gripper_qpos":
            val = np.asarray(val)
            if TARGET_ROBOT != "Panda":
                if not _GRIPPER_DEBUG_PRINTED:
                    print(
                        f"[DEBUG] {TARGET_ROBOT} raw robot0_gripper_qpos: "
                        f"shape={val.shape}, min={val.min():.4f}, max={val.max():.4f}"
                    )
                    _GRIPPER_DEBUG_PRINTED = True
                if val.shape != cfg["grip_open"].shape:
                    print(
                        f"[WARN] {TARGET_ROBOT} gripper shape mismatch: "
                        f"obs={val.shape}, registry={cfg['grip_open'].shape}. "
                        f"Using registry grip_open fallback."
                    )
                    val = cfg["grip_open"]
                val = normalize_gripper(val, cfg["grip_open"], cfg["grip_closed"])
        parts.append(val)

    return np.concatenate(parts, axis=0).astype(np.float32)


RobomimicLowdimWrapper.get_observation = patched_get_obs

_orig_step = RobomimicLowdimWrapper.step


def patched_step(self, action):
    _, reward, done, info = _orig_step(self, action)
    obs = self.get_observation()
    return obs, reward, done, info


RobomimicLowdimWrapper.step = patched_step


@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to checkpoint.ckpt")
@click.option("-o", "--output_dir", required=True, help="Directory to save results")
@click.option("-r", "--robot", default="Panda", help="Eval Robot (Panda, IIWA, Jaco, UR5e)")
@click.option("-t", "--task", default=None, help="Override Task")
@click.option("-n", "--n_test", default=50, help="Number of rollouts")
@click.option("--guidance_weight", default=0.0, type=float, help="Feasibility guidance base weight (0 to disable)")
@click.option("--lock_joint", default=None, type=str, help="Optional joint fault lock as 'joint_idx:value'")
def main(checkpoint, output_dir, robot, task, n_test, guidance_weight, lock_joint):
    global TARGET_ROBOT, TARGET_TASK, _GRIPPER_DEBUG_PRINTED
    TARGET_ROBOT = robot
    TARGET_TASK = task
    _GRIPPER_DEBUG_PRINTED = False

    import dill
    import hydra
    import diffusion_policy.env_runner.robomimic_lowdim_runner as runner_mod
    from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv

    runner_mod.AsyncVectorEnv = SyncVectorEnv

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    try:
        resolved_dataset_path = _resolve_dataset_path(cfg.task.env_runner.dataset_path)
        cfg.task.env_runner.dataset_path = resolved_dataset_path
        if hasattr(cfg.task, "dataset") and hasattr(cfg.task.dataset, "dataset_path"):
            cfg.task.dataset.dataset_path = resolved_dataset_path
    except Exception as e:
        print(f"[WARN] Failed to resolve dataset_path from cfg: {e}")

    cfg.task.env_runner.n_test = n_test
    cfg.task.env_runner.n_test_vis = 5
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_envs = 1

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg=cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    print(f"\n[EVAL START] Robot: {robot} | Mode: RemoteBot | Task: {task}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "media"), exist_ok=True)

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if guidance_weight > 0.0 and hasattr(policy, "set_feasibility_oracle"):
        oracle = _build_oracle_for_robot(robot_name=robot, device=device)
        _apply_joint_lock(oracle, lock_joint)
        policy.set_feasibility_oracle(oracle)
        policy.guidance_weight_base = float(guidance_weight)
        print(f"[INFO] Feasibility guidance enabled (weight={policy.guidance_weight_base:.4f})")
    else:
        print("[INFO] Feasibility guidance disabled (guidance_weight <= 0)")

    policy.to(device)
    policy.eval()

    runner_log = env_runner.run(policy)

    res_path = os.path.join(output_dir, "eval_log.json")
    log_data = {}
    for k, v in runner_log.items():
        if isinstance(v, (np.floating, float, np.integer, int, bool)):
            log_data[k] = float(v)
    log_data.update({"robot": robot, "task_override": task, "mode": "remotebot", "ckpt": checkpoint})

    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    print(f"[EVAL DONE] Success Rate: {log_data.get('test/mean_score', 0):.4f} saved to {res_path}")


if __name__ == "__main__":
    main()
