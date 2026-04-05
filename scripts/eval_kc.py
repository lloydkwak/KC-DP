"""
Cross-Embodiment Evaluation Framework for KC-DP
================================================================
Fixes Controller Gain Instability by preserving native dataset configs for Panda.
"""

import os
import sys
import json
import click
import torch
import numpy as np
import multiprocessing
from kc_dp.kinematics.kq_feature import KQFeatureComputer
from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule

try:
    multiprocessing.set_start_method('spawn', force=True)
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party", "diffusion_policy"))

os.environ.setdefault("MUJOCO_GL", "egl")

from omegaconf import OmegaConf
try:
    OmegaConf.register_new_resolver("load_stats", lambda path, key: torch.load(path, map_location='cpu')[key] if os.path.exists(path) else None, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass

ROBOT_REGISTRY = {
    'Panda': {
        'urdf_path': 'data/urdf/franka_panda/urdf/panda.urdf',
        'ee_frame': 'panda_link8',
        'n_arm_joints': 7,
        'grip_open': np.array([0.001, -0.001]),
        'grip_closed': np.array([0.039, -0.039])
    },
    'IIWA': {
        'urdf_path': 'data/urdf/kuka_iiwa/urdf/iiwa7.urdf',
        'ee_frame': 'lbr_iiwa_link_7', 
        'n_arm_joints': 7,
        'grip_open': np.array([0.49, 0.31, -0.36, -0.47, 0.04, -0.32]),
        'grip_closed': np.array([0.01, 0.04, 0.06, -0.01, 0.02, 0.07])
    },
    'Jaco': {
        'urdf_path': 'data/urdf/jaco/robots/kinova.urdf',
        'ee_frame': 'j2s6s200_link_6',  
        'n_arm_joints': 6,             
        'grip_open': np.array([0.60, 0.05, 0.60, 0.05, 0.60, 0.05]),
        'grip_closed': np.array([0.75, 0.14, 0.75, 0.14, 0.75, 0.14])
    },
    'UR5e': {
        'urdf_path': 'data/urdf/ur5/urdf/ur5_robot.urdf',
        'ee_frame': 'tool0',         
        'n_arm_joints': 6,
        'grip_open': np.array([0.52, -0.10, 0.25, 0.51, -0.15, 0.26]),
        'grip_closed': np.array([-0.02, -0.25, -0.21, -0.02, -0.25, -0.21])
    }
}

TARGET_ROBOT = 'Panda'
TARGET_TASK = None
VANILLA_MODE = False
_KQ_COMP = None
_GRIPPER_DEBUG_PRINTED = False
_KQ_CLIP = 3.0
_KQ_SCALE = 1.0
_USE_DOF_MASK = False
_EXPECTED_OBS_DIM = None
_DATASET_PATH_DEBUG_PRINTED = False
_REF_L_CHAR = None


def _resolve_dataset_path(dataset_path: str) -> str:
    if dataset_path is None:
        raise FileNotFoundError("dataset_path is None")

    path = str(dataset_path)
    if os.path.exists(path):
        return path

    candidates = []

    # 1) common remap: data/robomimic -> third_party/diffusion_policy/data/robomimic
    if path.startswith("data/robomimic/"):
        candidates.append(
            path.replace(
                "data/robomimic/",
                "third_party/diffusion_policy/data/robomimic/",
                1,
            )
        )

    # 2) project-root anchored
    candidates.append(os.path.join(_PROJECT_ROOT, path))

    # 3) project-root anchored remap
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

    raise FileNotFoundError(
        f"Dataset not found: '{path}'. Tried: {[path] + candidates}"
    )

def normalize_gripper(grip_qpos, grip_open, grip_closed):
    range_q = grip_open - grip_closed
    safe_range = np.where(np.abs(range_q) < 1e-6, 1.0, range_q)
    per_joint = np.clip((grip_qpos - grip_closed) / safe_range, 0.0, 1.0)
    openness = np.mean(per_joint)
    panda_open, panda_closed = np.array([0.001, -0.001]), np.array([0.039, -0.039])
    return panda_closed + openness * (panda_open - panda_closed)


def get_hkm_feature(q_pos):
    global _KQ_COMP
    n_joints = len(_KQ_COMP.q_min_base)
    q = np.asarray(q_pos)[:n_joints].astype(np.float64)
    k = _KQ_COMP.compute_normalized(q)

    # For robots with fewer than max_dof joints (e.g., 6-DoF),
    # neutralize missing periodic channels to avoid dead-signal OOD.
    # periodic block indices: [21:35] = [sin(7), cos(7)]
    if n_joints < 7:
        for j in range(n_joints, 7):
            k[21 + j] = 0.0       # sin(j)
            k[21 + 7 + j] = 0.0   # cos(j)

    k = np.clip(k, -_KQ_CLIP, _KQ_CLIP)
    return (k * float(_KQ_SCALE)).astype(np.float32)


def get_dof_mask_for_robot(robot_name: str) -> np.ndarray:
    n = int(ROBOT_REGISTRY[robot_name]['n_arm_joints'])
    mask = np.zeros(7, dtype=np.float32)
    mask[: min(n, 7)] = 1.0
    return mask


def _get_reference_l_char() -> float:
    global _REF_L_CHAR
    if _REF_L_CHAR is None:
        km = AnalyticKinematicModule(
            urdf_path=os.path.join(_PROJECT_ROOT, ROBOT_REGISTRY['Panda']['urdf_path']),
            ee_frame_name=ROBOT_REGISTRY['Panda']['ee_frame'],
            max_dof=7,
        )
        _REF_L_CHAR = float(km.L_char)
    return _REF_L_CHAR

import robomimic.utils.file_utils as FileUtils

_orig_get_env_meta = FileUtils.get_env_metadata_from_dataset

def patched_get_env_meta(dataset_path):
    global _DATASET_PATH_DEBUG_PRINTED
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    if (not _DATASET_PATH_DEBUG_PRINTED) and (resolved_dataset_path != dataset_path):
        print(f"[INFO] Resolved dataset_path: {dataset_path} -> {resolved_dataset_path}")
        _DATASET_PATH_DEBUG_PRINTED = True

    meta = _orig_get_env_meta(resolved_dataset_path)
    if TARGET_ROBOT != 'Panda':
        meta['env_kwargs']['robots'] = [TARGET_ROBOT]
        # Keep dataset controller config as-is; only switch robot identity.
        meta['env_kwargs']['gripper_types'] = "default"
        
    if TARGET_TASK is not None:
        task_map = {'can': 'PickPlaceCan', 'lift': 'Lift', 'square': 'NutAssemblySquare'}
        mapped_task = task_map.get(TARGET_TASK.lower())
        if mapped_task:
            meta['env_name'] = mapped_task

    meta['env_kwargs']['has_offscreen_renderer'] = True
    meta['env_kwargs']['has_renderer'] = False
    return meta

FileUtils.get_env_metadata_from_dataset = patched_get_env_meta

import robomimic.utils.env_utils as EnvUtils
_orig_create_env = EnvUtils.create_env_from_metadata

def patched_create_env(env_meta, env_name=None, render=False, render_offscreen=False, use_image_obs=False, **kwargs):
    return _orig_create_env(env_meta=env_meta, env_name=env_name, render=render, render_offscreen=True, use_image_obs=use_image_obs, **kwargs)
EnvUtils.create_env_from_metadata = patched_create_env

from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

def patched_get_obs(self):
    global _GRIPPER_DEBUG_PRINTED, _USE_DOF_MASK
    cfg = ROBOT_REGISTRY[TARGET_ROBOT]
    raw_obs = self.env.get_observation()
    parts = []
    
    for key in self.obs_keys:
        val = raw_obs[key]
        if key == 'robot0_gripper_qpos':
            val = np.asarray(val)
            if TARGET_ROBOT != 'Panda':
                if not _GRIPPER_DEBUG_PRINTED:
                    print(
                        f"[DEBUG] {TARGET_ROBOT} raw robot0_gripper_qpos: "
                        f"shape={val.shape}, min={val.min():.4f}, max={val.max():.4f}"
                    )
                    _GRIPPER_DEBUG_PRINTED = True
                if val.shape != cfg['grip_open'].shape:
                    print(
                        f"[WARN] {TARGET_ROBOT} gripper shape mismatch: "
                        f"obs={val.shape}, registry={cfg['grip_open'].shape}. "
                        f"Using registry grip_open fallback."
                    )
                    val = cfg['grip_open']
                val = normalize_gripper(val, cfg['grip_open'], cfg['grip_closed'])
        parts.append(val)
        
    obs = np.concatenate(parts, axis=0).astype(np.float32)
    
    if not VANILLA_MODE:
        actual_q = raw_obs['robot0_joint_pos']
        obs = np.concatenate([obs, get_hkm_feature(actual_q)])
        if _USE_DOF_MASK:
            obs = np.concatenate([obs, get_dof_mask_for_robot(TARGET_ROBOT)])
    return obs
RobomimicLowdimWrapper.get_observation = patched_get_obs

_orig_step = RobomimicLowdimWrapper.step
def patched_step(self, action):
    _, reward, done, info = _orig_step(self, action)
    obs = self.get_observation()
    return obs, reward, done, info
RobomimicLowdimWrapper.step = patched_step


@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint.ckpt')
@click.option('-o', '--output_dir', required=True, help='Directory to save results')
@click.option('-r', '--robot', default='Panda', help='Eval Robot (Panda, IIWA, Jaco, UR5e)')
@click.option('-t', '--task', default=None, help='Override Task')
@click.option('-n', '--n_test', default=50, help='Number of rollouts')
@click.option('--k_scale', default=None, type=float, help='Scale for k(q) feature injection (KC mode only). If omitted, uses checkpoint cfg.task.dataset.k_feature_scale when available.')
@click.option('--no_kq', is_flag=True, help='Run in Vanilla DP mode')
def main(checkpoint, output_dir, robot, task, n_test, k_scale, no_kq):
    global TARGET_ROBOT, TARGET_TASK, VANILLA_MODE, _KQ_COMP, _GRIPPER_DEBUG_PRINTED, _KQ_SCALE, _USE_DOF_MASK, _EXPECTED_OBS_DIM
    TARGET_ROBOT = robot
    TARGET_TASK = task
    VANILLA_MODE = no_kq
    _KQ_SCALE = 1.0
    _GRIPPER_DEBUG_PRINTED = False

    import dill
    import hydra
    import diffusion_policy.env_runner.robomimic_lowdim_runner as runner_mod
    from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
    runner_mod.AsyncVectorEnv = SyncVectorEnv

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']

    # Align k feature scale with training config by default.
    if k_scale is None:
        try:
            _KQ_SCALE = float(getattr(cfg.task.dataset, 'k_feature_scale', 1.0))
        except Exception:
            _KQ_SCALE = 1.0
    else:
        _KQ_SCALE = float(k_scale)

    try:
        _EXPECTED_OBS_DIM = int(getattr(cfg.policy, 'obs_dim', getattr(cfg, 'obs_dim', 0)))
    except Exception:
        _EXPECTED_OBS_DIM = None

    base_obs_dim = int(getattr(cfg.task, 'obs_dim', 23))
    _USE_DOF_MASK = bool((not VANILLA_MODE) and (_EXPECTED_OBS_DIM is not None) and (_EXPECTED_OBS_DIM >= (base_obs_dim + 49)))
    if (not VANILLA_MODE) and _USE_DOF_MASK:
        print(f"[INFO] KC eval input mode: k(q)+dof_mask (expected_obs_dim={_EXPECTED_OBS_DIM})")
    elif not VANILLA_MODE:
        print(f"[INFO] KC eval input mode: k(q) only (expected_obs_dim={_EXPECTED_OBS_DIM})")

    if not VANILLA_MODE:
        mix67_stat_path = os.path.join(_PROJECT_ROOT, "data", "k_q_stats_virtual_mix67.pt")
        virtual_stat_path = os.path.join(_PROJECT_ROOT, "data", "k_q_stats_virtual.pt")
        base_stat_path = os.path.join(_PROJECT_ROOT, "data", "k_q_stats.pt")
        if os.path.exists(mix67_stat_path):
            stat_path = mix67_stat_path
        elif os.path.exists(virtual_stat_path):
            stat_path = virtual_stat_path
        else:
            stat_path = base_stat_path

        if not os.path.exists(stat_path):
            print(f"\n[CRITICAL ERROR] '{stat_path}' not found!")
            sys.exit(1)

        cfg_r = ROBOT_REGISTRY[robot]
        _KQ_COMP = KQFeatureComputer(
            urdf_path=os.path.join(_PROJECT_ROOT, cfg_r['urdf_path']),
            ee_frame_name=cfg_r['ee_frame'],
            max_dof=7,
            stats_path=stat_path,
            use_physical_limits=True,
        )
        # Keep k(q) scaling aligned with training (Panda-based virtual stats).
        ref_l = _get_reference_l_char()
        _KQ_COMP.km.L_char = ref_l
        _KQ_COMP.km.S_matrix = np.diag([1.0 / ref_l] * 3 + [1.0] * 3)
        print(
            f"\n[DEBUG] Successfully loaded k_q_stats! "
            f"(Mean sum: {_KQ_COMP.k_mean.sum():.2f}, Std sum: {_KQ_COMP.k_std.sum():.2f})"
        )
    try:
        resolved_dataset_path = _resolve_dataset_path(cfg.task.env_runner.dataset_path)
        cfg.task.env_runner.dataset_path = resolved_dataset_path
        if hasattr(cfg.task, 'dataset') and hasattr(cfg.task.dataset, 'dataset_path'):
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
    
    print(f"\n[EVAL START] Robot: {robot} | Mode: {'Vanilla' if no_kq else 'KC-DP'} | Task: {task}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'media'), exist_ok=True)
    
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    policy.eval()
    
    runner_log = env_runner.run(policy)
    
    res_path = os.path.join(output_dir, 'eval_log.json')
    log_data = {}
    for k, v in runner_log.items():
        if isinstance(v, (np.floating, float, np.integer, int, bool)):
            log_data[k] = float(v)
    log_data.update({'robot': robot, 'task_override': task, 'mode': 'vanilla' if no_kq else 'kc_dp', 'ckpt': checkpoint})
    
    with open(res_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"[EVAL DONE] Success Rate: {log_data.get('test/mean_score', 0):.4f} saved to {res_path}")

if __name__ == '__main__':
    main()