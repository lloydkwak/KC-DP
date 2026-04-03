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
    'Panda': {'urdf_path': 'data/urdf/franka_panda/urdf/panda.urdf', 'ee_frame': 'panda_link8', 'n_arm_joints': 7, 'grip_open': np.array([0.001, -0.001]), 'grip_closed': np.array([0.039, -0.039])},
    'IIWA': {'urdf_path': 'data/urdf/kuka_iiwa/urdf/iiwa7.urdf', 'ee_frame': 'iiwa_link_7', 'n_arm_joints': 7, 'grip_open': np.array([0.49, 0.31, -0.36, -0.47, 0.04, -0.32]), 'grip_closed': np.array([0.01, 0.04, 0.06, -0.01, 0.02, 0.07])},
    'Jaco': {'urdf_path': 'data/urdf/jaco/robots/kinova.urdf', 'ee_frame': 'j2n6s200_end_effector', 'n_arm_joints': 7, 'grip_open': np.array([0.60, 0.05, 0.60, 0.05, 0.60, 0.05]), 'grip_closed': np.array([0.75, 0.14, 0.75, 0.14, 0.75, 0.14])},
    'UR5e': {'urdf_path': 'data/urdf/ur5/urdf/ur5_robot.urdf', 'ee_frame': 'ee_link', 'n_arm_joints': 6, 'grip_open': np.array([0.52, -0.10, 0.25, 0.51, -0.15, 0.26]), 'grip_closed': np.array([-0.02, -0.25, -0.21, -0.02, -0.25, -0.21])}
}

TARGET_ROBOT = 'Panda'
TARGET_TASK = None
VANILLA_MODE = False
_KM_CACHE = None
_K_MEAN, _K_STD = np.zeros(42), np.ones(42)

def normalize_gripper(grip_qpos, grip_open, grip_closed):
    range_q = grip_open - grip_closed
    safe_range = np.where(np.abs(range_q) < 1e-6, 1.0, range_q)
    per_joint = np.clip((grip_qpos - grip_closed) / safe_range, 0.0, 1.0)
    openness = np.mean(per_joint)
    panda_open, panda_closed = np.array([0.001, -0.001]), np.array([0.039, -0.039])
    return panda_closed + openness * (panda_open - panda_closed)

# Insert this updated function into scripts/eval_kc.py
def get_hkm_feature(q_pos):
    global _KM_CACHE, _K_MEAN, _K_STD
    km, q_min_base, q_max_base, n_joints = _KM_CACHE
    q = np.asarray(q_pos)[:n_joints].astype(np.float64)
    
    # Force physical limits into the Virtual Sampler's 315-degree distribution bounds
    max_range_rad = np.radians(315.0)
    current_range = q_max_base - q_min_base
    
    q_min_eval = np.where(current_range > max_range_rad, q - (max_range_rad / 2), q_min_base)
    q_max_eval = np.where(current_range > max_range_rad, q + (max_range_rad / 2), q_max_base)
    
    raw_kq = km.compute_k_q_with_custom_limits(q[np.newaxis], q_min_eval, q_max_eval)[0]
    
    if len(raw_kq) == 35:
        periodic = raw_kq[:14]
        vech_21 = raw_kq[14:]
        M6 = np.zeros((6, 6))
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                M6[i, j] = M6[j, i] = vech_21[idx]
                idx += 1
        M7 = np.zeros((7, 7))
        M7[:6, :6] = M6
        vech_28 = [M7[i, j] for i in range(7) for j in range(i, 7)]
        raw_kq = np.concatenate([periodic, vech_28])
        
    return ((raw_kq - _K_MEAN) / _K_STD).astype(np.float32)

import robomimic.utils.file_utils as FileUtils
import robosuite

_orig_get_env_meta = FileUtils.get_env_metadata_from_dataset

def patched_get_env_meta(dataset_path):
    meta = _orig_get_env_meta(dataset_path)
    
    # [CRITICAL FIX] Panda일 때는 데이터셋의 튜닝된 제어기를 절대 건드리지 않습니다.
    if TARGET_ROBOT != 'Panda':
        meta['env_kwargs']['robots'] = [TARGET_ROBOT]
        
        # Cross-embodiment의 경우 새로운 기본 제어기를 로드하되, 데이터셋의 Delta 모드 설정은 계승합니다.
        is_delta = True
        orig_cfg = meta['env_kwargs'].get('controller_configs', {})
        if isinstance(orig_cfg, dict) and 'control_delta' in orig_cfg:
            is_delta = orig_cfg['control_delta']
            
        ctrl_config = robosuite.load_controller_config(default_controller="OSC_POSE")
        ctrl_config['control_delta'] = is_delta
        meta['env_kwargs']['controller_configs'] = ctrl_config
        
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
    cfg = ROBOT_REGISTRY[TARGET_ROBOT]
    raw_obs = self.env.get_observation()
    parts = []
    
    for key in self.obs_keys:
        val = raw_obs[key]
        if key == 'robot0_gripper_qpos':
            val = np.asarray(val)
            # Panda는 순정 그리퍼 값을 유지하고, 다른 로봇만 매핑합니다.
            if TARGET_ROBOT != 'Panda':
                if val.shape != cfg['grip_open'].shape:
                    val = cfg['grip_open']
                val = normalize_gripper(val, cfg['grip_open'], cfg['grip_closed'])
        parts.append(val)
        
    obs = np.concatenate(parts, axis=0).astype(np.float32)
    
    if not VANILLA_MODE:
        actual_q = raw_obs['robot0_joint_pos']
        obs = np.concatenate([obs, get_hkm_feature(actual_q)])
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
@click.option('--no_kq', is_flag=True, help='Run in Vanilla DP mode')
def main(checkpoint, output_dir, robot, task, n_test, no_kq):
    global TARGET_ROBOT, TARGET_TASK, VANILLA_MODE, _KM_CACHE, _K_MEAN, _K_STD
    TARGET_ROBOT = robot
    TARGET_TASK = task
    VANILLA_MODE = no_kq
    
    if not VANILLA_MODE:
        stat_path = os.path.join(_PROJECT_ROOT, "data", "k_q_stats.pt")
        if not os.path.exists(stat_path):
            print(f"\n[CRITICAL ERROR] '{stat_path}' not found!")
            sys.exit(1)
            
        st = torch.load(stat_path, map_location='cpu')
        _K_MEAN = np.array(st['mean'], dtype=np.float32)
        _K_STD = np.clip(np.array(st['std'], dtype=np.float32), 1e-6, None)
        print(f"\n[DEBUG] Successfully loaded k_q_stats! (Mean sum: {_K_MEAN.sum():.2f}, Std sum: {_K_STD.sum():.2f})")
            
        cfg_r = ROBOT_REGISTRY[robot]
        from kc_dp.kinematics.feature_extractor import AnalyticKinematicModule
        km = AnalyticKinematicModule(urdf_path=os.path.join(_PROJECT_ROOT, cfg_r['urdf_path']), ee_frame_name=cfg_r['ee_frame'], max_dof=7)
        idx = km._arm_q_indices[:cfg_r['n_arm_joints']]
        q_min = np.where(np.isinf(km.model.lowerPositionLimit[idx]), -2*np.pi, km.model.lowerPositionLimit[idx]).astype(np.float64)
        q_max = np.where(np.isinf(km.model.upperPositionLimit[idx]), 2*np.pi, km.model.upperPositionLimit[idx]).astype(np.float64)
        _KM_CACHE = (km, q_min, q_max, cfg_r['n_arm_joints'])

    import dill
    import hydra
    import diffusion_policy.env_runner.robomimic_lowdim_runner as runner_mod
    from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
    runner_mod.AsyncVectorEnv = SyncVectorEnv

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
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
    log_data = {k: float(v) if isinstance(v, (np.float32, float)) else v for k, v in runner_log.items()}
    log_data.update({'robot': robot, 'task_override': task, 'mode': 'vanilla' if no_kq else 'kc_dp', 'ckpt': checkpoint})
    
    with open(res_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"[EVAL DONE] Success Rate: {log_data.get('test/mean_score', 0):.4f} saved to {res_path}")

if __name__ == '__main__':
    main()