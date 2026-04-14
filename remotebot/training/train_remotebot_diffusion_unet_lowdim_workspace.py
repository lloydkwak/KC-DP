import os
import sys
import json
import subprocess
from pathlib import Path
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel


class TrainRemoteBotDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def _validate_action_mode_consistency(self, cfg: OmegaConf):
        dataset_cfg = cfg.task.dataset
        env_runner_cfg = cfg.task.env_runner

        dataset_path = str(getattr(dataset_cfg, 'dataset_path', ''))
        ds_abs = bool(getattr(dataset_cfg, 'abs_action', False))
        env_abs = bool(getattr(env_runner_cfg, 'abs_action', ds_abs))

        if ds_abs != env_abs:
            raise ValueError(
                f"Inconsistent abs_action between dataset ({ds_abs}) and env_runner ({env_abs})."
            )

        is_abs_path = ('low_dim_abs' in dataset_path)
        if is_abs_path and not ds_abs:
            raise ValueError(
                f"dataset_path points to abs-action file ({dataset_path}) but abs_action=False."
            )
        if (not is_abs_path) and ds_abs and ('low_dim' in dataset_path):
            raise ValueError(
                f"dataset_path points to delta-action file ({dataset_path}) but abs_action=True."
            )

    def _run_cross_robot_eval(self, cfg: OmegaConf, checkpoint_path: str) -> dict:
        """
        Optional periodic cross-robot eval during training.
        Uses scripts/eval_remotebot.py and stores videos under output_dir/cross_eval/.
        """
        tcfg = cfg.training
        if not bool(getattr(tcfg, 'cross_eval_enable', False)):
            return {}

        every = int(getattr(tcfg, 'cross_eval_every', 0))
        if every <= 0 or (self.epoch % every) != 0:
            return {}

        robots = list(getattr(tcfg, 'cross_eval_robots', []))
        if len(robots) == 0:
            return {}

        n_test = int(getattr(tcfg, 'cross_eval_n_test', 10))
        guidance_weight = float(getattr(tcfg, 'cross_eval_guidance_weight', 0.0))
        keep_ckpt = bool(getattr(tcfg, 'cross_eval_keep_checkpoint', False))

        project_root = Path(__file__).resolve().parents[2]
        eval_script = project_root / 'scripts' / 'eval_remotebot.py'

        if not eval_script.exists():
            print(f"[WARN] cross_eval skipped. eval script not found: {eval_script}")
            return {}

        env = os.environ.copy()
        py_path_prefix = os.pathsep.join([
            str(project_root),
            str(project_root / 'third_party' / 'diffusion_policy')
        ])
        env['PYTHONPATH'] = py_path_prefix if not env.get('PYTHONPATH') else f"{py_path_prefix}{os.pathsep}{env['PYTHONPATH']}"

        metrics = {}
        for robot in robots:
            out_dir = Path(self.output_dir) / 'cross_eval' / f'epoch_{self.epoch:04d}' / str(robot)
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(eval_script),
                '-c', str(checkpoint_path),
                '-o', str(out_dir),
                '-r', str(robot),
                '-n', str(n_test),
                '--guidance_weight', str(guidance_weight),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(project_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
                eval_log = out_dir / 'eval_log.json'
                if eval_log.exists():
                    with open(eval_log, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                    score = float(d.get('test/mean_score', 0.0))
                    metrics[f'cross_eval/{robot}/mean_score'] = score
                else:
                    metrics[f'cross_eval/{robot}/mean_score'] = 0.0
                print(f"[cross_eval] {robot} done. output={out_dir}")
            except Exception as e:
                metrics[f'cross_eval/{robot}/error'] = 1.0
                print(f"[WARN] cross_eval failed for {robot}: {e}")

        if not keep_ckpt:
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass

        return metrics

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        self._validate_action_mode_consistency(cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)

        train_loader_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
        if not isinstance(train_loader_kwargs, dict):
            train_loader_kwargs = dict(cfg.dataloader)

        train_sampler = None
        if hasattr(dataset, 'get_train_sampler'):
            train_sampler = dataset.get_train_sampler()
        if train_sampler is not None:
            train_loader_kwargs['sampler'] = train_sampler
            train_loader_kwargs['shuffle'] = False

        train_dataloader = DataLoader(dataset, **train_loader_kwargs)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch, obs_dict, gt_action, result, pred_action, mse

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # Optional: cross-robot eval videos/metrics during training.
                # Save a temporary checkpoint for eval_kc.py, then run selected robots.
                cross_eval_enable = bool(getattr(cfg.training, 'cross_eval_enable', False))
                cross_eval_every = int(getattr(cfg.training, 'cross_eval_every', 0))
                if cross_eval_enable and cross_eval_every > 0 and (self.epoch % cross_eval_every) == 0:
                    tmp_ckpt_path = os.path.join(
                        self.output_dir,
                        'checkpoints',
                        f'cross_eval_epoch_{self.epoch:04d}.ckpt'
                    )
                    os.makedirs(os.path.dirname(tmp_ckpt_path), exist_ok=True)
                    self.save_checkpoint(path=tmp_ckpt_path)
                    cross_metrics = self._run_cross_robot_eval(cfg, tmp_ckpt_path)
                    if len(cross_metrics) > 0:
                        step_log.update(cross_metrics)

                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
