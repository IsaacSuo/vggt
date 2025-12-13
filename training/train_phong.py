# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Phong Training Script

训练MaterialHead和LightHead的脚本
支持:
1. 冻结backbone，只训练MaterialHead和LightHead
2. Phong渲染损失
3. 训练监控和可视化
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vggt.models.vggt import VGGT
from training.rendering.phong_renderer import SimplePhongRenderer
from training.rendering.phong_loss import PhongLossWithRegularization
from training.monitoring.phong_monitor import PhongTrainingMonitor, VisualizationSaver, create_training_summary


class PhongTrainer:
    """
    Phong训练器

    专门用于训练MaterialHead和LightHead
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 创建日志目录
        self.log_dir = Path(config.get('log_dir', './phong_training_logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        # 初始化组件
        self._setup_model()
        self._setup_renderer()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_monitor()

        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        print(f"[PhongTrainer] Initialized. Logs at: {self.log_dir}")

    def _save_config(self):
        """保存训练配置"""
        config_path = self.log_dir / "train_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

    def _setup_model(self):
        """初始化模型"""
        print("[PhongTrainer] Setting up model...")

        # 加载预训练的VGGT
        self.model = VGGT(
            enable_material=True,
            enable_light=True,
        )

        # 加载预训练权重
        pretrained_path = self.config.get('pretrained_checkpoint')
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[PhongTrainer] Loading pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # 处理checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 加载权重 (strict=False 允许新增head的权重缺失)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"[PhongTrainer] Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            if missing:
                print(f"[PhongTrainer] Missing keys (expected for new heads): {missing[:5]}...")

        # 冻结backbone
        self._freeze_backbone()

        self.model.to(self.device)

        # 打印参数统计
        self._print_param_stats()

    def _freeze_backbone(self):
        """冻结backbone参数，只训练MaterialHead和LightHead"""
        freeze_patterns = self.config.get('freeze_patterns', [
            'patch_embed',
            'aggregator',
            'camera_head',
            'point_head',
            'depth_head',
        ])

        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            should_freeze = any(pattern in name for pattern in freeze_patterns)

            if should_freeze:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()

        print(f"[PhongTrainer] Frozen params: {frozen_count/1e6:.2f}M")
        print(f"[PhongTrainer] Trainable params: {trainable_count/1e6:.2f}M")

    def _print_param_stats(self):
        """打印各模块参数统计"""
        modules = {
            'material_head': getattr(self.model, 'material_head', None),
            'light_head': getattr(self.model, 'light_head', None),
        }

        for name, module in modules.items():
            if module is not None:
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"[PhongTrainer] {name}: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

    def _setup_renderer(self):
        """初始化渲染器"""
        renderer_config = self.config.get('renderer', {})
        self.renderer = SimplePhongRenderer(
            ambient_strength=renderer_config.get('ambient_strength', 0.3),
            use_learnable_light=renderer_config.get('use_learnable_light', True),
        )
        self.renderer.to(self.device)  # 移动到GPU
        print("[PhongTrainer] Renderer initialized")

    def _setup_loss(self):
        """初始化损失函数"""
        loss_config = self.config.get('loss', {})
        self.loss_fn = PhongLossWithRegularization(
            weight=loss_config.get('weight', 1.0),
            photometric_loss_type=loss_config.get('photometric_type', 'l1'),
            material_smoothness_weight=loss_config.get('smoothness_weight', 0.01),
            energy_conservation_weight=loss_config.get('energy_weight', 0.01),
        )
        print("[PhongTrainer] Loss function initialized")

    def _setup_optimizer(self):
        """初始化优化器"""
        optim_config = self.config.get('optimizer', {})

        # 收集可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=optim_config.get('lr', 1e-4),
            weight_decay=optim_config.get('weight_decay', 0.01),
            betas=optim_config.get('betas', (0.9, 0.999)),
        )

        # 学习率调度器
        scheduler_config = optim_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('max_epochs', 100),
                eta_min=scheduler_config.get('min_lr', 1e-6),
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1),
            )
        else:
            self.scheduler = None

        print(f"[PhongTrainer] Optimizer: AdamW, LR: {optim_config.get('lr', 1e-4)}")

    def _setup_monitor(self):
        """初始化监控器"""
        monitor_config = self.config.get('monitor', {})

        self.monitor = PhongTrainingMonitor(
            log_dir=str(self.log_dir),
            experiment_name=self.config.get('experiment_name', None),
            use_tensorboard=monitor_config.get('use_tensorboard', True),
            save_interval=monitor_config.get('save_interval', 100),
            viz_interval=monitor_config.get('viz_interval', 500),
        )

        self.viz_saver = VisualizationSaver(
            save_dir=str(self.log_dir / "visualizations")
        )

        print("[PhongTrainer] Monitor initialized")

    def train_step(self, batch: dict) -> dict:
        """
        单步训练

        Args:
            batch: 包含 'images' 的数据字典

        Returns:
            loss_dict: 损失字典
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 获取输入
        images = batch['images'].to(self.device)  # (B, S, C, H, W)

        # 前向传播
        with torch.cuda.amp.autocast(enabled=self.config.get('use_amp', False)):
            outputs = self.model(images=images)

            # 获取材质 (模型直接输出各属性，需要收集成dict)
            # 输出格式: (B, S, C, H, W) -> 需要转为 (B, S, H, W, C)
            materials = {}
            if 'diffuse' in outputs:
                materials['diffuse'] = outputs['diffuse'].permute(0, 1, 3, 4, 2)  # (B, S, H, W, 3)
            if 'specular' in outputs:
                materials['specular'] = outputs['specular'].permute(0, 1, 3, 4, 2)
            if 'roughness' in outputs:
                materials['roughness'] = outputs['roughness'].permute(0, 1, 3, 4, 2)
            if 'ambient_occlusion' in outputs:
                materials['ao'] = outputs['ambient_occlusion'].permute(0, 1, 3, 4, 2)

            # 获取光照参数
            light_params = {}
            if 'light_direction' in outputs:
                light_params['light_direction'] = outputs['light_direction']  # (B, S, 3)
            if 'light_intensity' in outputs:
                light_params['light_intensity'] = outputs['light_intensity']  # (B, S, 1)
            if 'light_color' in outputs:
                light_params['light_color'] = outputs['light_color']  # (B, S, 3)

            # 获取深度和法线 (用于渲染)
            depth = outputs.get('depth')  # (B, S, H, W, 1)
            normals = self._depth_to_normals(depth)  # (B, S, H, W, 3)

            # 渲染
            if materials and light_params:
                rendered = self.renderer.phong_shading(
                    normals=normals,
                    diffuse=materials.get('diffuse'),
                    specular=materials.get('specular'),
                    roughness=materials.get('roughness'),
                    ao=materials.get('ao'),
                    light_params=light_params,
                )
                # 数值安全: clamp渲染结果防止NaN (虚拟数据可能产生异常值)
                rendered = torch.clamp(rendered, 0.0, 1.0)
                rendered = torch.nan_to_num(rendered, nan=0.5, posinf=1.0, neginf=0.0)
            else:
                # 如果没有材质或光照，使用简单渲染
                rendered = images.permute(0, 1, 3, 4, 2)  # (B, S, H, W, C)

            # 准备目标图像
            target = images.permute(0, 1, 3, 4, 2)  # (B, S, H, W, C)

            # 数值安全: 对材质也做NaN处理 (虚拟数据可能产生异常)
            safe_materials = {}
            for k, v in materials.items():
                if v is not None:
                    safe_materials[k] = torch.nan_to_num(v, nan=0.5, posinf=1.0, neginf=0.0)

            # 计算损失
            loss_dict = self.loss_fn(
                rendered_image=rendered,
                target_image=target,
                materials=safe_materials,
            )

        # 反向传播
        total_loss = loss_dict['loss_phong_total']
        total_loss.backward()

        # 梯度裁剪
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_grad_norm
            )

        # 优化器步骤
        self.optimizer.step()

        # 记录监控信息
        self._log_step(loss_dict, materials, light_params)

        return loss_dict

    def _depth_to_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """
        从深度图计算法线 (简化版)

        Args:
            depth: (B, S, H, W, 1) 或 (B, S, H, W)

        Returns:
            normals: (B, S, H, W, 3)
        """
        if depth is None:
            return None

        # 处理 (B, S, H, W, 1) 格式
        if depth.dim() == 5:
            depth = depth.squeeze(-1)  # (B, S, H, W)

        B, S, H, W = depth.shape

        # 使用Sobel算子计算梯度
        # 这是简化版本，实际应该使用相机内参进行精确计算
        depth_padded = torch.nn.functional.pad(depth, (1, 1, 1, 1), mode='replicate')

        # 计算梯度
        dz_dx = (depth_padded[:, :, 1:-1, 2:] - depth_padded[:, :, 1:-1, :-2]) / 2.0
        dz_dy = (depth_padded[:, :, 2:, 1:-1] - depth_padded[:, :, :-2, 1:-1]) / 2.0

        # 构建法线
        normals = torch.stack([
            -dz_dx,
            -dz_dy,
            torch.ones_like(dz_dx),
        ], dim=-1)

        # 归一化
        normals = torch.nn.functional.normalize(normals, dim=-1)

        return normals

    def _log_step(self, loss_dict: dict, materials: dict, light_params: dict):
        """记录单步监控信息"""
        # 记录损失
        self.monitor.log_losses(loss_dict, self.global_step)

        # 记录材质统计
        if materials:
            self.monitor.log_material_stats(materials, self.global_step)

        # 记录光照统计
        if light_params:
            self.monitor.log_light_stats(light_params, self.global_step)

        # 记录梯度统计
        self.monitor.log_gradient_stats(self.model, self.global_step)

        # 步骤结束
        self.monitor.step_end(self.global_step)

        self.global_step += 1

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        训练一个epoch

        Args:
            dataloader: 数据加载器

        Returns:
            avg_loss: 平均损失
        """
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            loss_dict = self.train_step(batch)
            total_loss += loss_dict['loss_phong_total'].item()
            num_batches += 1

            # 打印进度
            if batch_idx % self.config.get('print_freq', 10) == 0:
                print(f"Epoch {self.epoch} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss_dict['loss_phong_total'].item():.4f}")

            # 保存可视化
            if self.global_step % self.config.get('viz_interval', 500) == 0:
                self._save_visualization(batch)

        avg_loss = total_loss / max(num_batches, 1)

        # epoch结束
        self.monitor.epoch_end(self.epoch)

        if self.scheduler:
            self.scheduler.step()

        self.epoch += 1

        return avg_loss

    def _save_visualization(self, batch: dict):
        """保存训练可视化"""
        self.model.eval()

        with torch.no_grad():
            images = batch['images'].to(self.device)
            outputs = self.model(images=images)

            materials = outputs.get('materials', {})
            light_params = outputs.get('light_params', {})
            depth = outputs.get('depth')
            normals = self._depth_to_normals(depth)

            if materials and light_params and normals is not None:
                rendered = self.renderer.phong_shading(
                    normals=normals,
                    diffuse=materials.get('diffuse'),
                    specular=materials.get('specular'),
                    roughness=materials.get('roughness'),
                    ao=materials.get('ao'),
                    light_params=light_params,
                )
            else:
                rendered = images.permute(0, 1, 3, 4, 2)

            target = images.permute(0, 1, 3, 4, 2)

            self.viz_saver.save_training_visualization(
                step=self.global_step,
                input_image=images,
                depth=depth,
                normals=normals,
                materials=materials,
                light_params=light_params,
                rendered=rendered,
                target=target,
            )

        self.model.train()

    def save_checkpoint(self, filename: str = None):
        """保存检查点"""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"

        checkpoint_path = self.log_dir / "checkpoints"
        checkpoint_path.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path / filename)

        # 同时保存latest
        torch.save(checkpoint, checkpoint_path / "checkpoint_latest.pt")

        print(f"[PhongTrainer] Checkpoint saved: {checkpoint_path / filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        print(f"[PhongTrainer] Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"[PhongTrainer] Resumed from epoch {self.epoch}, step {self.global_step}")

    def close(self):
        """关闭训练器"""
        self.monitor.close()
        print("[PhongTrainer] Closed")


def create_dummy_dataloader(batch_size: int = 2, num_samples: int = 100):
    """
    创建虚拟数据加载器 (用于测试)

    实际训练时应替换为真实数据集
    """
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, num_samples, img_size=(224, 224)):
            self.num_samples = num_samples
            self.img_size = img_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 生成随机图像数据
            images = torch.rand(2, 3, *self.img_size)  # (S=2, C, H, W)
            return {'images': images}

    dataset = DummyDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Phong Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--log_dir', type=str, default='./phong_training_logs', help='Log directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    args = parser.parse_args()

    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {}

    # 命令行参数覆盖配置
    config['pretrained_checkpoint'] = args.checkpoint or config.get('pretrained_checkpoint')
    config['max_epochs'] = args.epochs
    config['log_dir'] = args.log_dir
    config['experiment_name'] = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    if 'optimizer' not in config:
        config['optimizer'] = {}
    config['optimizer']['lr'] = args.lr

    # 创建训练器
    trainer = PhongTrainer(config)

    # 加载恢复检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 创建数据加载器 (这里使用虚拟数据，实际应使用真实数据集)
    print("[Main] Creating dataloader (dummy data for testing)...")
    dataloader = create_dummy_dataloader(batch_size=args.batch_size)

    # 训练循环
    print(f"[Main] Starting training for {config['max_epochs']} epochs...")
    try:
        for epoch in range(trainer.epoch, config['max_epochs']):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config['max_epochs']}")
            print(f"{'='*60}")

            avg_loss = trainer.train_epoch(dataloader)

            print(f"\nEpoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

            # 保存检查点
            if (epoch + 1) % config.get('save_freq', 10) == 0:
                trainer.save_checkpoint()

            # 更新最佳模型
            if avg_loss < trainer.best_loss:
                trainer.best_loss = avg_loss
                trainer.save_checkpoint("checkpoint_best.pt")

    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user")
        trainer.save_checkpoint("checkpoint_interrupted.pt")

    finally:
        trainer.close()

    # 生成训练摘要
    print("\n" + create_training_summary(str(trainer.log_dir / trainer.config['experiment_name'])))

    print("\n[Main] Training completed!")


if __name__ == "__main__":
    main()
