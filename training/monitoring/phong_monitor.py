# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Phong Training Monitor: 训练过程监控与日志记录

功能：
1. Loss曲线记录 (TensorBoard + JSON)
2. 材质预测统计 (分布、方差、异常检测)
3. 光照预测统计 (方向分布、是否坍缩)
4. 梯度监控 (检测梯度消失/爆炸)
5. 定期可视化保存
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

import torch
import torch.nn as nn
import numpy as np


class PhongTrainingMonitor:
    """
    Phong训练监控器

    记录训练过程中的所有关键指标，便于远程查看训练状态
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = None,
        use_tensorboard: bool = True,
        save_interval: int = 100,  # 每N步保存一次详细日志
        viz_interval: int = 500,   # 每N步保存一次可视化
    ):
        """
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
            use_tensorboard: 是否使用TensorBoard
            save_interval: 详细日志保存间隔
            viz_interval: 可视化保存间隔
        """
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.viz_interval = viz_interval

        # 创建子目录
        self.metrics_dir = self.log_dir / "metrics"
        self.viz_dir = self.log_dir / "visualizations"
        self.checkpoints_dir = self.log_dir / "checkpoints"

        for d in [self.metrics_dir, self.viz_dir, self.checkpoints_dir]:
            d.mkdir(exist_ok=True)

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

        # 内存中的指标历史
        self.metrics_history: Dict[str, List[float]] = {}
        self.step = 0
        self.epoch = 0

        # 异常检测阈值
        self.alert_thresholds = {
            'gradient_norm_max': 10.0,
            'gradient_norm_min': 1e-8,
            'light_direction_variance_min': 0.001,  # 光照方向坍缩检测
            'material_variance_min': 0.001,  # 材质坍缩检测
            'loss_spike_factor': 5.0,  # loss突增检测
        }

        # 保存配置
        self._save_config()

        print(f"[Monitor] Initialized at: {self.log_dir}")

    def _save_config(self):
        """保存监控配置"""
        config = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'save_interval': self.save_interval,
            'viz_interval': self.viz_interval,
            'alert_thresholds': self.alert_thresholds,
        }
        with open(self.log_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def log_scalar(self, name: str, value: float, step: int = None):
        """记录标量指标"""
        step = step if step is not None else self.step

        # 内存历史
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append({'step': step, 'value': value})

        # TensorBoard
        if self.writer:
            self.writer.add_scalar(name, value, step)

    def log_losses(self, loss_dict: Dict[str, torch.Tensor], step: int = None):
        """
        记录所有loss

        Args:
            loss_dict: {'loss_name': tensor_value}
        """
        step = step if step is not None else self.step

        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.log_scalar(f"loss/{name}", value, step)

        # 检测loss异常
        self._check_loss_anomaly(loss_dict, step)

    def log_material_stats(
        self,
        materials: Dict[str, torch.Tensor],
        step: int = None,
        prefix: str = "material"
    ):
        """
        记录材质预测统计

        Args:
            materials: {'diffuse': tensor, 'specular': tensor, ...}
        """
        step = step if step is not None else self.step

        stats = {}
        for name, tensor in materials.items():
            if tensor is None:
                continue

            with torch.no_grad():
                t = tensor.float()
                stats[f"{prefix}/{name}/mean"] = t.mean().item()
                stats[f"{prefix}/{name}/std"] = t.std().item()
                stats[f"{prefix}/{name}/min"] = t.min().item()
                stats[f"{prefix}/{name}/max"] = t.max().item()

                # 方差（检测模式坍缩）
                variance = t.var().item()
                stats[f"{prefix}/{name}/variance"] = variance

                # 记录到TensorBoard
                for stat_name, stat_value in stats.items():
                    if name in stat_name:
                        self.log_scalar(stat_name, stat_value, step)

        # 检测材质坍缩
        self._check_material_collapse(stats, step)

        return stats

    def log_light_stats(
        self,
        light_params: Dict[str, torch.Tensor],
        step: int = None
    ):
        """
        记录光照预测统计

        关键检测：光照方向是否坍缩为常数
        """
        step = step if step is not None else self.step

        stats = {}

        # 光照方向
        if 'light_direction' in light_params:
            light_dir = light_params['light_direction'].float()  # (B, S, 3)

            # 平均方向
            mean_dir = light_dir.mean(dim=[0, 1])
            stats['light/direction/mean_x'] = mean_dir[0].item()
            stats['light/direction/mean_y'] = mean_dir[1].item()
            stats['light/direction/mean_z'] = mean_dir[2].item()

            # 方向的方差（检测坍缩）
            dir_variance = light_dir.var(dim=[0, 1]).mean().item()
            stats['light/direction/variance'] = dir_variance

            # 与头灯[0,0,1]的余弦相似度
            headlight = torch.tensor([0., 0., 1.], device=light_dir.device)
            cosine_with_headlight = (light_dir * headlight).sum(dim=-1).mean().item()
            stats['light/direction/cosine_with_headlight'] = cosine_with_headlight

        # 光照强度
        if 'light_intensity' in light_params:
            intensity = light_params['light_intensity'].float()
            stats['light/intensity/mean'] = intensity.mean().item()
            stats['light/intensity/std'] = intensity.std().item()

        # 记录
        for stat_name, stat_value in stats.items():
            self.log_scalar(stat_name, stat_value, step)

        # 检测光照坍缩
        self._check_light_collapse(stats, step)

        return stats

    def log_gradient_stats(self, model: nn.Module, step: int = None):
        """
        记录梯度统计

        检测梯度消失和爆炸
        """
        step = step if step is not None else self.step

        stats = {}

        # 分模块记录
        modules_to_check = {
            'material_head': getattr(model, 'material_head', None),
            'light_head': getattr(model, 'light_head', None),
        }

        for module_name, module in modules_to_check.items():
            if module is None:
                continue

            grad_norms = []
            for name, param in module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)

            if grad_norms:
                stats[f"gradient/{module_name}/mean_norm"] = np.mean(grad_norms)
                stats[f"gradient/{module_name}/max_norm"] = np.max(grad_norms)
                stats[f"gradient/{module_name}/min_norm"] = np.min(grad_norms)
                stats[f"gradient/{module_name}/num_params"] = len(grad_norms)

        # 记录
        for stat_name, stat_value in stats.items():
            self.log_scalar(stat_name, stat_value, step)

        # 检测梯度异常
        self._check_gradient_anomaly(stats, step)

        return stats

    def _check_loss_anomaly(self, loss_dict: Dict, step: int):
        """检测loss异常"""
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            # 检测NaN/Inf
            if np.isnan(value) or np.isinf(value):
                self._log_alert(f"CRITICAL: {name} is NaN/Inf at step {step}", step)

            # 检测突增
            history_key = f"loss/{name}"
            if history_key in self.metrics_history and len(self.metrics_history[history_key]) > 10:
                recent_values = [h['value'] for h in self.metrics_history[history_key][-10:]]
                avg_recent = np.mean(recent_values)
                if value > avg_recent * self.alert_thresholds['loss_spike_factor']:
                    self._log_alert(f"WARNING: {name} spiked to {value:.4f} (avg was {avg_recent:.4f})", step)

    def _check_material_collapse(self, stats: Dict, step: int):
        """检测材质坍缩"""
        for key, value in stats.items():
            if 'variance' in key and value < self.alert_thresholds['material_variance_min']:
                material_name = key.split('/')[1]
                self._log_alert(f"WARNING: Material {material_name} may be collapsing (variance={value:.6f})", step)

    def _check_light_collapse(self, stats: Dict, step: int):
        """检测光照坍缩"""
        if 'light/direction/variance' in stats:
            variance = stats['light/direction/variance']
            if variance < self.alert_thresholds['light_direction_variance_min']:
                self._log_alert(f"WARNING: Light direction may be collapsing (variance={variance:.6f})", step)

        if 'light/direction/cosine_with_headlight' in stats:
            cosine = stats['light/direction/cosine_with_headlight']
            if cosine > 0.95:
                self._log_alert(f"WARNING: Light direction close to headlight [0,0,1] (cosine={cosine:.4f})", step)

    def _check_gradient_anomaly(self, stats: Dict, step: int):
        """检测梯度异常"""
        for key, value in stats.items():
            if 'max_norm' in key:
                if value > self.alert_thresholds['gradient_norm_max']:
                    self._log_alert(f"WARNING: Gradient explosion in {key}: {value:.4f}", step)
            if 'mean_norm' in key:
                if value < self.alert_thresholds['gradient_norm_min']:
                    self._log_alert(f"WARNING: Gradient vanishing in {key}: {value:.8f}", step)

    def _log_alert(self, message: str, step: int):
        """记录警报"""
        alert_file = self.log_dir / "alerts.log"
        timestamp = datetime.now().isoformat()
        with open(alert_file, 'a') as f:
            f.write(f"[{timestamp}] Step {step}: {message}\n")
        print(f"[ALERT] {message}")

    def save_metrics_snapshot(self, step: int = None):
        """保存指标快照到JSON"""
        step = step if step is not None else self.step

        snapshot = {
            'step': step,
            'epoch': self.epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }

        # 获取每个指标的最新值
        for name, history in self.metrics_history.items():
            if history:
                snapshot['metrics'][name] = history[-1]['value']

        # 保存
        snapshot_file = self.metrics_dir / f"snapshot_step_{step:08d}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        # 同时更新latest.json
        with open(self.metrics_dir / "latest.json", 'w') as f:
            json.dump(snapshot, f, indent=2)

    def save_full_history(self):
        """保存完整指标历史"""
        history_file = self.metrics_dir / "full_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f)

    def step_end(self, step: int = None):
        """每步结束时调用"""
        step = step if step is not None else self.step
        self.step = step + 1

        # 定期保存
        if step % self.save_interval == 0:
            self.save_metrics_snapshot(step)

    def epoch_end(self, epoch: int):
        """每个epoch结束时调用"""
        self.epoch = epoch
        self.save_metrics_snapshot()
        self.save_full_history()

        # 打印epoch摘要
        self._print_epoch_summary(epoch)

    def _print_epoch_summary(self, epoch: int):
        """打印epoch摘要"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*60}")

        # 打印关键指标
        key_metrics = [
            'loss/loss_phong_total',
            'loss/loss_phong_photometric',
            'material/diffuse/variance',
            'light/direction/variance',
            'light/direction/cosine_with_headlight',
            'gradient/material_head/mean_norm',
            'gradient/light_head/mean_norm',
        ]

        for metric in key_metrics:
            if metric in self.metrics_history and self.metrics_history[metric]:
                value = self.metrics_history[metric][-1]['value']
                print(f"  {metric}: {value:.6f}")

        print(f"{'='*60}\n")

    def close(self):
        """关闭监控器"""
        if self.writer:
            self.writer.close()
        self.save_full_history()
        print(f"[Monitor] Closed. Logs saved to: {self.log_dir}")


class VisualizationSaver:
    """
    可视化保存器

    定期保存训练过程中的可视化图片
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_training_visualization(
        self,
        step: int,
        input_image: torch.Tensor,
        depth: torch.Tensor,
        normals: torch.Tensor,
        materials: Dict[str, torch.Tensor],
        light_params: Dict[str, torch.Tensor],
        rendered: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        保存训练可视化

        Args:
            所有tensor预期形状: (B, S, ...), 取第一个sample保存
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            # 取第一个sample
            def to_numpy(t, idx=0):
                if t is None:
                    return None
                if len(t.shape) > 4:
                    t = t[0, 0]  # (B, S, ...) -> first sample, first frame
                elif len(t.shape) > 3:
                    t = t[0]
                return t.detach().cpu().numpy()

            # Row 1: Input, Depth, Normals, Target
            ax = axes[0, 0]
            img = to_numpy(input_image)
            if img is not None:
                if img.shape[0] == 3:  # (C, H, W)
                    img = img.transpose(1, 2, 0)
                ax.imshow(np.clip(img, 0, 1))
            ax.set_title('Input')
            ax.axis('off')

            ax = axes[0, 1]
            d = to_numpy(depth)
            if d is not None:
                if len(d.shape) == 3:
                    d = d[..., 0]
                ax.imshow(d, cmap='viridis')
            ax.set_title('Depth')
            ax.axis('off')

            ax = axes[0, 2]
            n = to_numpy(normals)
            if n is not None:
                n = (n + 1) / 2  # [-1,1] -> [0,1]
                ax.imshow(np.clip(n, 0, 1))
            ax.set_title('Normals')
            ax.axis('off')

            ax = axes[0, 3]
            t = to_numpy(target)
            if t is not None:
                if t.shape[0] == 3:
                    t = t.transpose(1, 2, 0)
                ax.imshow(np.clip(t, 0, 1))
            ax.set_title('Target')
            ax.axis('off')

            # Row 2: Diffuse, Specular, Rendered, Diff
            ax = axes[1, 0]
            diff = to_numpy(materials.get('diffuse'))
            if diff is not None:
                if diff.shape[0] == 3:
                    diff = diff.transpose(1, 2, 0)
                ax.imshow(np.clip(diff, 0, 1))
            ax.set_title('Diffuse')
            ax.axis('off')

            ax = axes[1, 1]
            spec = to_numpy(materials.get('specular'))
            if spec is not None:
                if spec.shape[0] == 3:
                    spec = spec.transpose(1, 2, 0)
                ax.imshow(np.clip(spec, 0, 1))
            ax.set_title('Specular')
            ax.axis('off')

            ax = axes[1, 2]
            r = to_numpy(rendered)
            if r is not None:
                if len(r.shape) == 3 and r.shape[-1] == 3:
                    pass
                elif r.shape[0] == 3:
                    r = r.transpose(1, 2, 0)
                ax.imshow(np.clip(r, 0, 1))
            ax.set_title('Rendered')
            ax.axis('off')

            ax = axes[1, 3]
            if r is not None and t is not None:
                if t.shape[0] == 3:
                    t = t.transpose(1, 2, 0)
                diff_img = np.abs(r - t)
                ax.imshow(diff_img, cmap='hot')
            ax.set_title('|Rendered - Target|')
            ax.axis('off')

            # 添加光照信息
            if 'light_direction' in light_params:
                light_dir = light_params['light_direction'][0, 0].detach().cpu().numpy()
                fig.suptitle(f'Step {step} | Light Dir: [{light_dir[0]:.2f}, {light_dir[1]:.2f}, {light_dir[2]:.2f}]',
                           fontsize=12)

            plt.tight_layout()
            save_path = self.save_dir / f"viz_step_{step:08d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            return str(save_path)

        except Exception as e:
            print(f"Warning: Failed to save visualization: {e}")
            return None


def create_training_summary(log_dir: str) -> str:
    """
    生成训练摘要报告

    可以在本地运行来查看服务器上的训练状态
    """
    log_dir = Path(log_dir)

    summary = []
    summary.append("=" * 60)
    summary.append("Training Summary Report")
    summary.append("=" * 60)

    # 读取配置
    config_file = log_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        summary.append(f"\nExperiment: {config.get('experiment_name', 'unknown')}")
        summary.append(f"Started: {config.get('start_time', 'unknown')}")

    # 读取最新指标
    latest_file = log_dir / "metrics" / "latest.json"
    if latest_file.exists():
        with open(latest_file) as f:
            latest = json.load(f)
        summary.append(f"\nLatest Step: {latest.get('step', 0)}")
        summary.append(f"Epoch: {latest.get('epoch', 0)}")
        summary.append(f"Timestamp: {latest.get('timestamp', 'unknown')}")

        summary.append("\nKey Metrics:")
        for name, value in latest.get('metrics', {}).items():
            summary.append(f"  {name}: {value:.6f}")

    # 读取警报
    alerts_file = log_dir / "alerts.log"
    if alerts_file.exists():
        with open(alerts_file) as f:
            alerts = f.readlines()
        if alerts:
            summary.append(f"\nRecent Alerts ({len(alerts)} total):")
            for alert in alerts[-5:]:  # 最近5条
                summary.append(f"  {alert.strip()}")

    summary.append("\n" + "=" * 60)

    return "\n".join(summary)


if __name__ == "__main__":
    # 测试监控器
    print("Testing PhongTrainingMonitor...")

    monitor = PhongTrainingMonitor(
        log_dir="./test_logs",
        experiment_name="test_run",
        use_tensorboard=False
    )

    # 模拟训练循环
    for step in range(10):
        # 模拟loss
        monitor.log_losses({
            'loss_phong_total': 0.5 - step * 0.01,
            'loss_phong_photometric': 0.4 - step * 0.008,
        }, step)

        # 模拟材质统计
        fake_materials = {
            'diffuse': torch.rand(1, 1, 3, 64, 64),
            'specular': torch.rand(1, 1, 3, 64, 64),
        }
        monitor.log_material_stats(fake_materials, step)

        # 模拟光照统计
        fake_light = {
            'light_direction': torch.randn(1, 1, 3),
            'light_intensity': torch.rand(1, 1, 1) + 0.5,
        }
        # 归一化
        fake_light['light_direction'] = torch.nn.functional.normalize(
            fake_light['light_direction'], dim=-1
        )
        monitor.log_light_stats(fake_light, step)

        monitor.step_end(step)

    monitor.epoch_end(0)
    monitor.close()

    # 生成摘要
    print("\n" + create_training_summary("./test_logs/test_run"))

    # 清理测试目录
    import shutil
    shutil.rmtree("./test_logs", ignore_errors=True)

    print("\nMonitor test passed!")
