#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
查看Phong训练状态

可以在本地运行来查看服务器上的训练状态
支持从远程同步的日志目录读取

Usage:
    python view_training_status.py --log_dir /path/to/logs/experiment_name
    python view_training_status.py --log_dir /path/to/logs/experiment_name --watch
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime


def load_latest_metrics(log_dir: Path) -> dict:
    """加载最新指标"""
    latest_file = log_dir / "metrics" / "latest.json"
    if not latest_file.exists():
        return None

    with open(latest_file) as f:
        return json.load(f)


def load_config(log_dir: Path) -> dict:
    """加载配置"""
    config_file = log_dir / "config.json"
    if not config_file.exists():
        return None

    with open(config_file) as f:
        return json.load(f)


def load_alerts(log_dir: Path, limit: int = 10) -> list:
    """加载警报"""
    alerts_file = log_dir / "alerts.log"
    if not alerts_file.exists():
        return []

    with open(alerts_file) as f:
        alerts = f.readlines()

    return alerts[-limit:]


def format_time_ago(timestamp_str: str) -> str:
    """格式化时间差"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        delta = datetime.now() - timestamp
        seconds = delta.total_seconds()

        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds/60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds/3600)}h ago"
        else:
            return f"{int(seconds/86400)}d ago"
    except:
        return "unknown"


def print_status(log_dir: Path, verbose: bool = False):
    """打印训练状态"""
    print("\n" + "=" * 70)
    print("Phong Training Status")
    print("=" * 70)

    # 配置信息
    config = load_config(log_dir)
    if config:
        print(f"\nExperiment: {config.get('experiment_name', 'unknown')}")
        print(f"Started: {config.get('start_time', 'unknown')}")
    else:
        print(f"\nLog directory: {log_dir}")

    # 最新指标
    metrics = load_latest_metrics(log_dir)
    if metrics:
        print(f"\n--- Latest Status ---")
        print(f"Step: {metrics.get('step', 0)}")
        print(f"Epoch: {metrics.get('epoch', 0)}")
        print(f"Updated: {format_time_ago(metrics.get('timestamp', ''))}")

        print(f"\n--- Key Metrics ---")

        # 损失
        losses = {k: v for k, v in metrics.get('metrics', {}).items() if 'loss' in k}
        if losses:
            print("\nLosses:")
            for name, value in sorted(losses.items()):
                short_name = name.replace('loss/', '').replace('loss_', '')
                print(f"  {short_name}: {value:.6f}")

        # 材质统计
        materials = {k: v for k, v in metrics.get('metrics', {}).items() if 'material' in k}
        if materials and verbose:
            print("\nMaterial Stats:")
            for name, value in sorted(materials.items()):
                short_name = name.replace('material/', '')
                print(f"  {short_name}: {value:.6f}")

        # 光照统计
        lights = {k: v for k, v in metrics.get('metrics', {}).items() if 'light' in k}
        if lights:
            print("\nLight Stats:")
            for name, value in sorted(lights.items()):
                short_name = name.replace('light/', '')
                print(f"  {short_name}: {value:.6f}")

        # 梯度统计
        gradients = {k: v for k, v in metrics.get('metrics', {}).items() if 'gradient' in k}
        if gradients and verbose:
            print("\nGradient Stats:")
            for name, value in sorted(gradients.items()):
                short_name = name.replace('gradient/', '')
                print(f"  {short_name}: {value:.6f}")

    else:
        print("\nNo metrics found yet.")

    # 警报
    alerts = load_alerts(log_dir, limit=5)
    if alerts:
        print(f"\n--- Recent Alerts ({len(alerts)}) ---")
        for alert in alerts:
            print(f"  {alert.strip()}")

    # 可视化文件
    viz_dir = log_dir / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png"))
        if viz_files:
            latest_viz = max(viz_files, key=lambda p: p.stat().st_mtime)
            print(f"\n--- Latest Visualization ---")
            print(f"  {latest_viz.name}")
            print(f"  Updated: {format_time_ago(datetime.fromtimestamp(latest_viz.stat().st_mtime).isoformat())}")

    # 检查点
    ckpt_dir = log_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.glob("*.pt"))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
            print(f"\n--- Latest Checkpoint ---")
            print(f"  {latest_ckpt.name}")

    print("\n" + "=" * 70)


def watch_status(log_dir: Path, interval: int = 30, verbose: bool = False):
    """持续监控训练状态"""
    print(f"Watching {log_dir} every {interval}s. Press Ctrl+C to stop.\n")

    try:
        while True:
            # 清屏
            os.system('cls' if os.name == 'nt' else 'clear')

            print_status(log_dir, verbose)

            print(f"\nRefreshing in {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")


def plot_loss_history(log_dir: Path):
    """绘制损失曲线 (如果matplotlib可用)"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        history_file = log_dir / "metrics" / "full_history.json"
        if not history_file.exists():
            print("No history file found.")
            return

        with open(history_file) as f:
            history = json.load(f)

        # 找到损失相关的指标
        loss_keys = [k for k in history.keys() if 'loss' in k and 'total' in k]

        if not loss_keys:
            print("No loss metrics found.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for key in loss_keys:
            data = history[key]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(steps, values, label=key.replace('loss/', ''))

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = log_dir / "loss_curve.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Loss curve saved to: {save_path}")

    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(description='View Phong Training Status')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to log directory')
    parser.add_argument('--watch', action='store_true', help='Continuously watch status')
    parser.add_argument('--interval', type=int, default=30, help='Watch interval in seconds')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    parser.add_argument('--plot', action='store_true', help='Plot loss curve')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Directory not found: {log_dir}")
        sys.exit(1)

    if args.plot:
        plot_loss_history(log_dir)
    elif args.watch:
        watch_status(log_dir, args.interval, args.verbose)
    else:
        print_status(log_dir, args.verbose)


if __name__ == "__main__":
    main()
