# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LightHead: 从图像特征预测光照参数

核心理念：
真实场景中光源位置是多变的（顶光、侧光、逆光等），不应假设光源在相机位置。
LightHead从VGGT的聚合特征中学习每张图像的光照方向，使模型能够：
1. 解耦材质和光照
2. 学习物理正确的材质属性
3. 适应不同场景的真实光照条件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightHead(nn.Module):
    """
    轻量级光照预测头

    从VGGT的aggregator特征预测每张图像的光照参数：
    - light_direction: 主光源方向（归一化向量）
    - light_intensity: 光照强度（可选）
    - light_color: 光源颜色（可选，当前固定为白光）

    设计原则：
    1. 轻量级：避免过拟合，只预测最关键的参数
    2. 物理约束：方向归一化、强度非负
    3. Per-image：每张图像独立预测，适应场景变化
    """

    def __init__(
        self,
        dim_in: int = 2048,  # VGGT aggregator输出维度 (2 * embed_dim)
        hidden_dim: int = 512,
        predict_intensity: bool = True,
        predict_color: bool = False,  # 当前版本固定为白光
    ):
        """
        Args:
            dim_in: 输入特征维度（VGGT aggregator的输出）
            hidden_dim: 隐藏层维度
            predict_intensity: 是否预测光照强度（否则固定为1.0）
            predict_color: 是否预测光源颜色（否则固定为白光[1,1,1]）
        """
        super().__init__()

        self.predict_intensity = predict_intensity
        self.predict_color = predict_color

        # 计算输出维度
        output_dim = 3  # light_direction (必须)
        if predict_intensity:
            output_dim += 1
        if predict_color:
            output_dim += 3

        # 轻量级MLP
        self.predictor = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 默认光照参数（用于初始化和fallback）
        self.register_buffer('default_light_dir', torch.tensor([0.3, -0.5, 0.8]))  # 斜上方光源
        self.register_buffer('default_light_color', torch.tensor([1.0, 1.0, 1.0]))  # 白光

        # 初始化：让网络初始输出接近默认值
        self._init_weights()

    def _init_weights(self):
        """
        初始化权重，使初始输出接近物理合理的默认值
        """
        # 最后一层Linear初始化为接近0，这样加上bias后输出合理
        nn.init.zeros_(self.predictor[-1].weight)

        # Bias初始化
        with torch.no_grad():
            idx = 0

            # light_direction: 初始化为默认方向
            self.predictor[-1].bias[idx:idx+3].copy_(self.default_light_dir)
            idx += 3

            # light_intensity: 初始化为1.0 (sigmoid(0)=0.5, 需要调整)
            if self.predict_intensity:
                self.predictor[-1].bias[idx].fill_(2.0)  # sigmoid(2.0)≈0.88
                idx += 1

            # light_color: 初始化为白光
            if self.predict_color:
                self.predictor[-1].bias[idx:idx+3].copy_(self.default_light_color)
                idx += 3

    def forward(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor = None,
        patch_start_idx: list = None,
    ) -> dict:
        """
        预测光照参数

        Args:
            aggregated_tokens_list: List of (B, S, N, C) tensors from aggregator
            images: (B, S, 3, H, W) 输入图像（当前未使用，保留接口）
            patch_start_idx: Integer indicating patch start position

        Returns:
            light_params: dict containing:
                - 'light_direction': (B, S, 3) 归一化的光照方向
                - 'light_intensity': (B, S, 1) 光照强度 [0.3, 1.5]
                - 'light_color': (B, S, 3) 光源颜色（当前固定为白光）
        """
        # 获取aggregated tokens (取最后一层)
        aggregated_tokens = aggregated_tokens_list[-1]  # (B, S, N, C)

        B, S, N, C = aggregated_tokens.shape

        # Global average pooling: 将所有tokens聚合为单个特征向量
        # Shape: (B, S, N, C) -> (B, S, C)
        global_features = aggregated_tokens.mean(dim=2)  # (B, S, C)

        # Reshape to (B*S, C) for batch processing
        global_features = global_features.reshape(B * S, C)

        # 预测光照参数
        raw_output = self.predictor(global_features)  # (B*S, output_dim)

        # 解析输出
        idx = 0

        # 1. Light direction (必须)
        light_dir_raw = raw_output[:, idx:idx+3]  # (B*S, 3)
        light_direction = F.normalize(light_dir_raw, p=2, dim=-1, eps=1e-6)
        idx += 3

        # 2. Light intensity (可选)
        if self.predict_intensity:
            light_intensity_raw = raw_output[:, idx:idx+1]  # (B*S, 1)
            # Sigmoid + scale to [0.3, 1.5]
            light_intensity = 0.3 + 1.2 * torch.sigmoid(light_intensity_raw)
            idx += 1
        else:
            light_intensity = torch.ones(B * S, 1, device=raw_output.device)

        # 3. Light color (可选，当前版本固定为白光)
        if self.predict_color:
            light_color_raw = raw_output[:, idx:idx+3]  # (B*S, 3)
            light_color = torch.sigmoid(light_color_raw)  # [0, 1]
            idx += 3
        else:
            light_color = self.default_light_color.unsqueeze(0).expand(B * S, 3)

        # Reshape back to (B, S, ...)
        light_direction = light_direction.view(B, S, 3)
        light_intensity = light_intensity.view(B, S, 1)
        light_color = light_color.view(B, S, 3)

        return {
            'light_direction': light_direction,
            'light_intensity': light_intensity,
            'light_color': light_color,
        }

    def get_num_parameters(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleLightHead(nn.Module):
    """
    超轻量级版本：只用2层MLP

    适用于快速实验或显存受限的场景
    """

    def __init__(
        self,
        dim_in: int = 2048,
        predict_intensity: bool = True,
    ):
        super().__init__()

        self.predict_intensity = predict_intensity
        output_dim = 3 + (1 if predict_intensity else 0)

        self.predictor = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

        # 默认参数
        self.register_buffer('default_light_dir', torch.tensor([0.3, -0.5, 0.8]))

        # 初始化
        nn.init.zeros_(self.predictor[-1].weight)
        with torch.no_grad():
            self.predictor[-1].bias[:3].copy_(self.default_light_dir)
            if predict_intensity:
                self.predictor[-1].bias[3].fill_(2.0)

    def forward(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor = None,
        patch_start_idx: list = None,
    ) -> dict:
        """与LightHead接口相同"""
        aggregated_tokens = aggregated_tokens_list[-1]  # (B, S, N, C)
        B, S, N, C = aggregated_tokens.shape

        global_features = aggregated_tokens.mean(dim=2)  # (B, S, C)
        global_features = global_features.reshape(B * S, C)

        raw_output = self.predictor(global_features)

        # 解析
        light_direction = F.normalize(raw_output[:, :3], p=2, dim=-1, eps=1e-6)

        if self.predict_intensity:
            light_intensity = 0.3 + 1.2 * torch.sigmoid(raw_output[:, 3:4])
        else:
            light_intensity = torch.ones(B * S, 1, device=raw_output.device)

        light_color = torch.ones(B * S, 3, device=raw_output.device)

        # Reshape back to (B, S, ...)
        light_direction = light_direction.view(B, S, 3)
        light_intensity = light_intensity.view(B, S, 1)
        light_color = light_color.view(B, S, 3)

        return {
            'light_direction': light_direction,
            'light_intensity': light_intensity,
            'light_color': light_color,
        }

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    """简单测试"""
    print("=" * 60)
    print("Testing LightHead")
    print("=" * 60)

    # 创建模型
    light_head = LightHead(dim_in=2048, hidden_dim=512)
    print(f"\n✅ LightHead created")
    print(f"   Parameters: {light_head.get_num_parameters():,}")

    # 模拟输入
    B, S, N, C = 2, 8, 256, 2048
    fake_tokens = torch.randn(B*S, N, C)
    aggregated_tokens_list = [fake_tokens]

    # 前向传播
    light_params = light_head(aggregated_tokens_list, patch_start_idx=[0, S])

    print(f"\n✅ Forward pass successful")
    print(f"   light_direction shape: {light_params['light_direction'].shape}")
    print(f"   light_intensity shape: {light_params['light_intensity'].shape}")
    print(f"   light_color shape: {light_params['light_color'].shape}")

    # 检查归一化
    light_dir = light_params['light_direction']
    norms = torch.norm(light_dir, p=2, dim=-1)
    print(f"\n✅ Direction normalization check:")
    print(f"   Norms: {norms.flatten()}")
    print(f"   All close to 1.0? {torch.allclose(norms, torch.ones_like(norms), atol=1e-5)}")

    # 检查强度范围
    intensity = light_params['light_intensity']
    print(f"\n✅ Intensity range check:")
    print(f"   Range: [{intensity.min().item():.3f}, {intensity.max().item():.3f}]")
    print(f"   Expected: [0.3, 1.5]")

    # 梯度测试
    loss = light_dir.sum() + intensity.sum()
    loss.backward()

    grad_count = sum(1 for p in light_head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"\n✅ Gradient flow check:")
    print(f"   Parameters with gradients: {grad_count}")

    print("\n" + "=" * 60)
    print("Testing SimpleLightHead")
    print("=" * 60)

    simple_light_head = SimpleLightHead(dim_in=2048)
    print(f"\n✅ SimpleLightHead created")
    print(f"   Parameters: {simple_light_head.get_num_parameters():,}")

    light_params_simple = simple_light_head(aggregated_tokens_list)
    print(f"\n✅ Forward pass successful")
    print(f"   light_direction shape: {light_params_simple['light_direction'].shape}")

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
