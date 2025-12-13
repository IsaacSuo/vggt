# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Phong Loss: 基于物理渲染的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhongLoss(nn.Module):
    """
    基于物理渲染的损失函数

    计算渲染图像与真实图像之间的误差
    支持多种损失类型的组合
    """

    def __init__(
        self,
        weight: float = 0.1,
        photometric_loss_type: str = "l1",  # "l1", "l2", "smooth_l1"
        use_perceptual_loss: bool = False,
        perceptual_weight: float = 0.1,
    ):
        """
        Args:
            weight: 总体损失权重
            photometric_loss_type: 光度损失类型
            use_perceptual_loss: 是否使用感知损失 (需要VGG)
            perceptual_weight: 感知损失权重
        """
        super().__init__()

        self.weight = weight
        self.photometric_loss_type = photometric_loss_type
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight

        # 感知损失需要预训练的VGG网络 (暂时不实现，避免额外依赖)
        self.perceptual_loss_fn = None
        if use_perceptual_loss:
            print("Warning: Perceptual loss not implemented yet. Will be added in future.")

    def compute_photometric_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算光度损失 (逐像素颜色差异)

        Args:
            rendered: (B, S, H, W, 3) 渲染图像
            target: (B, S, H, W, 3) 目标图像
            mask: (B, S, H, W, 1) 有效像素mask

        Returns:
            loss: 标量损失
        """
        if self.photometric_loss_type == "l1":
            loss = torch.abs(rendered - target)
        elif self.photometric_loss_type == "l2":
            loss = (rendered - target) ** 2
        elif self.photometric_loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(rendered, target, reduction='none')
        else:
            raise ValueError(f"Unknown photometric loss type: {self.photometric_loss_type}")

        # 应用mask
        if mask is not None:
            loss = loss * mask
            # 计算平均损失 (只在有效像素上)
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss

    def forward(
        self,
        rendered_image: torch.Tensor,
        target_image: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算总损失

        Args:
            rendered_image: (B, S, H, W, 3) 渲染的图像
            target_image: (B, S, H, W, 3) 目标图像
            mask: (B, S, H, W, 1) 可选的mask

        Returns:
            total_loss: 标量损失
        """
        # 1. 光度损失
        photometric_loss = self.compute_photometric_loss(rendered_image, target_image, mask)

        total_loss = photometric_loss

        # 2. 感知损失 (如果启用)
        if self.use_perceptual_loss and self.perceptual_loss_fn is not None:
            # TODO: 实现感知损失
            pass

        # 应用总权重
        total_loss = total_loss * self.weight

        return total_loss


class PhongLossWithRegularization(PhongLoss):
    """
    带正则化的Phong损失

    额外的正则化项:
    1. 材质平滑度: 相邻像素的材质应该相似
    2. 物理约束: diffuse + specular ≤ 1 (能量守恒)
    """

    def __init__(
        self,
        weight: float = 0.1,
        material_smoothness_weight: float = 0.01,
        energy_conservation_weight: float = 0.01,
        **kwargs
    ):
        super().__init__(weight=weight, **kwargs)

        self.material_smoothness_weight = material_smoothness_weight
        self.energy_conservation_weight = energy_conservation_weight

    def compute_material_smoothness_loss(self, materials: dict) -> torch.Tensor:
        """
        材质平滑度损失: 鼓励相邻像素的材质相似

        Args:
            materials: 包含diffuse, specular, roughness等, shape (B, S, H, W, C)

        Returns:
            smoothness_loss: 标量
        """
        total_loss = 0.0
        count = 0

        for key in ['diffuse', 'specular', 'roughness']:
            if key in materials and materials[key] is not None:
                mat = materials[key]  # (B, S, H, W, C)

                # 水平方向梯度 (沿W维度)
                grad_w = torch.abs(mat[:, :, :, :-1, :] - mat[:, :, :, 1:, :])
                total_loss += grad_w.mean()

                # 垂直方向梯度 (沿H维度)
                grad_h = torch.abs(mat[:, :, :-1, :, :] - mat[:, :, 1:, :, :])
                total_loss += grad_h.mean()

                count += 2

        return total_loss / max(count, 1)

    def compute_energy_conservation_loss(self, materials: dict) -> torch.Tensor:
        """
        能量守恒约束: diffuse + specular <= 1

        Args:
            materials: 包含diffuse, specular

        Returns:
            energy_loss: 标量
        """
        diffuse = materials.get('diffuse', None)
        specular = materials.get('specular', None)

        if diffuse is None or specular is None:
            return torch.tensor(0.0, device=diffuse.device if diffuse is not None else 'cpu')

        # 计算总反照率
        total_albedo = diffuse.mean(dim=-1) + specular.mean(dim=-1)  # 对RGB通道求平均

        # 惩罚超过1的部分
        violation = F.relu(total_albedo - 1.0)

        return violation.mean()

    def forward(
        self,
        rendered_image: torch.Tensor,
        target_image: torch.Tensor,
        materials: dict = None,
        mask: torch.Tensor = None
    ) -> dict:
        """
        计算总损失 (包含正则化)

        Args:
            rendered_image: (B, S, H, W, 3)
            target_image: (B, S, H, W, 3)
            materials: 材质字典 (用于正则化)
            mask: (B, S, H, W, 1)

        Returns:
            loss_dict: 包含各项损失的字典
        """
        # 基础光度损失
        photometric_loss = self.compute_photometric_loss(rendered_image, target_image, mask)

        loss_dict = {
            'loss_phong_photometric': photometric_loss,
        }

        total_loss = photometric_loss

        # 材质平滑度正则化
        if materials is not None and self.material_smoothness_weight > 0:
            smoothness_loss = self.compute_material_smoothness_loss(materials)
            loss_dict['loss_phong_smoothness'] = smoothness_loss
            total_loss += smoothness_loss * self.material_smoothness_weight

        # 能量守恒约束
        if materials is not None and self.energy_conservation_weight > 0:
            energy_loss = self.compute_energy_conservation_loss(materials)
            loss_dict['loss_phong_energy'] = energy_loss
            total_loss += energy_loss * self.energy_conservation_weight

        # 应用总权重
        loss_dict['loss_phong_total'] = total_loss * self.weight

        return loss_dict
