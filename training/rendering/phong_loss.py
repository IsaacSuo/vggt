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
    3. 法线一致性: 预测法线与深度导出法线的一致性
    """

    def __init__(
        self,
        weight: float = 0.1,
        material_smoothness_weight: float = 0.01,
        energy_conservation_weight: float = 0.01,
        normal_consistency_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(weight=weight, **kwargs)

        self.material_smoothness_weight = material_smoothness_weight
        self.energy_conservation_weight = energy_conservation_weight
        self.normal_consistency_weight = normal_consistency_weight

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

    def depth_to_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """
        从深度图计算法线

        Args:
            depth: (B, S, H, W) 或 (B, S, H, W, 1) 深度图

        Returns:
            normals: (B, S, H, W, 3) 单位法向量
        """
        # 处理 (B, S, H, W, 1) 格式
        if depth.dim() == 5:
            depth = depth.squeeze(-1)

        B, S, H, W = depth.shape

        # 使用Sobel算子计算梯度
        depth_padded = F.pad(depth, (1, 1, 1, 1), mode='replicate')

        # 计算梯度
        dz_dx = (depth_padded[:, :, 1:-1, 2:] - depth_padded[:, :, 1:-1, :-2]) / 2.0
        dz_dy = (depth_padded[:, :, 2:, 1:-1] - depth_padded[:, :, :-2, 1:-1]) / 2.0

        # 构建法线 (相机坐标系: z朝外)
        normals = torch.stack([
            -dz_dx,
            -dz_dy,
            torch.ones_like(dz_dx),
        ], dim=-1)

        # 归一化
        normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)

        return normals

    def compute_normal_consistency_loss(
        self,
        predicted_normals: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算法线一致性损失

        使用余弦距离: 1 - cos(predicted, depth_derived)

        Args:
            predicted_normals: (B, S, H, W, 3) 预测的法线
            depth: (B, S, H, W) 或 (B, S, H, W, 1) 深度图
            depth_conf: (B, S, H, W) 深度置信度 (可选，用于加权)

        Returns:
            consistency_loss: 标量
        """
        if depth is None or predicted_normals is None:
            return torch.tensor(0.0, device=predicted_normals.device if predicted_normals is not None else 'cpu')

        # 从深度计算法线
        depth_normals = self.depth_to_normals(depth)

        # 余弦相似度: dot(n1, n2) for unit vectors
        cosine_sim = (predicted_normals * depth_normals).sum(dim=-1)  # (B, S, H, W)

        # 余弦距离: 1 - cos
        cosine_dist = 1.0 - cosine_sim

        # 如果有深度置信度，使用它加权
        if depth_conf is not None:
            # 归一化置信度
            conf_weight = depth_conf / (depth_conf.mean() + 1e-6)
            cosine_dist = cosine_dist * conf_weight

        return cosine_dist.mean()

    def forward(
        self,
        rendered_image: torch.Tensor,
        target_image: torch.Tensor,
        materials: dict = None,
        mask: torch.Tensor = None,
        predicted_normals: torch.Tensor = None,
        depth: torch.Tensor = None,
        depth_conf: torch.Tensor = None,
    ) -> dict:
        """
        计算总损失 (包含正则化)

        Args:
            rendered_image: (B, S, H, W, 3)
            target_image: (B, S, H, W, 3)
            materials: 材质字典 (用于正则化)
            mask: (B, S, H, W, 1)
            predicted_normals: (B, S, H, W, 3) 预测的法线
            depth: (B, S, H, W) 或 (B, S, H, W, 1) 深度图
            depth_conf: (B, S, H, W) 深度置信度

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
            total_loss = total_loss + smoothness_loss * self.material_smoothness_weight

        # 能量守恒约束
        if materials is not None and self.energy_conservation_weight > 0:
            energy_loss = self.compute_energy_conservation_loss(materials)
            loss_dict['loss_phong_energy'] = energy_loss
            total_loss = total_loss + energy_loss * self.energy_conservation_weight

        # 法线一致性约束
        if predicted_normals is not None and depth is not None and self.normal_consistency_weight > 0:
            normal_loss = self.compute_normal_consistency_loss(predicted_normals, depth, depth_conf)
            loss_dict['loss_phong_normal_consistency'] = normal_loss
            total_loss = total_loss + normal_loss * self.normal_consistency_weight

        # 应用总权重
        loss_dict['loss_phong_total'] = total_loss * self.weight

        return loss_dict
