# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SimplePhongRenderer: 可微分Phong光照渲染器

特点:
1. 支持可学习的光照方向（每张图像独立预测）
2. 从深度图计算法线
3. 实现 Ambient + Diffuse + Specular 着色
4. 完全可微分，支持端到端训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePhongRenderer(nn.Module):
    """
    可微分Phong渲染器，支持可学习光照

    特点:
    - 支持per-image光照方向（由LightHead预测）
    - 视线方向固定沿着Z轴正方向 [0, 0, 1]（相机朝向）
    - 光照方向可学习，从图像特征推断真实光源位置
    """

    def __init__(
        self,
        ambient_strength: float = 0.3,
        use_learnable_light: bool = True,
    ):
        """
        Args:
            ambient_strength: 环境光强度系数
            use_learnable_light: 是否使用可学习光照（True: 从light_params读取, False: 使用固定默认值）
        """
        super().__init__()

        self.ambient_strength = ambient_strength
        self.use_learnable_light = use_learnable_light

        # 视线方向（固定：相机朝向）
        self.register_buffer('view_dir', torch.tensor([0.0, 0.0, 1.0]))

        # 默认光照参数（当use_learnable_light=False或light_params未提供时使用）
        self.register_buffer('default_light_dir', torch.tensor([0.3, -0.5, 0.8]))  # 斜上方光源
        self.register_buffer('default_light_color', torch.tensor([1.0, 1.0, 1.0]))  # 白光
        self.default_light_intensity = 1.0

    def depth_to_normals(self, depth: torch.Tensor, intrinsics: torch.Tensor = None) -> torch.Tensor:
        """
        从深度图计算表面法线

        使用中心差分近似梯度:
        dz/dx ≈ (z[x+1] - z[x-1]) / 2
        dz/dy ≈ (z[y+1] - z[y-1]) / 2

        Args:
            depth: (B, S, H, W) 深度图
            intrinsics: (B, S, 3, 3) 相机内参 (当前版本暂时忽略，使用简化版本)

        Returns:
            normals: (B, S, H, W, 3) 单位法向量
        """
        B, S, H, W = depth.shape

        # 添加通道维度
        depth = depth.unsqueeze(-1)  # (B, S, H, W, 1)

        # 计算深度梯度 (中心差分)
        # dz/dx: 水平方向
        dz_dx = depth[..., :, 2:, :] - depth[..., :, :-2, :]  # (B, S, H, W-2, 1)
        # Padding回原尺寸
        dz_dx = F.pad(dz_dx, (0, 0, 1, 1, 0, 0), mode='replicate')

        # dz/dy: 垂直方向
        dz_dy = depth[..., 2:, :, :] - depth[..., :-2, :, :]  # (B, S, H-2, W, 1)
        # Padding回原尺寸
        dz_dy = F.pad(dz_dy, (0, 0, 0, 0, 1, 1), mode='replicate')

        # 构造法向量: N = normalize([-dz/dx, -dz/dy, 1])
        # 注意负号：深度增加时，表面向内倾斜
        normals = torch.cat([
            -dz_dx,
            -dz_dy,
            torch.ones_like(depth)
        ], dim=-1)  # (B, S, H, W, 3)

        # 归一化为单位向量
        normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)

        return normals

    def phong_shading(
        self,
        normals: torch.Tensor,
        diffuse: torch.Tensor,
        specular: torch.Tensor,
        roughness: torch.Tensor,
        ao: torch.Tensor = None,
        light_params: dict = None,
    ) -> torch.Tensor:
        """
        Phong着色模型

        I = ambient + diffuse * (N·L) + specular * (N·H)^shininess

        Args:
            normals: (B, S, H, W, 3) 表面法向量
            diffuse: (B, S, H, W, 3) 漫反射系数
            specular: (B, S, H, W, 3) 镜面反射系数
            roughness: (B, S, H, W, 1) 粗糙度
            ao: (B, S, H, W, 1) 环境光遮蔽 (可选)
            light_params: dict 光照参数 (可选), 包含:
                - 'light_direction': (B, S, 3) 光照方向
                - 'light_intensity': (B, S, 1) 光照强度
                - 'light_color': (B, S, 3) 光源颜色
                如果为None，使用默认参数

        Returns:
            shaded_color: (B, S, H, W, 3) 着色后的颜色
        """
        B, S, H, W, _ = normals.shape

        # 解析光照参数
        if light_params is not None and self.use_learnable_light:
            # 使用预测的光照参数
            light_dir_base = light_params['light_direction']  # (B, S, 3)
            light_intensity_base = light_params['light_intensity']  # (B, S, 1)
            light_color_base = light_params['light_color']  # (B, S, 3)

            # Expand to (B, S, H, W, 3/1)
            light_dir = light_dir_base.unsqueeze(2).unsqueeze(3).expand(B, S, H, W, 3)
            light_intensity = light_intensity_base.unsqueeze(2).unsqueeze(3).expand(B, S, H, W, 1)
            light_color = light_color_base.unsqueeze(2).unsqueeze(3).expand(B, S, H, W, 3)
        else:
            # 使用默认光照参数
            light_dir = self.default_light_dir.view(1, 1, 1, 1, 3).expand(B, S, H, W, 3)
            light_intensity = torch.full((B, S, H, W, 1), self.default_light_intensity,
                                        device=normals.device, dtype=normals.dtype)
            light_color = self.default_light_color.view(1, 1, 1, 1, 3).expand(B, S, H, W, 3)

        # 1. Ambient 分量 (环境光)
        if ao is not None:
            ambient = diffuse * self.ambient_strength * ao
        else:
            ambient = diffuse * self.ambient_strength

        # 2. Diffuse 分量 (Lambertian 漫反射)
        # N · L (法向量与光线方向的点积)
        n_dot_l = torch.sum(normals * light_dir, dim=-1, keepdim=True)  # (B, S, H, W, 1)
        n_dot_l = torch.clamp(n_dot_l, min=0.0)  # 只保留正值（面向光源的部分）

        diffuse_shading = diffuse * n_dot_l * light_intensity * light_color

        # 3. Specular 分量 (Blinn-Phong 高光)
        # 计算半角向量 H = normalize(L + V)
        view_dir = self.view_dir.view(1, 1, 1, 1, 3).expand(B, S, H, W, 3)
        half_vec = F.normalize(light_dir + view_dir, p=2, dim=-1, eps=1e-6)

        # N · H
        n_dot_h = torch.sum(normals * half_vec, dim=-1, keepdim=True)  # (B, S, H, W, 1)
        n_dot_h = torch.clamp(n_dot_h, min=0.0)

        # 粗糙度转换为Phong指数
        # roughness ∈ [0, 1], shininess ∈ [1, 100]
        # roughness越小(光滑) → shininess越大(高光越集中)
        shininess = (1.0 - roughness.clamp(0.01, 0.99)) * 99.0 + 1.0

        # 高光项
        specular_shading = specular * torch.pow(n_dot_h + 1e-6, shininess) * light_intensity * light_color

        # 只在被光照到的区域有高光 (n_dot_l > 0)
        specular_shading = specular_shading * (n_dot_l > 0).float()

        # 4. 合成最终颜色
        final_color = ambient + diffuse_shading + specular_shading

        # Clamp到有效范围
        final_color = torch.clamp(final_color, 0.0, 1.0)

        return final_color

    def forward(
        self,
        depth: torch.Tensor,
        materials: dict,
        light_params: dict = None,
        intrinsics: torch.Tensor = None,
    ) -> tuple:
        """
        完整的渲染流程

        Args:
            depth: (B, S, H, W) 深度图
            materials: 包含以下键的字典:
                - 'diffuse': (B, S, H, W, 3) 漫反射颜色
                - 'specular': (B, S, H, W, 3) 镜面反射颜色
                - 'roughness': (B, S, H, W, 1) 粗糙度
                - 'ambient_occlusion': (B, S, H, W, 1) 环境光遮蔽 (可选)
            light_params: 光照参数字典 (可选):
                - 'light_direction': (B, S, 3) 光照方向
                - 'light_intensity': (B, S, 1) 光照强度
                - 'light_color': (B, S, 3) 光源颜色
            intrinsics: (B, S, 3, 3) 相机内参 (当前版本暂不使用)

        Returns:
            rendered_image: (B, S, H, W, 3) 渲染后的图像
            normals: (B, S, H, W, 3) 计算得到的法向量 (用于可视化和损失)
        """
        # 1. 从深度图计算法线
        normals = self.depth_to_normals(depth, intrinsics)

        # 2. 提取材质属性
        diffuse = materials['diffuse']
        specular = materials['specular']
        roughness = materials['roughness']
        ao = materials.get('ambient_occlusion', None)

        # 3. 执行Phong着色（传入light_params）
        rendered_image = self.phong_shading(
            normals=normals,
            diffuse=diffuse,
            specular=specular,
            roughness=roughness,
            ao=ao,
            light_params=light_params,  # 传递光照参数
        )

        return rendered_image, normals


class SimplePhongRendererWithMask(SimplePhongRenderer):
    """
    带Mask的Phong渲染器

    在渲染时考虑有效像素mask，用于处理:
    - 无效深度区域
    - 遮挡
    - 边界
    """

    def forward(
        self,
        depth: torch.Tensor,
        materials: dict,
        mask: torch.Tensor = None,
        light_params: dict = None,
        intrinsics: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            mask: (B, S, H, W) 有效像素mask (1=有效, 0=无效)
            light_params: 光照参数字典（与父类相同）
            其他参数见父类

        Returns:
            rendered_image: (B, S, H, W, 3)
            normals: (B, S, H, W, 3)
            mask: (B, S, H, W, 1) 扩展到与图像相同的维度
        """
        # 调用父类的渲染（传递light_params）
        rendered_image, normals = super().forward(depth, materials, light_params, intrinsics)

        # 应用mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, S, H, W, 1)
            rendered_image = rendered_image * mask_expanded
            # 注意：normals不需要mask，因为它们用于损失计算时会单独处理
        else:
            mask_expanded = torch.ones_like(rendered_image[..., :1])

        return rendered_image, normals, mask_expanded
