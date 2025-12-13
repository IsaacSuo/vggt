# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NormalHead: 预测表面法线
复用DPT Head的架构，输出3通道单位法向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .dpt_head import _make_scratch, _make_fusion_block, custom_interpolate
from .utils import create_uv_grid, position_grid_to_embed


class NormalHead(nn.Module):
    """
    Surface normal prediction head.

    Outputs:
        - normals: [B, S, H, W, 3] - 单位法向量 (相机坐标系)

    Args:
        dim_in: Input dimension from aggregator tokens
        patch_size: Patch size (default: 14)
        features: Feature channels for DPT fusion (default: 256)
        out_channels: Output channels for each DPT layer
        intermediate_layer_idx: Which transformer layers to use
        pos_embed: Whether to use positional embedding
        down_ratio: Output downsampling ratio
    """

    def __init__(
        self,
        dim_in: int = 768,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        down_ratio: int = 1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        # Layer normalization
        self.norm = nn.LayerNorm(dim_in)

        # Projection layers (same as DPT)
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        # Resize layers for multi-scale features
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Feature fusion modules (same as DPT)
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # Output head for normal prediction
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)

        # Final prediction layers - output 3 channels for normal xyz
        head_features = features // 2
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            # 不使用Sigmoid，因为法线分量可以是负数
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        """
        Forward pass to predict surface normals.

        Args:
            aggregated_tokens_list: List of aggregated tokens from transformer layers
            images: Input images [B, S, 3, H, W]
            patch_start_idx: Starting index for patch tokens
            frames_chunk_size: Number of frames to process at once

        Returns:
            normals: [B, S, H, W, 3] - 单位法向量
        """
        B, S, _, H, W = images.shape

        # Process in chunks if needed for memory efficiency
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Chunked processing
        all_normals = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            chunk_normals = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx,
                frames_start_idx, frames_end_idx
            )
            all_normals.append(chunk_normals)

        return torch.cat(all_normals, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        """
        Implementation of forward pass.
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Extract and process features from multiple layers
        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            # Extract patch tokens
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frame chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            # Reshape for processing
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            # Project and resize
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse multi-scale features
        out = self._scratch_forward(out)

        # Interpolate to target resolution
        target_size = (
            int(patch_h * self.patch_size / self.down_ratio),
            int(patch_w * self.patch_size / self.down_ratio)
        )
        out = custom_interpolate(out, target_size, mode="bilinear", align_corners=True)

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        # Final normal prediction
        out = self.scratch.output_conv2(out)  # (B*S, 3, H, W)

        # Reshape to (B, S, 3, H, W)
        out = out.view(B, S, 3, H, W)

        # Convert to (B, S, H, W, 3) and normalize to unit vectors
        out = out.permute(0, 1, 3, 4, 2)  # (B, S, H, W, 3)
        normals = F.normalize(out, p=2, dim=-1, eps=1e-6)

        return normals

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Apply positional embedding to features."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def _scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward through fusion blocks."""
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out
