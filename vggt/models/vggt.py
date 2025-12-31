# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, List

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        self._lora_enabled = False

    def enable_lora(
        self,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        target_block_type: str = "global",
        block_indices: Optional[List[int]] = None,
        freeze_base: bool = True,
    ) -> "VGGT":
        """
        Enable LoRA fine-tuning for the model.

        Args:
            rank: Rank of the LoRA matrices
            alpha: Scaling factor for LoRA
            dropout: Dropout probability for LoRA path
            target_modules: Which modules to apply LoRA to (default: ["qkv"])
            target_block_type: "global", "frame", or "both"
            block_indices: Specific block indices to target (None = all)
            freeze_base: Whether to freeze base model parameters

        Returns:
            self for chaining
        """
        from vggt.lora import LoRAConfig, inject_lora_to_model, freeze_base_model

        if target_modules is None:
            target_modules = ["qkv"]

        config = LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )

        inject_lora_to_model(
            self,
            config=config,
            target_block_type=target_block_type,
            block_indices=block_indices,
        )

        if freeze_base:
            freeze_base_model(self, unfreeze_lora=True)

        self._lora_enabled = True
        return self

    def merge_lora(self) -> "VGGT":
        """Merge LoRA weights into base model for efficient inference."""
        if not self._lora_enabled:
            return self

        from vggt.lora import merge_lora_weights
        merge_lora_weights(self)
        return self

    def save_lora(self, path: str) -> None:
        """Save only LoRA weights to a file."""
        from vggt.lora import save_lora_weights
        save_lora_weights(self, path)

    def load_lora(self, path: str, strict: bool = True) -> "VGGT":
        """Load LoRA weights from a file."""
        from vggt.lora import load_lora_weights
        load_lora_weights(self, path, strict=strict)
        return self

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        visual_hull_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            visual_hull_mask (torch.Tensor, optional): Binary mask with shape [S, H, W] or [B, S, H, W].
                1 = foreground (object), 0 = background.
                When provided, attention from/to background patches will be suppressed
                in global attention blocks. This is used for Visual-Hull-Aware inference.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Handle visual_hull_mask dimensions
        if visual_hull_mask is not None and len(visual_hull_mask.shape) == 3:
            visual_hull_mask = visual_hull_mask.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images, visual_hull_mask=visual_hull_mask
        )

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

