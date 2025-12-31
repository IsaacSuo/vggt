# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LoRA (Low-Rank Adaptation) implementation for VGGT.

This module provides LoRA layers that can be injected into existing linear layers
to enable efficient fine-tuning with minimal parameter overhead.

Reference: https://arxiv.org/abs/2106.09685
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA injection."""
    rank: int = 32
    alpha: float = 32.0  # Scaling factor, often set equal to rank
    dropout: float = 0.0
    target_modules: List[str] = None  # e.g., ["qkv", "proj"]

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["qkv"]  # Default: only QKV projection


class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear layer.

    This wraps an existing nn.Linear layer and adds low-rank adaptation matrices.
    The output is: y = Wx + (BA)x * (alpha / rank)

    Where:
        - W is the frozen original weight
        - B is the low-rank down-projection (in_features -> rank)
        - A is the low-rank up-projection (rank -> out_features)
        - alpha/rank is the scaling factor

    Args:
        original_layer: The original nn.Linear layer to wrap
        rank: The rank of the low-rank matrices
        alpha: Scaling factor for the LoRA output
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        # A: down-projection, initialized with Kaiming uniform
        # B: up-projection, initialized with zeros (so LoRA starts as identity)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Track if merged
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining original layer with LoRA adaptation.
        """
        # Original path (frozen)
        result = self.original_layer(x)

        if not self.merged:
            # LoRA path
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
            result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """
        Merge LoRA weights into the original layer for inference efficiency.
        After merging, the layer behaves as a regular linear layer.
        """
        if self.merged:
            return

        with torch.no_grad():
            # W' = W + B @ A * scaling
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.original_layer.weight.data += delta_weight

        self.merged = True

    def unmerge_weights(self):
        """
        Unmerge LoRA weights from the original layer.
        Useful for continuing training after merge.
        """
        if not self.merged:
            return

        with torch.no_grad():
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.original_layer.weight.data -= delta_weight

        self.merged = False

    @property
    def weight(self) -> torch.Tensor:
        """Return the effective weight (original + LoRA if not merged)."""
        if self.merged:
            return self.original_layer.weight
        else:
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            return self.original_layer.weight + delta

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Return the bias from the original layer."""
        return self.original_layer.bias


class LoRAAttention(nn.Module):
    """
    Wrapper for Attention layer with LoRA on QKV projection.

    This class wraps an existing Attention module and replaces its qkv projection
    with a LoRA-enhanced version.
    """

    def __init__(
        self,
        attention_module: nn.Module,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        target_qkv: bool = True,
        target_proj: bool = False,
    ):
        super().__init__()

        self.attention = attention_module

        # Replace qkv with LoRA version
        if target_qkv and hasattr(attention_module, 'qkv'):
            original_qkv = attention_module.qkv
            lora_qkv = LoRALinear(original_qkv, rank=rank, alpha=alpha, dropout=dropout)
            attention_module.qkv = lora_qkv

        # Replace output projection with LoRA version
        if target_proj and hasattr(attention_module, 'proj'):
            original_proj = attention_module.proj
            lora_proj = LoRALinear(original_proj, rank=rank, alpha=alpha, dropout=dropout)
            attention_module.proj = lora_proj

    def forward(self, *args, **kwargs):
        return self.attention(*args, **kwargs)
