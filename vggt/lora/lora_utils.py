# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for LoRA integration with VGGT.
"""

import logging
from typing import List, Dict, Optional, Iterator, Tuple
from pathlib import Path

import torch
import torch.nn as nn

from .lora_layers import LoRALinear, LoRAConfig

logger = logging.getLogger(__name__)


def inject_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    target_block_type: str = "global",  # "global", "frame", or "both"
    block_indices: Optional[List[int]] = None,  # None means all blocks
) -> nn.Module:
    """
    Inject LoRA layers into VGGT model's attention blocks.

    Args:
        model: The VGGT model (or Aggregator)
        config: LoRA configuration
        target_block_type: Which blocks to target ("global", "frame", or "both")
        block_indices: Specific block indices to target (None = all)

    Returns:
        The modified model with LoRA layers injected
    """
    aggregator = model.aggregator if hasattr(model, 'aggregator') else model

    blocks_to_modify = []

    if target_block_type in ["global", "both"]:
        for idx, block in enumerate(aggregator.global_blocks):
            if block_indices is None or idx in block_indices:
                blocks_to_modify.append(("global", idx, block))

    if target_block_type in ["frame", "both"]:
        for idx, block in enumerate(aggregator.frame_blocks):
            if block_indices is None or idx in block_indices:
                blocks_to_modify.append(("frame", idx, block))

    lora_count = 0

    for block_type, idx, block in blocks_to_modify:
        for module_name in config.target_modules:
            if module_name == "qkv" and hasattr(block.attn, 'qkv'):
                original_layer = block.attn.qkv
                if not isinstance(original_layer, LoRALinear):
                    lora_layer = LoRALinear(
                        original_layer,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout,
                    )
                    block.attn.qkv = lora_layer
                    lora_count += 1
                    logger.debug(f"Injected LoRA into {block_type}_blocks[{idx}].attn.qkv")

            elif module_name == "proj" and hasattr(block.attn, 'proj'):
                original_layer = block.attn.proj
                if not isinstance(original_layer, LoRALinear):
                    lora_layer = LoRALinear(
                        original_layer,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout,
                    )
                    block.attn.proj = lora_layer
                    lora_count += 1
                    logger.debug(f"Injected LoRA into {block_type}_blocks[{idx}].attn.proj")

    logger.info(f"Injected {lora_count} LoRA layers into model")
    return model


def get_lora_parameters(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Get all LoRA parameters from a model.

    Args:
        model: The model with LoRA layers

    Yields:
        Tuples of (parameter_name, parameter)
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for param_name, param in module.named_parameters():
                if 'lora_' in param_name:
                    yield f"{name}.{param_name}", param


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.

    Args:
        model: The model with LoRA layers

    Returns:
        Dictionary of LoRA parameter names to tensors
    """
    lora_state = {}
    for name, param in get_lora_parameters(model):
        lora_state[name] = param.data.clone()
    return lora_state


def freeze_base_model(model: nn.Module, unfreeze_lora: bool = True) -> nn.Module:
    """
    Freeze all parameters in the model except LoRA parameters.

    Args:
        model: The model to freeze
        unfreeze_lora: Whether to keep LoRA parameters trainable

    Returns:
        The modified model
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_lora:
        # Unfreeze LoRA parameters
        for name, param in get_lora_parameters(model):
            param.requires_grad = True
            logger.debug(f"Unfroze LoRA parameter: {name}")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into the base model for efficient inference.

    Args:
        model: The model with LoRA layers

    Returns:
        The model with merged weights
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            logger.debug(f"Merged LoRA weights in {name}")

    return model


def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Unmerge all LoRA weights from the base model.

    Args:
        model: The model with merged LoRA layers

    Returns:
        The model with unmerged weights
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()
            logger.debug(f"Unmerged LoRA weights in {name}")

    return model


def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    Save only LoRA weights to a file.

    Args:
        model: The model with LoRA layers
        path: Path to save the weights
    """
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, path)
    logger.info(f"Saved {len(lora_state)} LoRA tensors to {path}")


def load_lora_weights(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    """
    Load LoRA weights from a file.

    Args:
        model: The model with LoRA layers
        path: Path to load the weights from
        strict: Whether to require all keys to match

    Returns:
        The model with loaded LoRA weights
    """
    lora_state = torch.load(path, map_location='cpu')

    # Get current model's LoRA parameter names
    model_lora_params = dict(get_lora_parameters(model))

    missing = []
    unexpected = []

    for name, tensor in lora_state.items():
        if name in model_lora_params:
            model_lora_params[name].data.copy_(tensor)
        else:
            unexpected.append(name)

    for name in model_lora_params:
        if name not in lora_state:
            missing.append(name)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Error loading LoRA weights:\n"
            f"  Missing keys: {missing}\n"
            f"  Unexpected keys: {unexpected}"
        )

    if missing:
        logger.warning(f"Missing LoRA keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected LoRA keys: {unexpected}")

    logger.info(f"Loaded LoRA weights from {path}")
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in the model, separated by type.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total = 0
    trainable = 0
    lora = 0

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
        if 'lora_' in name:
            lora += param.numel()

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "lora": lora,
    }
