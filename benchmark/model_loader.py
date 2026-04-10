from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

from benchmark.plan import LoRASpec, ModelSpec
from vggt.models.vggt import VGGT


logger = logging.getLogger(__name__)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module.") :]: value for key, value in state_dict.items()}


def _extract_model_state_dict(checkpoint_obj: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "model" in checkpoint_obj:
        return checkpoint_obj["model"]
    return checkpoint_obj


def _enable_lora(model: VGGT, lora_spec: LoRASpec) -> None:
    model.enable_lora(
        config_or_rank=lora_spec.rank,
        alpha=lora_spec.alpha,
        dropout=lora_spec.dropout,
        target_modules=lora_spec.target_modules,
        target_block_type=lora_spec.target_block_type,
        block_indices=lora_spec.block_indices,
        freeze_base=lora_spec.freeze_base,
    )


def load_model(spec: ModelSpec, device: torch.device) -> Tuple[VGGT, Dict[str, object]]:
    checkpoint_path = Path(spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    logger.info("Loading model `%s` from %s", spec.name, checkpoint_path)
    model = VGGT(**spec.model_kwargs)

    if spec.lora is not None:
        logger.info("Enabling LoRA for model `%s` before loading checkpoint", spec.name)
        _enable_lora(model, spec.lora)

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_model_state_dict(checkpoint_obj))
    missing, unexpected = model.load_state_dict(state_dict, strict=spec.strict)

    if spec.lora is not None and spec.lora.merge_after_load:
        logger.info("Merging LoRA weights into base model for `%s`", spec.name)
        model.merge_lora()

    model.eval()
    model.to(device)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "strict": spec.strict,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "use_visual_hull_mask": spec.use_visual_hull_mask,
    }
    return model, metadata
