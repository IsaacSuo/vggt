from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LoRASpec:
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["qkv"])
    target_block_type: str = "global"
    block_indices: Optional[List[int]] = None
    freeze_base: bool = False
    merge_after_load: bool = False


@dataclass
class ModelSpec:
    name: str
    checkpoint_path: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    use_visual_hull_mask: bool = True
    lora: Optional[LoRASpec] = None


@dataclass
class DatasetSpec:
    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkPlan:
    models: List[ModelSpec]
    datasets: List[DatasetSpec]
    seed: int = 42


def _parse_lora_spec(raw: Optional[Dict[str, Any]]) -> Optional[LoRASpec]:
    if raw is None:
        return None
    return LoRASpec(**raw)


def _parse_model_spec(raw: Dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=raw["name"],
        checkpoint_path=raw["checkpoint_path"],
        model_kwargs=raw.get("model_kwargs", {}),
        strict=bool(raw.get("strict", False)),
        use_visual_hull_mask=bool(raw.get("use_visual_hull_mask", True)),
        lora=_parse_lora_spec(raw.get("lora")),
    )


def _parse_dataset_spec(raw: Dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        name=raw["name"],
        type=raw["type"],
        config=raw.get("config", {}),
    )


def load_plan(path: str | Path) -> BenchmarkPlan:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    models = [_parse_model_spec(item) for item in raw.get("models", [])]
    datasets = [_parse_dataset_spec(item) for item in raw.get("datasets", [])]

    if not models:
        raise ValueError("Benchmark plan must contain at least one model.")
    if not datasets:
        raise ValueError("Benchmark plan must contain at least one dataset.")

    return BenchmarkPlan(
        models=models,
        datasets=datasets,
        seed=int(raw.get("seed", 42)),
    )
