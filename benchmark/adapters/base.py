from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import torch


@dataclass
class BenchmarkSample:
    sample_id: str
    images: torch.Tensor
    masks: Optional[torch.Tensor] = None
    depths: Optional[torch.Tensor] = None
    cam_points: Optional[torch.Tensor] = None
    world_points: Optional[torch.Tensor] = None
    point_masks: Optional[torch.Tensor] = None
    extrinsics: Optional[torch.Tensor] = None
    intrinsics: Optional[torch.Tensor] = None
    raw_depths: Optional[torch.Tensor] = None
    raw_extrinsics: Optional[torch.Tensor] = None
    raw_intrinsics: Optional[torch.Tensor] = None
    normalization_scale: Optional[float] = None
    normalization_reference_extrinsic: Optional[torch.Tensor] = None
    gt_mesh_path: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    protocol: Dict[str, object] = field(default_factory=dict)


class BenchmarkDatasetAdapter(ABC):
    def __init__(self, spec):
        self.spec = spec

    @abstractmethod
    def iter_samples(self) -> Iterator[BenchmarkSample]:
        raise NotImplementedError
