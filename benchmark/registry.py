from __future__ import annotations

from benchmark.adapters.base import BenchmarkDatasetAdapter
from benchmark.adapters.openmaterial import OpenMaterialBenchmarkAdapter
from benchmark.plan import DatasetSpec


def create_dataset_adapter(spec: DatasetSpec) -> BenchmarkDatasetAdapter:
    if spec.type == "openmaterial":
        return OpenMaterialBenchmarkAdapter(spec)
    raise ValueError(f"Unsupported benchmark dataset type: {spec.type}")
