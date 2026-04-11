from __future__ import annotations

from benchmark.adapters.base import BenchmarkDatasetAdapter
from benchmark.plan import DatasetSpec


def create_dataset_adapter(spec: DatasetSpec) -> BenchmarkDatasetAdapter:
    if spec.type == "openmaterial":
        from benchmark.adapters.openmaterial import OpenMaterialBenchmarkAdapter

        return OpenMaterialBenchmarkAdapter(spec)
    if spec.type == "nero_glossy_synthetic":
        from benchmark.adapters.nero import NeROGlossySyntheticAdapter

        return NeROGlossySyntheticAdapter(spec)
    if spec.type == "nero_glossy_real":
        from benchmark.adapters.nero import NeROGlossyRealAdapter

        return NeROGlossyRealAdapter(spec)
    raise ValueError(f"Unsupported benchmark dataset type: {spec.type}")
