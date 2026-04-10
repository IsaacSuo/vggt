from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
for path in (REPO_ROOT, TRAINING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmark.metrics import compute_sample_metrics
from benchmark.model_loader import load_model
from benchmark.plan import BenchmarkPlan, load_plan
from benchmark.registry import create_dataset_adapter


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone multi-model, multi-dataset VGGT benchmarks."
    )
    parser.add_argument("--plan", required=True, help="Path to the benchmark plan JSON.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary and per-sample results will be written.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. `cuda:0` or `cpu`.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_override: str | None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def evaluate_model_on_dataset(model, model_spec, dataset_spec, device: torch.device, amp_dtype: torch.dtype) -> List[Dict[str, object]]:
    adapter = create_dataset_adapter(dataset_spec)
    rows: List[Dict[str, object]] = []

    for sample in adapter.iter_samples():
        images = sample.images.to(device)
        masks = sample.masks.to(device) if (sample.masks is not None and model_spec.use_visual_hull_mask) else None

        autocast_enabled = device.type == "cuda"
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=amp_dtype):
                predictions = model(images, visual_hull_mask=masks)

        metrics = compute_sample_metrics(predictions, sample)

        row: Dict[str, object] = {
            "model": model_spec.name,
            "dataset": dataset_spec.name,
            "sample_id": sample.sample_id,
        }
        row.update(sample.metadata)
        row.update(metrics)
        rows.append(row)

    return rows


def aggregate_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["dataset"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (model_name, dataset_name), group_rows in sorted(grouped.items()):
        metric_names = sorted({
            key for row in group_rows for key, value in row.items()
            if key not in {"model", "dataset", "sample_id", "seq_name", "frame_ids"}
            and isinstance(value, (int, float))
        })
        summary: Dict[str, object] = {
            "model": model_name,
            "dataset": dataset_name,
            "sample_count": len(group_rows),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group_rows if metric_name in row]
            if values:
                summary[metric_name] = float(sum(values) / len(values))
        summary_rows.append(summary)

    return summary_rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    if not summary_rows:
        path.write_text("# Benchmark Summary\n\nNo rows.\n", encoding="utf-8")
        return

    metric_names = sorted({
        key for row in summary_rows for key in row.keys()
        if key not in {"model", "dataset", "sample_count"}
    })
    header = "| model | dataset | sample_count | " + " | ".join(metric_names) + " |"
    separator = "| --- | --- | ---: | " + " | ".join(["---:"] * len(metric_names)) + " |"
    lines = ["# Benchmark Summary", "", header, separator]

    for row in summary_rows:
        metric_values = []
        for key in metric_names:
            value = row.get(key, "")
            if isinstance(value, float):
                metric_values.append(f"{value:.6f}")
            else:
                metric_values.append(str(value))
        lines.append(
            "| {model} | {dataset} | {sample_count} | {metrics} |".format(
                model=row["model"],
                dataset=row["dataset"],
                sample_count=row["sample_count"],
                metrics=" | ".join(metric_values),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(plan: BenchmarkPlan, output_dir: Path, device: torch.device) -> None:
    amp_dtype = choose_amp_dtype(device)
    all_rows: List[Dict[str, object]] = []
    model_metadata: Dict[str, object] = {}

    for model_spec in plan.models:
        model, metadata = load_model(model_spec, device)
        model_metadata[model_spec.name] = metadata

        for dataset_spec in plan.datasets:
            logger.info("Evaluating model `%s` on dataset `%s`", model_spec.name, dataset_spec.name)
            rows = evaluate_model_on_dataset(model, model_spec, dataset_spec, device, amp_dtype)
            all_rows.extend(rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = aggregate_rows(all_rows)

    write_json(output_dir / "plan_snapshot.json", {
        "seed": plan.seed,
        "models": [asdict(model) for model in plan.models],
        "datasets": [asdict(dataset) for dataset in plan.datasets],
    })
    write_json(output_dir / "model_load_metadata.json", model_metadata)
    write_json(output_dir / "per_sample.json", all_rows)
    write_csv(output_dir / "per_sample.csv", all_rows)
    write_json(output_dir / "summary.json", summary_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_markdown_summary(output_dir / "summary.md", summary_rows)


def main() -> None:
    args = parse_args()
    configure_logging()

    plan = load_plan(args.plan)
    set_seeds(plan.seed)

    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)
    logger.info("Running benchmark on device %s", device)
    run(plan, output_dir, device)


if __name__ == "__main__":
    main()
