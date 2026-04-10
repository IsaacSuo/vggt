#!/usr/bin/env python3

import argparse
import json
import os
import os.path as osp
import random
from collections import defaultdict
from glob import glob
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create scene-disjoint OpenMaterial train/test manifests."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to OpenMaterial dataset root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where train.txt, test.txt, and summary.json will be written.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of scenes assigned to the held-out test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation.",
    )
    parser.add_argument(
        "--min_num_images",
        type=int,
        default=8,
        help="Only include scenes whose train/test transform file has at least this many frames.",
    )
    return parser.parse_args()


def _infer_material_group(scene_name: str) -> str:
    if "-" not in scene_name:
        return "unknown"
    return scene_name.rsplit("-", 1)[-1]


def _load_scene_records(data_dir: str, min_num_images: int) -> List[Dict[str, str]]:
    records = []
    pattern = osp.join(data_dir, "*", "*", "transforms_train.json")
    for train_tf in sorted(glob(pattern)):
        scene_dir = osp.dirname(train_tf)
        test_tf = osp.join(scene_dir, "transforms_test.json")
        if not osp.exists(test_tf):
            continue

        with open(train_tf, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(test_tf, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        if len(train_data.get("frames", [])) < min_num_images:
            continue
        if len(test_data.get("frames", [])) < min_num_images:
            continue

        scene_name = osp.basename(scene_dir)
        hash_id = osp.basename(osp.dirname(scene_dir))
        records.append(
            {
                "id": f"{hash_id}/{scene_name}",
                "scene_name": scene_name,
                "hash_id": hash_id,
                "material_group": _infer_material_group(scene_name),
            }
        )
    return records


def _split_group(records: List[Dict[str, str]], test_ratio: float, rng: random.Random) -> tuple[List[str], List[str]]:
    ids = [record["id"] for record in records]
    rng.shuffle(ids)

    if len(ids) == 1:
        return ids, []

    test_count = int(round(len(ids) * test_ratio))
    test_count = max(1, test_count)
    test_count = min(len(ids) - 1, test_count)
    train_count = len(ids) - test_count

    return ids[:train_count], ids[train_count:]


def _write_manifest(path: str, scene_ids: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for scene_id in sorted(scene_ids):
            f.write(scene_id + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test_ratio must be between 0 and 1")

    records = _load_scene_records(args.data_dir, min_num_images=args.min_num_images)
    if not records:
        raise ValueError("No OpenMaterial scenes matched the requested constraints.")

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["material_group"]].append(record)

    rng = random.Random(args.seed)
    train_ids: List[str] = []
    test_ids: List[str] = []
    group_summary = {}

    for group_name in sorted(grouped.keys()):
        group_train, group_test = _split_group(grouped[group_name], args.test_ratio, rng)
        train_ids.extend(group_train)
        test_ids.extend(group_test)
        group_summary[group_name] = {
            "total": len(grouped[group_name]),
            "train": len(group_train),
            "test": len(group_test),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = osp.join(args.output_dir, "train.txt")
    test_path = osp.join(args.output_dir, "test.txt")
    summary_path = osp.join(args.output_dir, "summary.json")

    _write_manifest(train_path, train_ids)
    _write_manifest(test_path, test_ids)

    summary = {
        "data_dir": args.data_dir,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "min_num_images": args.min_num_images,
        "total_scenes": len(records),
        "train_scenes": len(train_ids),
        "test_scenes": len(test_ids),
        "groups": group_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote train manifest: {train_path} ({len(train_ids)} scenes)")
    print(f"Wrote test manifest:  {test_path} ({len(test_ids)} scenes)")
    print(f"Wrote summary:        {summary_path}")


if __name__ == "__main__":
    main()
