#!/usr/bin/env python3

import argparse
import json
import os
import os.path as osp
import sys
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Keep this script runnable from the repo root without manual PYTHONPATH juggling.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
TRAINING_ROOT = THIS_FILE.parents[2]
for path in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from training.data.datasets.openmaterial import (
    _load_ply_mesh,
    _rasterize_mesh_depth,
    _resolve_gt_mesh_path,
    _resolve_precomputed_depth_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline precompute OpenMaterial mesh-depth supervision."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to OpenMaterial/datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which split to precompute",
    )
    parser.add_argument(
        "--depth_subdir",
        type=str,
        default="depth_mesh",
        help="Per-scene subdirectory name used to store cached .npy depth maps",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional separate cache root. If omitted, writes next to each scene under depth_subdir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached depths",
    )
    parser.add_argument(
        "--scene_filter",
        type=str,
        default=None,
        help="Optional substring filter on scene path",
    )
    parser.add_argument(
        "--limit_scenes",
        type=int,
        default=None,
        help="Optional cap on the number of scenes to process",
    )
    parser.add_argument(
        "--near_plane",
        type=float,
        default=1e-3,
        help="Near plane passed to the mesh rasterizer",
    )
    return parser.parse_args()


def iter_transform_files(data_dir: str, split: str, scene_filter: Optional[str]) -> Iterable[str]:
    splits = ["train", "test"] if split == "both" else [split]
    for split_name in splits:
        pattern = osp.join(data_dir, "*", "*", f"transforms_{split_name}.json")
        for transform_path in sorted(glob(pattern)):
            if scene_filter and scene_filter not in transform_path:
                continue
            yield transform_path


def build_intrinsic_matrix(intrinsics: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [intrinsics["fl_x"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fl_y"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def convert_frame_to_extrinsic(frame: Dict) -> np.ndarray:
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w_opencv = c2w.copy()
    c2w_opencv[:3, 1:3] *= -1
    return np.linalg.inv(c2w_opencv)


def load_scene_metadata(transform_path: str) -> Tuple[Dict, List[Dict], Dict[str, float]]:
    with open(transform_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    intrinsics = {
        "fl_x": data["fl_x"],
        "fl_y": data["fl_y"],
        "cx": data["cx"],
        "cy": data["cy"],
        "w": data.get("w"),
        "h": data.get("h"),
    }
    scene = {
        "dir": osp.dirname(transform_path),
        "name": osp.basename(osp.dirname(transform_path)),
        "hash_id": osp.basename(osp.dirname(osp.dirname(transform_path))),
        "transform_path": transform_path,
    }
    return scene, frames, intrinsics


def resolve_image_path(scene_dir: str, frame: Dict) -> str:
    image_path = osp.join(scene_dir, frame["file_path"])
    if osp.exists(image_path):
        return image_path
    fallback = osp.join(scene_dir, osp.basename(frame["file_path"]))
    if osp.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Image file not found for frame `{frame['file_path']}`")


def load_image_shape(image_path: str) -> Tuple[int, int]:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    return tuple(image.shape[:2])


def precompute_scene(
    scene: Dict,
    frames: List[Dict],
    intrinsics: Dict[str, float],
    data_dir: str,
    depth_subdir: str,
    output_root: Optional[str],
    near_plane: float,
    overwrite: bool,
) -> Tuple[int, int]:
    mesh_path = _resolve_gt_mesh_path(data_dir, scene["hash_id"])
    if mesh_path is None:
        raise FileNotFoundError(
            f"GT mesh not found for scene `{scene['name']}` (hash `{scene['hash_id']}`)"
        )

    vertices_world, faces = _load_ply_mesh(mesh_path)
    intri_opencv = build_intrinsic_matrix(intrinsics)

    written = 0
    skipped = 0
    for frame in frames:
        image_path = resolve_image_path(scene["dir"], frame)
        depth_path = _resolve_precomputed_depth_path(
            image_path,
            depth_subdir=depth_subdir,
            scene_dir=scene["dir"],
            cache_root=output_root,
        )
        if osp.exists(depth_path) and not overwrite:
            skipped += 1
            continue

        os.makedirs(osp.dirname(depth_path), exist_ok=True)
        extri_opencv = convert_frame_to_extrinsic(frame)
        image_shape_hw = load_image_shape(image_path)
        depth_map = _rasterize_mesh_depth(
            vertices_world,
            faces,
            extri_opencv,
            intri_opencv,
            image_shape_hw,
            near_plane=near_plane,
        )
        np.save(depth_path, depth_map.astype(np.float32, copy=False))
        written += 1

    return written, skipped


def main() -> None:
    args = parse_args()
    transform_files = list(iter_transform_files(args.data_dir, args.split, args.scene_filter))
    if args.limit_scenes is not None:
        transform_files = transform_files[: args.limit_scenes]

    if not transform_files:
        raise ValueError("No OpenMaterial scenes matched the requested filters.")

    total_scenes = 0
    total_written = 0
    total_skipped = 0

    for index, transform_path in enumerate(transform_files, start=1):
        scene, frames, intrinsics = load_scene_metadata(transform_path)
        written, skipped = precompute_scene(
            scene=scene,
            frames=frames,
            intrinsics=intrinsics,
            data_dir=args.data_dir,
            depth_subdir=args.depth_subdir,
            output_root=args.output_root,
            near_plane=args.near_plane,
            overwrite=args.overwrite,
        )
        total_scenes += 1
        total_written += written
        total_skipped += skipped
        print(
            f"[{index}/{len(transform_files)}] {scene['name']}: "
            f"wrote {written}, skipped {skipped}"
        )

    print(
        "Done. "
        f"Scenes: {total_scenes}, depth files written: {total_written}, skipped: {total_skipped}"
    )


if __name__ == "__main__":
    main()
