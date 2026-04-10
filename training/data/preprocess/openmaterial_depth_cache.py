#!/usr/bin/env python3

import argparse
import concurrent.futures
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

_NVDIFFRAST_CTX = None


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
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=["cpu", "pytorch3d", "nvdiffrast"],
        help="Rasterization backend. GPU backends require the corresponding library and a CUDA device.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of scene workers for CPU precompute. Ignored by the GPU backend.",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=8,
        help="Number of frames per rasterization batch for the GPU backends.",
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


def resolve_image_shape(frame: Dict, intrinsics: Dict[str, float], image_path: str) -> Tuple[int, int]:
    height = frame.get("h", intrinsics.get("h"))
    width = frame.get("w", intrinsics.get("w"))
    if height is not None and width is not None:
        return int(height), int(width)
    return load_image_shape(image_path)


def collect_scene_jobs(
    data_dir: str,
    split: str,
    scene_filter: Optional[str],
    limit_scenes: Optional[int],
) -> List[Dict]:
    jobs_by_scene: Dict[str, Dict] = {}
    for transform_path in iter_transform_files(data_dir, split, scene_filter):
        scene_dir = osp.dirname(transform_path)
        job = jobs_by_scene.setdefault(
            scene_dir,
            {
                "scene_dir": scene_dir,
                "scene_name": osp.basename(scene_dir),
                "hash_id": osp.basename(osp.dirname(scene_dir)),
                "transform_paths": {},
            },
        )
        split_name = "train" if transform_path.endswith("transforms_train.json") else "test"
        job["transform_paths"][split_name] = transform_path

    jobs = [jobs_by_scene[key] for key in sorted(jobs_by_scene.keys())]
    if limit_scenes is not None:
        jobs = jobs[:limit_scenes]
    return jobs


def _rasterize_mesh_depth_pytorch3d_batch(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    extrinsics: List[np.ndarray],
    intrinsic: np.ndarray,
    image_shape_hw: Tuple[int, int],
    near_plane: float,
) -> List[np.ndarray]:
    import torch
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.structures import Meshes

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch3D backend requested but CUDA is not available")

    device = torch.device("cuda")
    height, width = image_shape_hw
    if vertices_world.size == 0 or faces.size == 0:
        return [np.zeros((height, width), dtype=np.float32) for _ in extrinsics]

    vertices_world_t = torch.from_numpy(vertices_world).to(device=device, dtype=torch.float32)
    faces_t = torch.from_numpy(faces).to(device=device, dtype=torch.int64)

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    scale = float(min(height, width))

    verts_batches = []
    for extrinsic in extrinsics:
        extrinsic_t = torch.from_numpy(extrinsic).to(device=device, dtype=torch.float32)
        verts_cam = vertices_world_t @ extrinsic_t[:3, :3].T + extrinsic_t[:3, 3]

        z = verts_cam[:, 2]
        x_screen = fx * (verts_cam[:, 0] / z) + cx
        y_screen = fy * (verts_cam[:, 1] / z) + cy
        x_ndc = -((x_screen - (width / 2.0)) * 2.0 / scale)
        y_ndc = -((y_screen - (height / 2.0)) * 2.0 / scale)
        verts_batches.append(torch.stack([x_ndc, y_ndc, z], dim=-1))

    meshes = Meshes(verts=verts_batches, faces=[faces_t] * len(verts_batches))
    _, zbuf, _, _ = rasterize_meshes(
        meshes,
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        perspective_correct=True,
        clip_barycentric_coords=True,
        cull_backfaces=False,
        z_clip_value=near_plane,
        cull_to_frustum=True,
    )

    depth_maps = zbuf[..., 0].detach().cpu().numpy().astype(np.float32, copy=False)
    depth_maps[depth_maps < near_plane] = 0.0
    depth_maps[~np.isfinite(depth_maps)] = 0.0
    depth_maps[depth_maps < 0.0] = 0.0
    return [depth_maps[idx] for idx in range(depth_maps.shape[0])]


def _get_nvdiffrast_context(device):
    global _NVDIFFRAST_CTX
    if _NVDIFFRAST_CTX is None:
        import nvdiffrast.torch as dr

        _NVDIFFRAST_CTX = dr.RasterizeCudaContext(device=device)
    return _NVDIFFRAST_CTX


def _rasterize_mesh_depth_nvdiffrast_batch(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    extrinsics: List[np.ndarray],
    intrinsic: np.ndarray,
    image_shape_hw: Tuple[int, int],
    near_plane: float,
) -> List[np.ndarray]:
    import torch
    import nvdiffrast.torch as dr

    if not torch.cuda.is_available():
        raise RuntimeError("nvdiffrast backend requested but CUDA is not available")

    device = torch.device("cuda")
    height, width = image_shape_hw
    if vertices_world.size == 0 or faces.size == 0:
        return [np.zeros((height, width), dtype=np.float32) for _ in extrinsics]

    vertices_world_t = torch.from_numpy(vertices_world).to(device=device, dtype=torch.float32)
    faces_t = torch.from_numpy(faces).to(device=device, dtype=torch.int32).contiguous()

    batch_size = len(extrinsics)
    extrinsics_t = torch.from_numpy(np.stack(extrinsics, axis=0)).to(device=device, dtype=torch.float32)
    verts_cam = vertices_world_t.unsqueeze(0) @ extrinsics_t[:, :3, :3].transpose(1, 2)
    verts_cam = verts_cam + extrinsics_t[:, None, :3, 3]

    z_cam = verts_cam[..., 2]
    positive_z = z_cam[z_cam > near_plane]
    if positive_z.numel() == 0:
        return [np.zeros((height, width), dtype=np.float32) for _ in extrinsics]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    z_safe = torch.clamp(z_cam, min=near_plane)
    x_screen = fx * (verts_cam[..., 0] / z_safe) + cx
    y_screen = fy * (verts_cam[..., 1] / z_safe) + cy
    x_ndc = (2.0 * x_screen / float(width)) - 1.0
    y_ndc = 1.0 - (2.0 * y_screen / float(height))

    far_plane = torch.max(positive_z).item() * 1.01
    far_plane = max(far_plane, near_plane + 1.0)
    z_ndc = (2.0 * (z_safe - near_plane) / (far_plane - near_plane)) - 1.0

    clip_pos = torch.stack(
        [
            x_ndc * z_safe,
            y_ndc * z_safe,
            z_ndc * z_safe,
            z_safe,
        ],
        dim=-1,
    ).contiguous()

    depth_attr = z_cam.unsqueeze(-1).contiguous()
    glctx = _get_nvdiffrast_context(device)
    rast, rast_db = dr.rasterize(glctx, clip_pos, faces_t, (height, width))
    interpolated_depth, _ = dr.interpolate(depth_attr, rast, faces_t, rast_db=rast_db)

    depth = interpolated_depth[..., 0]
    depth = torch.where(rast[..., 3] > 0.0, depth, torch.zeros_like(depth))
    depth = torch.where(depth > near_plane, depth, torch.zeros_like(depth))
    depth = torch.flip(depth, dims=[1])
    depth = depth.detach().cpu().numpy().astype(np.float32, copy=False)
    depth[~np.isfinite(depth)] = 0.0
    return [depth[idx] for idx in range(batch_size)]


def precompute_scene(
    scene: Dict,
    frames: List[Dict],
    intrinsics: Dict[str, float],
    data_dir: str,
    depth_subdir: str,
    output_root: Optional[str],
    near_plane: float,
    overwrite: bool,
    backend: str,
    frame_batch_size: int,
    vertices_world: np.ndarray,
    faces: np.ndarray,
) -> Tuple[int, int]:
    intri_opencv = build_intrinsic_matrix(intrinsics)

    written = 0
    skipped = 0
    pending_frames = []

    def flush_gpu_batch() -> None:
        nonlocal written
        if not pending_frames:
            return
        extrinsics = [item["extrinsic"] for item in pending_frames]
        if backend == "pytorch3d":
            depth_maps = _rasterize_mesh_depth_pytorch3d_batch(
                vertices_world=vertices_world,
                faces=faces,
                extrinsics=extrinsics,
                intrinsic=intri_opencv,
                image_shape_hw=pending_frames[0]["image_shape_hw"],
                near_plane=near_plane,
            )
        elif backend == "nvdiffrast":
            depth_maps = _rasterize_mesh_depth_nvdiffrast_batch(
                vertices_world=vertices_world,
                faces=faces,
                extrinsics=extrinsics,
                intrinsic=intri_opencv,
                image_shape_hw=pending_frames[0]["image_shape_hw"],
                near_plane=near_plane,
            )
        else:
            raise ValueError(f"Unsupported GPU backend: {backend}")
        for item, depth_map in zip(pending_frames, depth_maps):
            np.save(item["depth_path"], depth_map.astype(np.float32, copy=False))
            written += 1
        pending_frames.clear()

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
        image_shape_hw = resolve_image_shape(frame, intrinsics, image_path)

        if backend in {"pytorch3d", "nvdiffrast"}:
            if pending_frames and pending_frames[0]["image_shape_hw"] != image_shape_hw:
                flush_gpu_batch()
            pending_frames.append(
                {
                    "depth_path": depth_path,
                    "extrinsic": extri_opencv,
                    "image_shape_hw": image_shape_hw,
                }
            )
            if len(pending_frames) >= frame_batch_size:
                flush_gpu_batch()
            continue

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

    if backend in {"pytorch3d", "nvdiffrast"}:
        flush_gpu_batch()

    return written, skipped


def process_scene_job(
    job: Dict,
    data_dir: str,
    depth_subdir: str,
    output_root: Optional[str],
    near_plane: float,
    overwrite: bool,
    backend: str,
    frame_batch_size: int,
) -> Dict:
    mesh_path = _resolve_gt_mesh_path(data_dir, job["hash_id"])
    if mesh_path is None:
        raise FileNotFoundError(
            f"GT mesh not found for scene `{job['scene_name']}` (hash `{job['hash_id']}`)"
        )

    vertices_world, faces = _load_ply_mesh(mesh_path)
    scene = {
        "dir": job["scene_dir"],
        "name": job["scene_name"],
        "hash_id": job["hash_id"],
    }

    written = 0
    skipped = 0
    split_stats = []
    for split_name in ("train", "test"):
        transform_path = job["transform_paths"].get(split_name)
        if transform_path is None:
            continue
        _, frames, intrinsics = load_scene_metadata(transform_path)
        split_written, split_skipped = precompute_scene(
            scene=scene,
            frames=frames,
            intrinsics=intrinsics,
            data_dir=data_dir,
            depth_subdir=depth_subdir,
            output_root=output_root,
            near_plane=near_plane,
            overwrite=overwrite,
            backend=backend,
            frame_batch_size=frame_batch_size,
            vertices_world=vertices_world,
            faces=faces,
        )
        written += split_written
        skipped += split_skipped
        split_stats.append(f"{split_name}: wrote {split_written}, skipped {split_skipped}")

    return {
        "scene_name": job["scene_name"],
        "written": written,
        "skipped": skipped,
        "split_stats": ", ".join(split_stats),
    }


def main() -> None:
    args = parse_args()
    jobs = collect_scene_jobs(
        data_dir=args.data_dir,
        split=args.split,
        scene_filter=args.scene_filter,
        limit_scenes=args.limit_scenes,
    )

    if not jobs:
        raise ValueError("No OpenMaterial scenes matched the requested filters.")

    total_scenes = 0
    total_written = 0
    total_skipped = 0

    if args.backend in {"pytorch3d", "nvdiffrast"}:
        for index, job in enumerate(jobs, start=1):
            result = process_scene_job(
                job=job,
                data_dir=args.data_dir,
                depth_subdir=args.depth_subdir,
                output_root=args.output_root,
                near_plane=args.near_plane,
                overwrite=args.overwrite,
                backend=args.backend,
                frame_batch_size=args.frame_batch_size,
            )
            total_scenes += 1
            total_written += result["written"]
            total_skipped += result["skipped"]
            print(
                f"[{index}/{len(jobs)}] {result['scene_name']}: "
                f"wrote {result['written']}, skipped {result['skipped']} "
                f"({result['split_stats']})"
            )
    else:
        max_workers = max(1, int(args.num_workers))
        if max_workers == 1:
            for index, job in enumerate(jobs, start=1):
                result = process_scene_job(
                    job=job,
                    data_dir=args.data_dir,
                    depth_subdir=args.depth_subdir,
                    output_root=args.output_root,
                    near_plane=args.near_plane,
                    overwrite=args.overwrite,
                    backend=args.backend,
                    frame_batch_size=args.frame_batch_size,
                )
                total_scenes += 1
                total_written += result["written"]
                total_skipped += result["skipped"]
                print(
                    f"[{index}/{len(jobs)}] {result['scene_name']}: "
                    f"wrote {result['written']}, skipped {result['skipped']} "
                    f"({result['split_stats']})"
                )
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_scene_job,
                        job,
                        args.data_dir,
                        args.depth_subdir,
                        args.output_root,
                        args.near_plane,
                        args.overwrite,
                        args.backend,
                        args.frame_batch_size,
                    )
                    for job in jobs
                ]
                for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    result = future.result()
                    total_scenes += 1
                    total_written += result["written"]
                    total_skipped += result["skipped"]
                    print(
                        f"[{index}/{len(jobs)}] {result['scene_name']}: "
                        f"wrote {result['written']}, skipped {result['skipped']} "
                        f"({result['split_stats']})"
                    )

    print(
        "Done. "
        f"Scenes: {total_scenes}, depth files written: {total_written}, skipped: {total_skipped}"
    )


if __name__ == "__main__":
    main()
