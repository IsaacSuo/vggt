from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
for path in (REPO_ROOT, TRAINING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmark.metrics import (
    _chunked_nearest_distances,
    _denormalize_extrinsics,
    _points_bbox_diagonal,
)
from benchmark.model_loader import load_model
from benchmark.plan import BenchmarkPlan, DatasetSpec, ModelSpec, load_plan
from benchmark.registry import create_dataset_adapter
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


_PLY_NUMPY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render NeRO GlossyReal benchmark reconstructions from dataset viewpoints."
    )
    parser.add_argument(
        "--plan",
        default="benchmark/examples/nero_glossyreal_plan.json",
        help="Path to benchmark plan JSON.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name override from the benchmark plan. Defaults to the first dataset.",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene name to visualize, e.g. `bear` or `GlossyReal/bear`.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names from the plan to visualize. Defaults to all models in the plan.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Index inside the selected benchmark frames to render from. Defaults to the middle frame.",
    )
    parser.add_argument(
        "--frame-id",
        default=None,
        help="Dataset frame id to render from, e.g. `1462` or `1462.jpg`. Overrides --frame-index.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where renders and metadata will be written.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. `cuda:0` or `cpu`.",
    )
    parser.add_argument(
        "--max-render-points",
        type=int,
        default=250000,
        help="Maximum number of points kept for rendering each point cloud.",
    )
    parser.add_argument(
        "--max-error-gt-points",
        type=int,
        default=120000,
        help="Maximum number of GT points used for nearest-neighbor error coloring.",
    )
    parser.add_argument(
        "--pred-pixel-stride",
        type=int,
        default=1,
        help="Stride used before back-projecting predicted depths. Increase to reduce density.",
    )
    parser.add_argument(
        "--voxel-size-ratio",
        type=float,
        default=1.0 / 256.0,
        help="Voxel size as a fraction of GT bbox diagonal for point cloud fusion/downsampling.",
    )
    parser.add_argument(
        "--render-radius",
        type=int,
        default=1,
        help="Point splat radius in pixels during rendering.",
    )
    parser.add_argument(
        "--error-max-ratio",
        type=float,
        default=0.02,
        help="Clip error colors at this fraction of GT bbox diagonal.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.08,
        help="Extra crop padding ratio around the rendered object bbox.",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Save rendered/downsampled point clouds as PLY files.",
    )
    return parser.parse_args()


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


def _select_dataset_spec(plan: BenchmarkPlan, dataset_name: str | None) -> DatasetSpec:
    if dataset_name is None:
        return plan.datasets[0]
    for spec in plan.datasets:
        if spec.name == dataset_name:
            return spec
    raise ValueError(f"Dataset `{dataset_name}` was not found in plan `{plan}`.")


def _select_model_specs(plan: BenchmarkPlan, model_names: Sequence[str] | None) -> List[ModelSpec]:
    if model_names is None:
        return list(plan.models)
    requested = set(model_names)
    specs = [spec for spec in plan.models if spec.name in requested]
    missing = requested.difference(spec.name for spec in specs)
    if missing:
        raise ValueError(f"Model(s) not found in plan: {sorted(missing)}")
    return specs


def _normalize_scene_query(scene: str) -> str:
    if scene.startswith("GlossyReal/"):
        return scene.split("/", 1)[1]
    return scene


def _find_sample(dataset_spec: DatasetSpec, scene_name: str):
    spec = DatasetSpec(
        name=dataset_spec.name,
        type=dataset_spec.type,
        config=dict(dataset_spec.config),
    )
    spec.config["scene_names"] = [_normalize_scene_query(scene_name)]
    adapter = create_dataset_adapter(spec)
    for sample in adapter.iter_samples():
        if sample.metadata.get("seq_name") == _normalize_scene_query(scene_name):
            return sample
    raise ValueError(f"Scene `{scene_name}` was not found in dataset `{dataset_spec.name}`.")


def _binary_ply_vertices_with_color(ply_bytes: bytes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    stream = io.BytesIO(ply_bytes)
    vertex_count = None
    vertex_properties: List[Tuple[str, str]] = []
    reading_vertex_properties = False
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header.")
        decoded = line.decode("ascii", errors="strict").strip()
        if decoded in {"ply", "format binary_little_endian 1.0"} or decoded.startswith("comment "):
            continue
        if decoded.startswith("obj_info "):
            continue
        if decoded.startswith("element vertex "):
            vertex_count = int(decoded.split()[2])
            vertex_properties = []
            reading_vertex_properties = True
            continue
        if decoded.startswith("element "):
            reading_vertex_properties = False
            continue
        if decoded.startswith("property ") and reading_vertex_properties:
            _, prop_type, prop_name = decoded.split()
            if prop_type not in _PLY_NUMPY_DTYPE_MAP:
                raise ValueError(f"Unsupported PLY property type `{prop_type}`.")
            vertex_properties.append((prop_name, "<" + _PLY_NUMPY_DTYPE_MAP[prop_type]))
            continue
        if decoded == "end_header":
            break

    if vertex_count is None or not vertex_properties:
        raise ValueError("PLY vertex metadata is missing.")

    vertex_dtype = np.dtype(vertex_properties)
    vertices = np.frombuffer(ply_bytes, dtype=vertex_dtype, count=vertex_count, offset=stream.tell())
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32, copy=False)

    color = None
    if all(channel in vertices.dtype.names for channel in ("red", "green", "blue")):
        color = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=-1)
        if color.dtype != np.uint8:
            color = color.astype(np.uint8, copy=False)
    return xyz, color


def _load_gt_colors_from_archive(dataset_spec: DatasetSpec, scene_name: str) -> Optional[np.ndarray]:
    archive_path = Path(dataset_spec.config["data_path"]).expanduser()
    member_name = f"GlossyReal/{scene_name}/object_point_cloud.ply"
    with tarfile.open(archive_path, "r:*") as tf:
        try:
            member = tf.getmember(member_name)
        except KeyError:
            return None
        _, colors = _binary_ply_vertices_with_color(tf.extractfile(member).read())
        return colors


def _torch_to_image_rgb(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _stable_seed(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _subsample_points(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    indices = np.sort(indices)
    sampled_points = points[indices]
    sampled_colors = colors[indices] if colors is not None else None
    return sampled_points, sampled_colors


def _voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.shape[0] == 0 or voxel_size <= 0:
        return points, colors

    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    down_points = points[unique_indices]
    down_colors = colors[unique_indices] if colors is not None else None
    return down_points, down_colors


def _predict_camera_and_depth(sample, predictions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_hw = tuple(sample.images.shape[-2:])
    pred_pose = predictions["pose_enc"].detach()
    pred_extrinsics_norm, pred_intrinsics = pose_encoding_to_extri_intri(pred_pose, image_size_hw=image_hw)
    pred_extrinsics = _denormalize_extrinsics(pred_extrinsics_norm.squeeze(0), sample)
    pred_intrinsics = pred_intrinsics.squeeze(0)
    pred_depth = predictions["depth"].detach().squeeze(0).squeeze(-1).to(torch.float32)
    pred_depth = pred_depth * float(sample.normalization_scale or 1.0)
    return pred_extrinsics, pred_intrinsics, pred_depth


def _backproject_predicted_point_cloud(
    sample,
    pred_extrinsics: torch.Tensor,
    pred_intrinsics: torch.Tensor,
    pred_depth: torch.Tensor,
    pixel_stride: int,
    voxel_size: float,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    images = sample.images.detach().cpu().to(torch.float32)
    extrinsics = pred_extrinsics.detach().cpu().to(torch.float32)
    intrinsics = pred_intrinsics.detach().cpu().to(torch.float32)
    depth = pred_depth.detach().cpu().to(torch.float32)

    frame_points: List[np.ndarray] = []
    frame_colors: List[np.ndarray] = []
    for frame_idx in range(depth.shape[0]):
        frame_depth = depth[frame_idx]
        frame_rgb = images[frame_idx]
        if pixel_stride > 1:
            frame_depth = frame_depth[::pixel_stride, ::pixel_stride]
            frame_rgb = frame_rgb[:, ::pixel_stride, ::pixel_stride]

        valid = torch.isfinite(frame_depth) & (frame_depth > 0)
        if valid.sum().item() == 0:
            continue

        h, w = frame_depth.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )

        z = frame_depth[valid]
        fx = intrinsics[frame_idx, 0, 0]
        fy = intrinsics[frame_idx, 1, 1]
        cx = intrinsics[frame_idx, 0, 2]
        cy = intrinsics[frame_idx, 1, 2]
        x = (xs[valid] - cx) * z / fx
        y = (ys[valid] - cy) * z / fy
        cam_points = torch.stack([x, y, z], dim=-1)

        rotation = extrinsics[frame_idx, :3, :3]
        translation = extrinsics[frame_idx, :3, 3]
        world_points = (cam_points - translation.unsqueeze(0)) @ rotation

        colors = frame_rgb.permute(1, 2, 0)[valid]
        frame_points.append(world_points.numpy().astype(np.float32, copy=False))
        frame_colors.append(np.clip(colors.numpy(), 0.0, 1.0).astype(np.float32, copy=False))

    if not frame_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    points = np.concatenate(frame_points, axis=0)
    colors = np.concatenate(frame_colors, axis=0)
    points, colors = _voxel_downsample(points, colors, voxel_size=voxel_size)
    points, colors = _subsample_points(points, colors, max_points=max_points, seed=seed)
    return points, colors


def _project_points(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_points = points_world @ extrinsic[:, :3].T + extrinsic[:, 3]
    depth = camera_points[:, 2]
    positive = depth > 1e-8
    z_safe = np.where(positive, depth, 1.0).astype(np.float32, copy=False)
    u = intrinsic[0, 0] * (camera_points[:, 0] / z_safe) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (camera_points[:, 1] / z_safe) + intrinsic[1, 2]
    return (
        u.astype(np.float32, copy=False),
        v.astype(np.float32, copy=False),
        depth.astype(np.float32, copy=False),
        positive,
    )


def _render_projected_point_cloud(
    points_world: np.ndarray,
    colors: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size_hw: Tuple[int, int],
    radius: int,
    background_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_size_hw
    canvas = np.ones((height, width, 3), dtype=np.float32)
    canvas[:] = np.asarray(background_rgb, dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    if points_world.shape[0] == 0:
        return canvas, depth_buffer

    u, v, depth, positive = _project_points(points_world, extrinsic, intrinsic)
    if not np.any(positive):
        return canvas, depth_buffer

    colors = colors[positive].astype(np.float32, copy=False)
    u = u[positive]
    v = v[positive]
    depth = depth[positive]

    x = np.rint(u).astype(np.int32)
    y = np.rint(v).astype(np.int32)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(in_bounds):
        return canvas, depth_buffer

    x = x[in_bounds]
    y = y[in_bounds]
    depth = depth[in_bounds]
    colors = colors[in_bounds].astype(np.float32, copy=False)

    order = np.argsort(depth)[::-1]
    x = x[order]
    y = y[order]
    depth = depth[order]
    colors = colors[order]

    for dx in range(-radius, radius + 1):
        xx = x + dx
        valid_x = (xx >= 0) & (xx < width)
        if not np.any(valid_x):
            continue
        for dy in range(-radius, radius + 1):
            yy = y + dy
            valid = valid_x & (yy >= 0) & (yy < height)
            if not np.any(valid):
                continue
            xx_valid = xx[valid]
            yy_valid = yy[valid]
            depth_valid = depth[valid]
            color_valid = colors[valid]

            closer = depth_valid < depth_buffer[yy_valid, xx_valid]
            if not np.any(closer):
                continue
            xx_valid = xx_valid[closer]
            yy_valid = yy_valid[closer]
            depth_valid = depth_valid[closer]
            color_valid = color_valid[closer]
            depth_buffer[yy_valid, xx_valid] = depth_valid
            canvas[yy_valid, xx_valid] = color_valid

    return canvas, depth_buffer


def _compute_pred_to_gt_errors(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_gt_points: int,
    seed: int,
) -> np.ndarray:
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    gt_points_sampled, _ = _subsample_points(gt_points, None, max_gt_points, seed=seed)
    pred_tensor = torch.from_numpy(pred_points).to(torch.float32)
    gt_tensor = torch.from_numpy(gt_points_sampled).to(torch.float32)
    distances = _chunked_nearest_distances(pred_tensor, gt_tensor, chunk_size=2048)
    return distances.cpu().numpy().astype(np.float32, copy=False)


def _colormap_errors(errors: np.ndarray, max_error: float) -> np.ndarray:
    if errors.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    clipped = np.clip(errors / max(max_error, 1e-8), 0.0, 1.0)
    colors = plt.get_cmap("turbo")(clipped)[:, :3]
    return colors.astype(np.float32, copy=False)


def _crop_box_from_mask(mask: np.ndarray, padding_ratio: float) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    pad_x = int(round(box_w * padding_ratio))
    pad_y = int(round(box_h * padding_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(mask.shape[1], x1 + pad_x)
    y1 = min(mask.shape[0], y1 + pad_y)
    return x0, y0, x1, y1


def _crop_box_from_projected_points(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size_hw: Tuple[int, int],
    padding_ratio: float,
    trim_quantile: float = 0.01,
) -> Tuple[int, int, int, int]:
    height, width = image_size_hw
    u, v, depth, positive = _project_points(points_world, extrinsic, intrinsic)
    if not np.any(positive):
        return 0, 0, width, height

    u = u[positive]
    v = v[positive]
    depth = depth[positive]
    in_bounds = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    if not np.any(in_bounds):
        return 0, 0, width, height

    u = u[in_bounds]
    v = v[in_bounds]
    lower = trim_quantile
    upper = 1.0 - trim_quantile
    x0 = int(np.floor(np.quantile(u, lower)))
    x1 = int(np.ceil(np.quantile(u, upper))) + 1
    y0 = int(np.floor(np.quantile(v, lower)))
    y1 = int(np.ceil(np.quantile(v, upper))) + 1
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    pad_x = int(round(box_w * padding_ratio))
    pad_y = int(round(box_h * padding_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    return x0, y0, x1, y1


def _crop_image(image: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    return image[y0:y1, x0:x1]


def _save_png(image: np.ndarray, path: Path) -> None:
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def _save_point_cloud_ply(points: np.ndarray, path: Path, colors: Optional[np.ndarray] = None) -> None:
    import struct

    with path.open("wb") as f:
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if colors is not None:
            header.extend(
                [
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                ]
            )
        header.append("end_header")
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if colors is None:
            for point in points:
                f.write(struct.pack("<fff", float(point[0]), float(point[1]), float(point[2])))
        else:
            colors_u8 = (np.clip(colors, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            for point, color in zip(points, colors_u8):
                f.write(struct.pack("<fffBBB", float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2])))


def _make_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    output_path: Path,
    suptitle: Optional[str] = None,
) -> None:
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.8), dpi=180)
    if cols == 1:
        axes = [axes]
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _predict_for_model(
    model_spec: ModelSpec,
    sample,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    model, _ = load_model(model_spec, device)
    images = sample.images.to(device)
    masks = sample.masks.to(device) if (sample.masks is not None and model_spec.use_visual_hull_mask) else None
    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=amp_dtype):
            predictions = model(images, visual_hull_mask=masks)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {key: value.detach().cpu() for key, value in predictions.items() if torch.is_tensor(value)}


def main() -> None:
    args = parse_args()

    plan = load_plan(args.plan)
    dataset_spec = _select_dataset_spec(plan, args.dataset)
    model_specs = _select_model_specs(plan, args.models)
    scene_name = _normalize_scene_query(args.scene)
    sample = _find_sample(dataset_spec, scene_name)

    frame_names = list(sample.metadata.get("frame_names", []))
    frame_ids = list(sample.metadata.get("frame_ids", []))
    if args.frame_id is not None:
        query = str(args.frame_id)
        target_candidates = {query, f"{query}.jpg", f"{query}.png"}
        frame_index = None
        for idx, (frame_name, frame_id) in enumerate(zip(frame_names, frame_ids)):
            if str(frame_name) in target_candidates or str(frame_id) == query:
                frame_index = idx
                break
        if frame_index is None:
            raise ValueError(f"Frame id `{args.frame_id}` was not found in selected benchmark frames.")
    elif args.frame_index is not None:
        frame_index = int(args.frame_index)
    else:
        frame_index = len(frame_names) // 2
    if frame_index < 0 or frame_index >= sample.images.shape[0]:
        raise ValueError(
            f"frame_index {frame_index} is out of range for {sample.images.shape[0]} benchmark frames."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    amp_dtype = choose_amp_dtype(device)

    gt_points = sample.gt_point_cloud.detach().cpu().numpy().astype(np.float32, copy=False)
    gt_colors = _load_gt_colors_from_archive(dataset_spec, scene_name)
    bbox_diagonal = _points_bbox_diagonal(torch.from_numpy(gt_points))
    voxel_size = bbox_diagonal * float(args.voxel_size_ratio)
    max_error = bbox_diagonal * float(args.error_max_ratio)

    gt_render_points, gt_render_colors = _voxel_downsample(
        gt_points,
        gt_colors.astype(np.float32) / 255.0 if gt_colors is not None else None,
        voxel_size=voxel_size,
    )
    gt_render_points, gt_render_colors = _subsample_points(
        gt_render_points,
        gt_render_colors,
        max_points=args.max_render_points,
        seed=7,
    )
    if gt_render_colors is None:
        gt_render_colors = np.full((gt_render_points.shape[0], 3), 0.65, dtype=np.float32)

    target_image = _torch_to_image_rgb(sample.images[frame_index])
    target_extrinsic = sample.raw_extrinsics[frame_index].detach().cpu().numpy().astype(np.float32, copy=False)
    target_intrinsic = sample.raw_intrinsics[frame_index].detach().cpu().numpy().astype(np.float32, copy=False)
    image_size_hw = tuple(sample.images.shape[-2:])

    gt_render, gt_depth = _render_projected_point_cloud(
        points_world=gt_render_points,
        colors=gt_render_colors,
        extrinsic=target_extrinsic,
        intrinsic=target_intrinsic,
        image_size_hw=image_size_hw,
        radius=args.render_radius,
    )
    crop_box = _crop_box_from_projected_points(
        gt_render_points,
        target_extrinsic,
        target_intrinsic,
        image_size_hw=image_size_hw,
        padding_ratio=args.crop_padding,
    )
    if crop_box == (0, 0, image_size_hw[1], image_size_hw[0]):
        crop_box = _crop_box_from_mask(np.isfinite(gt_depth), padding_ratio=args.crop_padding)

    _save_png(target_image, output_dir / "input.png")
    _save_png(gt_render, output_dir / "gt_render.png")
    _save_png(_crop_image(target_image, crop_box), output_dir / "input_crop.png")
    _save_png(_crop_image(gt_render, crop_box), output_dir / "gt_render_crop.png")

    full_images = [target_image]
    full_titles = [f"Input\n{frame_names[frame_index] if frame_names else frame_index}"]
    crop_images = [_crop_image(target_image, crop_box)]
    crop_titles = ["Input Crop"]
    error_images: List[np.ndarray] = []
    error_titles: List[str] = []

    metadata: Dict[str, object] = {
        "plan": str(Path(args.plan).resolve()),
        "dataset": dataset_spec.name,
        "scene": scene_name,
        "sample_id": sample.sample_id,
        "frame_index": frame_index,
        "frame_name": frame_names[frame_index] if frame_names else None,
        "frame_id": frame_ids[frame_index] if frame_ids else None,
        "bbox_diagonal": bbox_diagonal,
        "voxel_size": voxel_size,
        "crop_box_xyxy": crop_box,
        "models": [],
    }

    if args.save_ply:
        _save_point_cloud_ply(gt_render_points, output_dir / "gt_points.ply", gt_render_colors)

    for model_spec in model_specs:
        predictions = _predict_for_model(model_spec, sample, device=device, amp_dtype=amp_dtype)
        if "depth" not in predictions or "pose_enc" not in predictions:
            raise ValueError(
                f"Model `{model_spec.name}` did not return both `depth` and `pose_enc`; "
                "this script expects benchmark camera + depth outputs."
            )

        pred_extrinsics, pred_intrinsics, pred_depth = _predict_camera_and_depth(sample, predictions)
        pred_points, pred_colors = _backproject_predicted_point_cloud(
            sample=sample,
            pred_extrinsics=pred_extrinsics,
            pred_intrinsics=pred_intrinsics,
            pred_depth=pred_depth,
            pixel_stride=max(1, int(args.pred_pixel_stride)),
            voxel_size=voxel_size,
            max_points=args.max_render_points,
            seed=_stable_seed(scene_name, model_spec.name, "pred_points"),
        )

        pred_render, pred_depth_buffer = _render_projected_point_cloud(
            points_world=pred_points,
            colors=pred_colors,
            extrinsic=target_extrinsic,
            intrinsic=target_intrinsic,
            image_size_hw=image_size_hw,
            radius=args.render_radius,
        )
        pred_errors = _compute_pred_to_gt_errors(
            pred_points=pred_points,
            gt_points=gt_points,
            max_gt_points=args.max_error_gt_points,
            seed=_stable_seed(scene_name, model_spec.name, "error"),
        )
        pred_error_colors = _colormap_errors(pred_errors, max_error=max_error)
        error_render, _ = _render_projected_point_cloud(
            points_world=pred_points,
            colors=pred_error_colors,
            extrinsic=target_extrinsic,
            intrinsic=target_intrinsic,
            image_size_hw=image_size_hw,
            radius=args.render_radius,
        )

        _save_png(pred_render, output_dir / f"{model_spec.name}_render.png")
        _save_png(error_render, output_dir / f"{model_spec.name}_error.png")
        _save_png(_crop_image(pred_render, crop_box), output_dir / f"{model_spec.name}_render_crop.png")
        _save_png(_crop_image(error_render, crop_box), output_dir / f"{model_spec.name}_error_crop.png")
        if args.save_ply:
            _save_point_cloud_ply(pred_points, output_dir / f"{model_spec.name}_points.ply", pred_colors)

        full_images.append(pred_render)
        full_titles.append(model_spec.name)
        crop_images.append(_crop_image(pred_render, crop_box))
        crop_titles.append(f"{model_spec.name} Crop")
        error_images.append(error_render)
        error_titles.append(f"{model_spec.name} Error")

        finite_errors = pred_errors[np.isfinite(pred_errors)]
        metadata["models"].append(
            {
                "name": model_spec.name,
                "model_spec": asdict(model_spec),
                "point_count": int(pred_points.shape[0]),
                "error_mean": float(finite_errors.mean()) if finite_errors.size > 0 else None,
                "error_median": float(np.median(finite_errors)) if finite_errors.size > 0 else None,
                "error_p95": float(np.percentile(finite_errors, 95.0)) if finite_errors.size > 0 else None,
                "error_clip_max": max_error,
                "pred_depth_min": float(pred_depth[pred_depth > 0].min().item()) if torch.any(pred_depth > 0) else None,
                "pred_depth_max": float(pred_depth[pred_depth > 0].max().item()) if torch.any(pred_depth > 0) else None,
            }
        )

    full_images.append(gt_render)
    full_titles.append("GT")
    crop_images.append(_crop_image(gt_render, crop_box))
    crop_titles.append("GT Crop")

    _make_grid(
        full_images,
        full_titles,
        output_dir / "comparison_grid.png",
        suptitle=f"{scene_name} | benchmark frame {frame_names[frame_index] if frame_names else frame_index}",
    )
    _make_grid(
        crop_images,
        crop_titles,
        output_dir / "comparison_crop_grid.png",
        suptitle=f"{scene_name} | crop",
    )
    if error_images:
        _make_grid(
            error_images,
            error_titles,
            output_dir / "error_grid.png",
            suptitle=f"{scene_name} | error to GT (clip={max_error:.6f})",
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
