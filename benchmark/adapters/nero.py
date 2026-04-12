from __future__ import annotations

import hashlib
import io
import itertools
import json
import pickle
import struct
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency for rasterized mesh masks
    cv2 = None

try:
    import trimesh
except ImportError:  # pragma: no cover - optional dependency for GlossyReal mesh GT only
    trimesh = None

from benchmark.adapters.base import BenchmarkDatasetAdapter, BenchmarkSample
from benchmark.plan import DatasetSpec


_NERO_GLOSSYREAL_MESH_ALIGNMENT_CACHE_VERSION = "v3"
_NERO_GLOSSYREAL_MESH_ALIGNMENT_MAP_FILENAME = "nero_glossyreal_mesh_alignment_v2.json"
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

_COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

_COLMAP_SINGLE_FOCAL_MODELS = {
    "SIMPLE_PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
}


def _to_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    bottom = torch.zeros((*extrinsics.shape[:-2], 1, 4), dtype=extrinsics.dtype, device=extrinsics.device)
    bottom[..., -1] = 1.0
    return torch.cat([extrinsics, bottom], dim=-2)


def _normalize_camera_extrinsics_and_points(
    extrinsics: torch.Tensor,
    cam_points: torch.Tensor,
    world_points: torch.Tensor,
    depths: torch.Tensor,
    point_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    extrinsics_h = _to_homogeneous(extrinsics)
    first_inv = torch.linalg.inv(extrinsics_h[:, :1]).expand(-1, extrinsics_h.shape[1], -1, -1)
    normalized_extrinsics_h = extrinsics_h @ first_inv

    first_rotation = extrinsics_h[:, 0, :3, :3]
    first_translation = extrinsics_h[:, 0, :3, 3]
    normalized_world_points = (
        world_points @ first_rotation.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
        + first_translation.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    )

    valid_mask_float = point_masks.to(torch.float32)
    dist = normalized_world_points.norm(dim=-1)
    avg_scale = (
        (dist * valid_mask_float).sum(dim=(1, 2, 3))
        / (valid_mask_float.sum(dim=(1, 2, 3)) + 1e-3)
    ).clamp(min=1e-6, max=1e6)

    normalized_world_points = normalized_world_points / avg_scale.view(-1, 1, 1, 1, 1)
    normalized_cam_points = cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    normalized_depths = depths / avg_scale.view(-1, 1, 1, 1)
    normalized_extrinsics = normalized_extrinsics_h[:, :, :3].clone()
    normalized_extrinsics[:, :, :3, 3] = normalized_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)

    return normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths


def _normalize_camera_extrinsics_from_reference_points(
    extrinsics: torch.Tensor,
    reference_world_points: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics with shape [N, 3, 4], got {tuple(extrinsics.shape)}")
    if reference_world_points.ndim != 2 or reference_world_points.shape[-1] != 3:
        raise ValueError(
            "Expected reference_world_points with shape [M, 3], "
            f"got {tuple(reference_world_points.shape)}"
        )
    if reference_world_points.shape[0] == 0:
        raise ValueError("Cannot normalize NeRO GlossyReal cameras without reference 3D points.")

    extrinsics_h = _to_homogeneous(extrinsics)
    first_inv = torch.linalg.inv(extrinsics_h[:1]).expand(extrinsics_h.shape[0], -1, -1)
    normalized_extrinsics_h = extrinsics_h @ first_inv

    first_rotation = extrinsics[0, :3, :3]
    first_translation = extrinsics[0, :3, 3]
    reference_points_first_cam = reference_world_points @ first_rotation.transpose(-1, -2) + first_translation
    avg_scale = float(reference_points_first_cam.norm(dim=-1).mean().clamp(min=1e-6, max=1e6).item())

    normalized_extrinsics = normalized_extrinsics_h[:, :3].clone()
    normalized_extrinsics[:, :3, 3] = normalized_extrinsics[:, :3, 3] / avg_scale
    return normalized_extrinsics, avg_scale


def _select_frame_ids(num_frames_total: int, num_frames_target: int, strategy: str) -> np.ndarray:
    if num_frames_target > num_frames_total:
        raise ValueError(
            f"Requested {num_frames_target} frames, but only {num_frames_total} are available."
        )

    if strategy == "first_n":
        return np.arange(num_frames_target, dtype=np.int64)
    if strategy == "evenly_spaced":
        ids = np.linspace(0, num_frames_total - 1, num_frames_target)
        return np.round(ids).astype(np.int64)

    raise ValueError(f"Unsupported NeRO frame selection strategy: {strategy}")


def _load_binary_point_ply(points_bytes: bytes) -> np.ndarray:
    file_obj = io.BytesIO(points_bytes)
    vertex_count = None
    vertex_properties: List[Tuple[str, str]] = []
    reading_vertex_props = False

    while True:
        line = file_obj.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading NeRO PLY header.")
        decoded = line.decode("ascii", errors="strict").strip()
        if decoded == "ply" or decoded == "format binary_little_endian 1.0" or decoded.startswith("comment "):
            continue
        if decoded.startswith("obj_info "):
            continue
        if decoded.startswith("element vertex "):
            vertex_count = int(decoded.split()[2])
            vertex_properties = []
            reading_vertex_props = True
            continue
        if decoded.startswith("element "):
            reading_vertex_props = False
            continue
        if decoded.startswith("property ") and reading_vertex_props:
            _, prop_type, prop_name = decoded.split()
            if prop_type not in _PLY_NUMPY_DTYPE_MAP:
                raise ValueError(f"Unsupported PLY property type `{prop_type}`")
            vertex_properties.append((prop_name, "<" + _PLY_NUMPY_DTYPE_MAP[prop_type]))
            continue
        if decoded == "end_header":
            break

    if vertex_count is None or not vertex_properties:
        raise ValueError("NeRO PLY is missing vertex metadata.")

    vertex_dtype = np.dtype(vertex_properties)
    vertices = np.frombuffer(points_bytes, dtype=vertex_dtype, count=vertex_count, offset=file_obj.tell())
    required_props = ("x", "y", "z")
    if any(prop not in vertices.dtype.names for prop in required_props):
        raise ValueError("NeRO PLY does not contain XYZ vertex coordinates.")
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1)
    return xyz.astype(np.float32, copy=False)


def _extract_zip_member(zip_path: Path, member_name: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name, "r") as src, output_path.open("wb") as dst:
            dst.write(src.read())
    return output_path


def _load_mesh_alignment_map(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "scenes" not in payload or not isinstance(payload["scenes"], dict):
        raise ValueError(f"Invalid NeRO GlossyReal mesh alignment map: {path}")
    return payload


def _stable_seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _downsample_points_numpy(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected point array with shape [N, 3], got {tuple(points.shape)}")
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def _principal_axes_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected point array with shape [N, 3], got {tuple(points.shape)}")
    if points.shape[0] < 3:
        raise ValueError("Need at least three points to estimate a principal-axis basis.")

    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    scales = np.sqrt(np.maximum(eigenvalues, 1e-12))
    return centroid, eigenvectors, scales


def _candidate_similarity_transforms_from_pca(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> List[np.ndarray]:
    source_centroid, source_basis, source_scales = _principal_axes_basis(source_points)
    target_centroid, target_basis, target_scales = _principal_axes_basis(target_points)

    transforms: List[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        perm_matrix = np.eye(3, dtype=np.float64)[:, perm]
        perm_scales = source_scales[list(perm)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            sign_matrix = np.diag(np.asarray(signs, dtype=np.float64))
            rotation = target_basis @ perm_matrix @ sign_matrix @ source_basis.T
            scale = float(np.mean(target_scales / perm_scales))
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = target_centroid - scale * (rotation @ source_centroid)
            transforms.append(transform)
    return transforms


def _apply_similarity_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _mesh_to_point_cloud_distance_summary(
    mesh_points: np.ndarray,
    transform: np.ndarray,
    target_points: np.ndarray,
) -> Dict[str, float]:
    transformed_points = _apply_similarity_transform(mesh_points, transform)
    distances, _ = cKDTree(target_points).query(transformed_points, k=1)
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p90": float(np.quantile(distances, 0.90)),
        "p95": float(np.quantile(distances, 0.95)),
    }


def _nearest_distance_summary(source_points: np.ndarray, target_points: np.ndarray) -> Dict[str, float]:
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        return {"mean": float("inf"), "median": float("inf"), "p95": float("inf")}
    distances, _ = cKDTree(target_points).query(source_points, k=1)
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.quantile(distances, 0.95)),
    }


def _load_trimesh_mesh(mesh_path: Path):
    if trimesh is None:
        raise ImportError(
            "NeRO GlossyReal mesh GT requires `trimesh`. Install it in the current environment "
            "before using reconstruction_gt_source=`mesh_gt_zip`."
        )
    mesh = trimesh.load(mesh_path, file_type="ply", force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangular mesh from {mesh_path}, got {type(mesh).__name__}.")
    if mesh.vertices is None or len(mesh.vertices) == 0 or mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh GT at {mesh_path} does not contain vertices/faces.")
    return mesh


def _export_aligned_mesh(mesh, transform: np.ndarray, output_path: Path) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(transform)
    aligned_mesh.export(output_path, file_type="ply")
    return np.asarray(aligned_mesh.vertices, dtype=np.float32)


def _resize_image_rgb(image_rgb: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = Image.fromarray(image_rgb, mode="RGB")
    resized = pil_image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized[0, 0].numpy().astype(np.float32, copy=False)


def _resize_depth(depth: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(depth.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return resized[0, 0].numpy().astype(np.float32, copy=False)


def _depth_to_cam_world_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth_map.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    z = depth_map.astype(np.float32, copy=False)
    point_mask = z > 0
    x = (xs - float(intrinsic[0, 2])) * z / float(intrinsic[0, 0])
    y = (ys - float(intrinsic[1, 2])) * z / float(intrinsic[1, 1])
    cam_points = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)

    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    world_points = (cam_points - translation.reshape(1, 1, 3)) @ rotation
    return cam_points, world_points.astype(np.float32, copy=False), point_mask.astype(bool, copy=False)


def _read_exact(file_obj: io.BytesIO, num_bytes: int) -> bytes:
    data = file_obj.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(f"Unexpected EOF while reading {num_bytes} bytes from COLMAP binary.")
    return data


def _read_null_terminated_string(file_obj: io.BytesIO) -> str:
    chunks = bytearray()
    while True:
        char = file_obj.read(1)
        if not char:
            raise ValueError("Unexpected EOF while reading null-terminated COLMAP string.")
        if char == b"\x00":
            return chunks.decode("utf-8")
        chunks.extend(char)


def _qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64, copy=False)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _build_colmap_intrinsic_matrix(camera: Dict[str, object]) -> np.ndarray:
    model_name = str(camera["model_name"])
    params = np.asarray(camera["params"], dtype=np.float32)

    if model_name in _COLMAP_SINGLE_FOCAL_MODELS:
        if params.shape[0] < 3:
            raise ValueError(f"COLMAP camera model `{model_name}` is missing focal/cx/cy parameters.")
        fx = float(params[0])
        fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    else:
        if params.shape[0] < 4:
            raise ValueError(f"COLMAP camera model `{model_name}` is missing fx/fy/cx/cy parameters.")
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])

    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def _read_colmap_cameras(camera_bytes: bytes) -> Dict[int, Dict[str, object]]:
    file_obj = io.BytesIO(camera_bytes)
    num_cameras = struct.unpack("<Q", _read_exact(file_obj, 8))[0]
    cameras: Dict[int, Dict[str, object]] = {}
    for _ in range(num_cameras):
        camera_id, model_id = struct.unpack("<ii", _read_exact(file_obj, 8))
        width, height = struct.unpack("<QQ", _read_exact(file_obj, 16))
        if model_id not in _COLMAP_CAMERA_MODELS:
            raise ValueError(f"Unsupported COLMAP camera model id `{model_id}` in NeRO GlossyReal archive.")
        model_name, num_params = _COLMAP_CAMERA_MODELS[model_id]
        params = np.frombuffer(_read_exact(file_obj, 8 * num_params), dtype="<f8", count=num_params).astype(
            np.float32,
            copy=False,
        )
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model_id": model_id,
            "model_name": model_name,
            "width": int(width),
            "height": int(height),
            "params": params,
        }
    return cameras


def _read_colmap_images(image_bytes: bytes) -> List[Dict[str, object]]:
    file_obj = io.BytesIO(image_bytes)
    num_images = struct.unpack("<Q", _read_exact(file_obj, 8))[0]
    point_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3D_id", "<i8")])
    images: List[Dict[str, object]] = []
    for _ in range(num_images):
        image_id = struct.unpack("<i", _read_exact(file_obj, 4))[0]
        qvec = np.frombuffer(_read_exact(file_obj, 32), dtype="<f8", count=4)
        tvec = np.frombuffer(_read_exact(file_obj, 24), dtype="<f8", count=3)
        camera_id = struct.unpack("<i", _read_exact(file_obj, 4))[0]
        name = _read_null_terminated_string(file_obj)
        num_points2d = struct.unpack("<Q", _read_exact(file_obj, 8))[0]
        if num_points2d > 0:
            points2d = np.frombuffer(
                _read_exact(file_obj, int(num_points2d) * point_dtype.itemsize),
                dtype=point_dtype,
                count=num_points2d,
            )
            point3d_ids = points2d["point3D_id"]
            point3d_ids = point3d_ids[point3d_ids >= 0].astype(np.int64, copy=False)
            point3d_ids = np.unique(point3d_ids)
        else:
            point3d_ids = np.empty((0,), dtype=np.int64)

        rotation = _qvec_to_rotation_matrix(qvec)
        extrinsic = np.concatenate([rotation, tvec.reshape(3, 1)], axis=1).astype(np.float32, copy=False)
        images.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name,
                "extrinsic": extrinsic,
                "point3d_ids": point3d_ids,
            }
        )
    return images


def _frame_sort_key(name: str) -> Tuple[int, int | str]:
    stem = Path(name).stem
    if stem.isdigit():
        return 0, int(stem)
    return 1, name


def _shared_sparse_point_pairs(
    frame_records: Sequence[Dict[str, object]],
    min_shared_points: int,
) -> List[List[int]]:
    pair_indices: List[List[int]] = []
    for i in range(len(frame_records)):
        points_i = np.asarray(frame_records[i]["point3d_ids"], dtype=np.int64)
        for j in range(i + 1, len(frame_records)):
            points_j = np.asarray(frame_records[j]["point3d_ids"], dtype=np.int64)
            if min_shared_points <= 0:
                pair_indices.append([i, j])
                continue
            shared_count = int(np.intersect1d(points_i, points_j, assume_unique=True).size)
            if shared_count >= min_shared_points:
                pair_indices.append([i, j])
    return pair_indices


def _project_object_points_to_mask(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: Tuple[int, int],
    point_radius: int,
    dilation_kernel_size: int,
    projection_trim_quantile: float,
) -> np.ndarray:
    height, width = image_hw
    if points_world.size == 0:
        return np.zeros((height, width), dtype=bool)

    cam_points = points_world @ extrinsic[:, :3].transpose() + extrinsic[:, 3]
    depth = cam_points[:, 2]
    valid = np.isfinite(depth) & (depth > 1e-8)
    if not np.any(valid):
        return np.zeros((height, width), dtype=bool)

    cam_points = cam_points[valid]
    depth = depth[valid]
    u = intrinsic[0, 0] * (cam_points[:, 0] / depth) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (cam_points[:, 1] / depth) + intrinsic[1, 2]
    finite = np.isfinite(u) & np.isfinite(v)
    if not np.any(finite):
        return np.zeros((height, width), dtype=bool)
    u = u[finite]
    v = v[finite]
    x = np.rint(u).astype(np.int32)
    y = np.rint(v).astype(np.int32)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(in_bounds):
        return np.zeros((height, width), dtype=bool)

    x = x[in_bounds]
    y = y[in_bounds]
    trim_q = float(projection_trim_quantile)
    if 0.0 < trim_q < 0.5 and x.size >= 16:
        x_min = int(np.floor(np.quantile(x, trim_q)))
        x_max = int(np.ceil(np.quantile(x, 1.0 - trim_q)))
        y_min = int(np.floor(np.quantile(y, trim_q)))
        y_max = int(np.ceil(np.quantile(y, 1.0 - trim_q)))
        trimmed = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        if np.any(trimmed):
            x = x[trimmed]
            y = y[trimmed]
    mask = np.zeros((height, width), dtype=np.float32)
    for dx in range(-point_radius, point_radius + 1):
        xx = x + dx
        valid_x = (xx >= 0) & (xx < width)
        if not np.any(valid_x):
            continue
        for dy in range(-point_radius, point_radius + 1):
            yy = y + dy
            valid_xy = valid_x & (yy >= 0) & (yy < height)
            if not np.any(valid_xy):
                continue
            mask[yy[valid_xy], xx[valid_xy]] = 1.0

    if dilation_kernel_size > 1:
        kernel_size = int(dilation_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask_tensor = torch.from_numpy(mask)[None, None]
        mask = (
            F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)[0, 0]
            .numpy()
            .astype(np.float32, copy=False)
        )
    return mask > 0.5


def _simplify_mesh_vertex_clustering(
    vertices: np.ndarray,
    faces: np.ndarray,
    cluster_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if cluster_size <= 0:
        return vertices, faces

    grid = np.floor(vertices / cluster_size).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)

    simplified_vertices = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(simplified_vertices, inverse, vertices)
    simplified_vertices /= counts[:, None]

    simplified_faces = inverse[faces]
    valid = (
        (simplified_faces[:, 0] != simplified_faces[:, 1])
        & (simplified_faces[:, 1] != simplified_faces[:, 2])
        & (simplified_faces[:, 0] != simplified_faces[:, 2])
    )
    simplified_faces = simplified_faces[valid]
    if simplified_faces.shape[0] == 0:
        return simplified_vertices, simplified_faces

    dedup_key = np.sort(simplified_faces, axis=1)
    _, unique_indices = np.unique(dedup_key, axis=0, return_index=True)
    simplified_faces = simplified_faces[np.sort(unique_indices)]
    return simplified_vertices, simplified_faces


def _project_world_vertices(
    vertices_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cam_points = vertices_world @ extrinsic[:, :3].transpose() + extrinsic[:, 3]
    depth = cam_points[:, 2]
    u = intrinsic[0, 0] * (cam_points[:, 0] / np.maximum(depth, 1e-8)) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (cam_points[:, 1] / np.maximum(depth, 1e-8)) + intrinsic[1, 2]
    return np.stack([u, v], axis=-1), cam_points


def _rasterize_mesh_mask(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    if cv2 is None:
        raise ImportError(
            "Rasterized NeRO GlossyReal mesh masks require `opencv-python` / `cv2` in the current environment."
        )

    height, width = image_hw
    uv, cam_points = _project_world_vertices(vertices_world, extrinsic, intrinsic)
    tri_cam = cam_points[faces]
    tri_uv = uv[faces]

    valid_depth = np.all(tri_cam[:, :, 2] > 1e-6, axis=1)
    tri_uv = tri_uv[valid_depth]
    if tri_uv.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)

    min_xy = tri_uv.min(axis=1)
    max_xy = tri_uv.max(axis=1)
    in_view = (
        (max_xy[:, 0] >= 0)
        & (max_xy[:, 1] >= 0)
        & (min_xy[:, 0] < width)
        & (min_xy[:, 1] < height)
    )
    tri_uv = tri_uv[in_view]
    if tri_uv.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)

    polygons = np.rint(tri_uv).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    chunk_size = 5000
    for start in range(0, polygons.shape[0], chunk_size):
        chunk = polygons[start : start + chunk_size]
        cv2.fillPoly(mask, chunk, color=255, lineType=cv2.LINE_8)
    return mask > 0


class NeROGlossySyntheticAdapter(BenchmarkDatasetAdapter):
    def __init__(self, spec: DatasetSpec):
        super().__init__(spec)
        config = spec.config
        archive_path = Path(config["data_path"]).expanduser()
        if not archive_path.exists():
            raise FileNotFoundError(f"NeRO benchmark archive not found: {archive_path}")

        self.archive_path = archive_path
        self.subset = str(config.get("subset", "GlossySynthetic"))
        self.num_frames = int(config.get("num_frames", 16))
        self.frame_selection = str(config.get("frame_selection", "evenly_spaced"))
        self.image_size = int(config.get("img_size", 518))
        self.depth_scale = float(config.get("depth_scale", 5000.0))
        self.depth_invalid_value = int(config.get("depth_invalid_value", 65535))
        self.max_scenes = config.get("max_scenes")
        self.gt_point_sample_points = int(config.get("gt_point_sample_points", 20000))
        self.pred_point_sample_points = int(config.get("pred_point_sample_points", 20000))
        self.covisibility_min_overlap_ratio = float(config.get("covisibility_min_overlap_ratio", 0.05))
        self.covisibility_min_visible_points = int(config.get("covisibility_min_visible_points", 256))
        self.covisibility_depth_abs_tol = float(config.get("covisibility_depth_abs_tol", 0.01))
        self.covisibility_depth_rel_tol = float(config.get("covisibility_depth_rel_tol", 0.01))
        self.small_translation_epsilon = float(config.get("small_translation_epsilon", 1e-4))
        self.tsdf_resolution = int(config.get("tsdf_resolution", 256))
        self.tsdf_sdf_trunc_factor = float(config.get("tsdf_sdf_trunc_factor", 4.0))

        requested_scene_names = config.get("scene_names")
        self.scene_name_filter = set(requested_scene_names) if requested_scene_names is not None else None
        self.scenes = self._index_archive()

    def _index_archive(self) -> List[Dict[str, object]]:
        scene_assets: Dict[str, Dict[str, object]] = defaultdict(lambda: {"frames": set(), "gt_points_member": None})
        with tarfile.open(self.archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) != 3 or parts[0] != self.subset:
                    continue
                scene_name, filename = parts[1], parts[2]
                if self.scene_name_filter is not None and scene_name not in self.scene_name_filter:
                    continue
                if filename == "eval_pts.ply":
                    scene_assets[scene_name]["gt_points_member"] = member.name
                    continue
                if filename.endswith("-camera.pkl"):
                    frame_id = int(filename[:-11])
                    scene_assets[scene_name]["frames"].add(frame_id)
                    continue
                if filename.endswith("-depth.png"):
                    frame_id = int(filename[:-10])
                    scene_assets[scene_name]["frames"].add(frame_id)
                    continue
                if filename.endswith(".png") and filename[:-4].isdigit():
                    frame_id = int(filename[:-4])
                    scene_assets[scene_name]["frames"].add(frame_id)

        scenes: List[Dict[str, object]] = []
        for scene_name in sorted(scene_assets):
            frame_ids = sorted(int(frame_id) for frame_id in scene_assets[scene_name]["frames"])
            gt_points_member = scene_assets[scene_name]["gt_points_member"]
            if not frame_ids or gt_points_member is None:
                continue
            scenes.append(
                {
                    "name": scene_name,
                    "frame_ids": frame_ids,
                    "gt_points_member": gt_points_member,
                }
            )
        if self.max_scenes is not None:
            scenes = scenes[: int(self.max_scenes)]
        return scenes

    def _load_scene_gt_points(self, tf: tarfile.TarFile, member_name: str) -> torch.Tensor:
        points_bytes = tf.extractfile(member_name).read()
        return torch.from_numpy(_load_binary_point_ply(points_bytes))

    def _load_frame(
        self,
        tf: tarfile.TarFile,
        scene_name: str,
        frame_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_member = f"{self.subset}/{scene_name}/{frame_id}.png"
        depth_member = f"{self.subset}/{scene_name}/{frame_id}-depth.png"
        camera_member = f"{self.subset}/{scene_name}/{frame_id}-camera.pkl"

        image_rgba = np.asarray(Image.open(io.BytesIO(tf.extractfile(image_member).read())))
        depth_raw = np.asarray(Image.open(io.BytesIO(tf.extractfile(depth_member).read())))
        extrinsic, intrinsic = pickle.loads(tf.extractfile(camera_member).read())

        if image_rgba.ndim != 3 or image_rgba.shape[2] < 3:
            raise ValueError(f"Unexpected NeRO image format for {image_member}: {image_rgba.shape}")
        rgb = image_rgba[..., :3].astype(np.uint8, copy=False)
        if image_rgba.shape[2] >= 4:
            mask = (image_rgba[..., 3] > 0).astype(np.float32, copy=False)
        else:
            mask = np.ones(rgb.shape[:2], dtype=np.float32)

        depth = depth_raw.astype(np.float32, copy=False) / self.depth_scale
        depth[depth_raw == self.depth_invalid_value] = 0.0
        depth[mask < 0.5] = 0.0

        if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
            src_h, src_w = rgb.shape[:2]
            scale_x = self.image_size / float(src_w)
            scale_y = self.image_size / float(src_h)
            rgb = _resize_image_rgb(rgb, self.image_size)
            mask = _resize_mask(mask, self.image_size)
            depth = _resize_depth(depth, self.image_size)
            intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
            intrinsic[0, 0] *= scale_x
            intrinsic[1, 1] *= scale_y
            intrinsic[0, 2] *= scale_x
            intrinsic[1, 2] *= scale_y

        cam_points, world_points, point_mask = _depth_to_cam_world_points(
            depth_map=depth,
            extrinsic=np.asarray(extrinsic, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
        )
        point_mask = point_mask & (mask > 0.5)
        depth[~point_mask] = 0.0

        return (
            rgb,
            mask.astype(np.float32, copy=False),
            depth.astype(np.float32, copy=False),
            np.asarray(extrinsic, dtype=np.float32),
            np.asarray(intrinsic, dtype=np.float32),
            cam_points.astype(np.float32, copy=False),
            world_points.astype(np.float32, copy=False),
            point_mask.astype(bool, copy=False),
        )

    def iter_samples(self) -> Iterator[BenchmarkSample]:
        with tarfile.open(self.archive_path, "r:*") as tf:
            for scene in self.scenes:
                available_frame_ids = np.asarray(scene["frame_ids"], dtype=np.int64)
                selected_positions = _select_frame_ids(
                    num_frames_total=len(available_frame_ids),
                    num_frames_target=self.num_frames,
                    strategy=self.frame_selection,
                )
                frame_ids = available_frame_ids[selected_positions]

                images = []
                masks = []
                depths = []
                extrinsics = []
                intrinsics = []
                cam_points = []
                world_points = []
                point_masks = []
                for frame_id in frame_ids.tolist():
                    (
                        image,
                        mask,
                        depth,
                        extrinsic,
                        intrinsic,
                        cam_point,
                        world_point,
                        point_mask,
                    ) = self._load_frame(tf, scene_name=str(scene["name"]), frame_id=int(frame_id))
                    images.append(image)
                    masks.append(mask)
                    depths.append(depth)
                    extrinsics.append(extrinsic)
                    intrinsics.append(intrinsic)
                    cam_points.append(cam_point)
                    world_points.append(world_point)
                    point_masks.append(point_mask)

                image_tensor = torch.from_numpy(np.stack(images).astype(np.float32)).permute(0, 3, 1, 2) / 255.0
                mask_tensor = torch.from_numpy(np.stack(masks).astype(np.float32))
                depth_tensor = torch.from_numpy(np.stack(depths).astype(np.float32))
                extrinsic_tensor = torch.from_numpy(np.stack(extrinsics).astype(np.float32))
                intrinsic_tensor = torch.from_numpy(np.stack(intrinsics).astype(np.float32))
                cam_point_tensor = torch.from_numpy(np.stack(cam_points).astype(np.float32))
                world_point_tensor = torch.from_numpy(np.stack(world_points).astype(np.float32))
                point_mask_tensor = torch.from_numpy(np.stack(point_masks))

                first_extrinsic = extrinsic_tensor[0]
                first_rotation = first_extrinsic[:3, :3]
                first_translation = first_extrinsic[:3, 3]
                world_points_first_cam = (
                    world_point_tensor @ first_rotation.transpose(-1, -2).unsqueeze(0).unsqueeze(0)
                ) + first_translation.view(1, 1, 1, 3)
                world_point_dist = world_points_first_cam.norm(dim=-1)
                valid_mask_float = point_mask_tensor.to(torch.float32)
                avg_scale = float(
                    (
                        (world_point_dist * valid_mask_float).sum()
                        / (valid_mask_float.sum() + 1e-3)
                    ).clamp(min=1e-6, max=1e6).item()
                )

                normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = (
                    _normalize_camera_extrinsics_and_points(
                        extrinsics=extrinsic_tensor.unsqueeze(0),
                        cam_points=cam_point_tensor.unsqueeze(0),
                        world_points=world_point_tensor.unsqueeze(0),
                        depths=depth_tensor.unsqueeze(0),
                        point_masks=point_mask_tensor.unsqueeze(0),
                    )
                )

                sample_id = f"{self.subset}/{scene['name']}"
                gt_points = self._load_scene_gt_points(tf, str(scene["gt_points_member"]))
                yield BenchmarkSample(
                    sample_id=sample_id,
                    images=image_tensor,
                    masks=mask_tensor,
                    depths=normalized_depths.squeeze(0),
                    cam_points=normalized_cam_points.squeeze(0),
                    world_points=normalized_world_points.squeeze(0),
                    point_masks=point_mask_tensor,
                    extrinsics=normalized_extrinsics.squeeze(0),
                    intrinsics=intrinsic_tensor,
                    raw_depths=depth_tensor,
                    raw_extrinsics=extrinsic_tensor,
                    raw_intrinsics=intrinsic_tensor,
                    normalization_scale=avg_scale,
                    normalization_reference_extrinsic=first_extrinsic,
                    gt_point_cloud=gt_points,
                    metadata={
                        "seq_name": str(scene["name"]),
                        "frame_ids": frame_ids.tolist(),
                    },
                    protocol={
                        "pred_point_sample_points": self.pred_point_sample_points,
                        "gt_point_sample_points": self.gt_point_sample_points,
                        "covisibility_min_overlap_ratio": self.covisibility_min_overlap_ratio,
                        "covisibility_min_visible_points": self.covisibility_min_visible_points,
                        "covisibility_depth_abs_tol": self.covisibility_depth_abs_tol,
                        "covisibility_depth_rel_tol": self.covisibility_depth_rel_tol,
                        "small_translation_epsilon": self.small_translation_epsilon,
                        "tsdf_resolution": self.tsdf_resolution,
                        "tsdf_sdf_trunc_factor": self.tsdf_sdf_trunc_factor,
                    },
                )


class NeROGlossyRealAdapter(BenchmarkDatasetAdapter):
    def __init__(self, spec: DatasetSpec):
        super().__init__(spec)
        config = spec.config
        archive_path = Path(config["data_path"]).expanduser()
        if not archive_path.exists():
            raise FileNotFoundError(f"NeRO benchmark archive not found: {archive_path}")

        self.archive_path = archive_path
        self.subset = str(config.get("subset", "GlossyReal"))
        self.num_frames = int(config.get("num_frames", 16))
        self.frame_selection = str(config.get("frame_selection", "evenly_spaced"))
        self.image_size = int(config.get("img_size", 518))
        self.reconstruction_gt_source = str(config.get("reconstruction_gt_source", "mesh_gt_zip"))
        mesh_gt_zip_path = config.get("mesh_gt_zip_path")
        if mesh_gt_zip_path is None:
            mesh_gt_zip_path = archive_path.parent / "glossy-real-meshes-gt.zip"
        self.mesh_gt_zip_path = Path(mesh_gt_zip_path).expanduser()
        self.mesh_extract_root = Path(
            config.get(
                "mesh_extract_root",
                Path(tempfile.gettempdir()) / "vggt_benchmark_nero_glossyreal_meshes",
            )
        ).expanduser()
        mesh_alignment_map_path = config.get("mesh_alignment_map_path")
        if mesh_alignment_map_path is None:
            mesh_alignment_map_path = Path(__file__).with_name(_NERO_GLOSSYREAL_MESH_ALIGNMENT_MAP_FILENAME)
        self.mesh_alignment_map_path = Path(mesh_alignment_map_path).expanduser()
        self.mesh_alignment_map = (
            _load_mesh_alignment_map(self.mesh_alignment_map_path)
            if self.mesh_alignment_map_path.exists()
            else None
        )
        self.mesh_alignment_source_points = int(config.get("mesh_alignment_source_points", 12000))
        self.mesh_alignment_target_points = int(config.get("mesh_alignment_target_points", 12000))
        self.mesh_alignment_score_points = int(config.get("mesh_alignment_score_points", 50000))
        self.mesh_alignment_candidate_topk = int(config.get("mesh_alignment_candidate_topk", 6))
        self.mesh_alignment_icp_iterations = int(config.get("mesh_alignment_icp_iterations", 20))
        self.mesh_mask_cluster_ratio = float(config.get("mesh_mask_cluster_ratio", 1.0 / 250.0))
        self.max_scenes = config.get("max_scenes")
        self.gt_point_sample_points = int(config.get("gt_point_sample_points", 20000))
        self.pred_point_sample_points = int(config.get("pred_point_sample_points", 20000))
        self.object_mask_point_radius = int(config.get("object_mask_point_radius", 2))
        self.object_mask_dilation_kernel_size = int(config.get("object_mask_dilation_kernel_size", 7))
        self.object_mask_projection_trim_quantile = float(config.get("object_mask_projection_trim_quantile", 0.05))
        self.camera_min_shared_sparse_points = int(config.get("camera_min_shared_sparse_points", 16))
        self.small_translation_epsilon = float(config.get("small_translation_epsilon", 1e-4))
        self.tsdf_resolution = int(config.get("tsdf_resolution", 256))
        self.tsdf_sdf_trunc_factor = float(config.get("tsdf_sdf_trunc_factor", 4.0))

        requested_scene_names = config.get("scene_names")
        self.scene_name_filter = set(requested_scene_names) if requested_scene_names is not None else None
        self.scenes = self._index_archive()

    def _index_archive(self) -> List[Dict[str, object]]:
        scene_assets: Dict[str, Dict[str, object]] = defaultdict(
            lambda: {
                "image_members": {},
                "cameras_member": None,
                "images_member": None,
                "gt_points_member": None,
                "colmap_points_member": None,
                "object_points_member": None,
                "mesh_member": None,
            }
        )
        with tarfile.open(self.archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) < 3 or parts[0] != self.subset:
                    continue
                scene_name = parts[1]
                if self.scene_name_filter is not None and scene_name not in self.scene_name_filter:
                    continue
                relative_parts = parts[2:]

                if relative_parts[:1] == ["images"] and len(relative_parts) == 2:
                    filename = relative_parts[1]
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        scene_assets[scene_name]["image_members"][filename] = member.name
                    continue
                if relative_parts == ["colmap", "sparse", "0", "cameras.bin"]:
                    scene_assets[scene_name]["cameras_member"] = member.name
                    continue
                if relative_parts == ["colmap", "sparse", "0", "images.bin"]:
                    scene_assets[scene_name]["images_member"] = member.name
                    continue
                if relative_parts == ["colmap", "points.ply"]:
                    scene_assets[scene_name]["colmap_points_member"] = member.name
                    continue
                if relative_parts == ["object_point_cloud.ply"]:
                    scene_assets[scene_name]["object_points_member"] = member.name

            if self.mesh_gt_zip_path.exists():
                with zipfile.ZipFile(self.mesh_gt_zip_path, "r") as zf:
                    zip_names = set(zf.namelist())
                for scene_name in scene_assets:
                    mesh_member = f"{scene_name}-align.ply"
                    if mesh_member in zip_names:
                        scene_assets[scene_name]["mesh_member"] = mesh_member

            scenes: List[Dict[str, object]] = []
            for scene_name in sorted(scene_assets):
                assets = scene_assets[scene_name]
                if self.reconstruction_gt_source == "mesh_gt_zip":
                    gt_points_member = assets["mesh_member"]
                elif self.reconstruction_gt_source == "object_point_cloud":
                    gt_points_member = assets["object_points_member"]
                elif self.reconstruction_gt_source == "colmap_points":
                    gt_points_member = assets["colmap_points_member"]
                else:
                    raise ValueError(
                        "Unsupported GlossyReal reconstruction_gt_source "
                        f"`{self.reconstruction_gt_source}`. Expected `mesh_gt_zip`, `object_point_cloud` or `colmap_points`."
                    )
                if (
                    not assets["image_members"]
                    or assets["cameras_member"] is None
                    or assets["images_member"] is None
                    or gt_points_member is None
                ):
                    continue

                cameras = _read_colmap_cameras(tf.extractfile(str(assets["cameras_member"])).read())
                image_records = _read_colmap_images(tf.extractfile(str(assets["images_member"])).read())

                indexed_frames: List[Dict[str, object]] = []
                available_image_members = dict(assets["image_members"])
                for image_record in sorted(image_records, key=lambda item: _frame_sort_key(str(item["name"]))):
                    image_name = str(image_record["name"])
                    if image_name not in available_image_members:
                        continue
                    camera_id = int(image_record["camera_id"])
                    if camera_id not in cameras:
                        continue
                    indexed_frames.append(
                        {
                            "image_name": image_name,
                            "image_member": available_image_members[image_name],
                            "frame_id": int(Path(image_name).stem)
                            if Path(image_name).stem.isdigit()
                            else Path(image_name).stem,
                            "extrinsic": np.asarray(image_record["extrinsic"], dtype=np.float32),
                            "intrinsic": _build_colmap_intrinsic_matrix(cameras[camera_id]),
                            "point3d_ids": np.asarray(image_record["point3d_ids"], dtype=np.int64),
                        }
                    )

                if not indexed_frames:
                    continue
                scenes.append(
                    {
                        "name": scene_name,
                        "frames": indexed_frames,
                        "gt_points_member": gt_points_member,
                        "object_points_member": assets["object_points_member"],
                        "mesh_member": assets["mesh_member"],
                    }
                )

        if self.max_scenes is not None:
            scenes = scenes[: int(self.max_scenes)]
        return scenes

    def _estimate_mesh_alignment_transform(
        self,
        scene_name: str,
        mesh_vertices: np.ndarray,
        object_points_world: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        seed = _stable_seed_from_text(scene_name)
        source_points = _downsample_points_numpy(mesh_vertices, self.mesh_alignment_source_points, seed)
        target_points = _downsample_points_numpy(object_points_world, self.mesh_alignment_target_points, seed + 1)
        score_points = _downsample_points_numpy(mesh_vertices, self.mesh_alignment_score_points, seed + 2)

        candidates = _candidate_similarity_transforms_from_pca(
            source_points=source_points.astype(np.float64, copy=False),
            target_points=target_points.astype(np.float64, copy=False),
        )
        candidate_scores = [
            (
                _mesh_to_point_cloud_distance_summary(
                    mesh_points=score_points.astype(np.float64, copy=False),
                    transform=candidate,
                    target_points=target_points.astype(np.float64, copy=False),
                ),
                candidate,
            )
            for candidate in candidates
        ]
        candidate_scores.sort(
            key=lambda item: (
                item[0]["median"],
                item[0]["p90"],
                item[0]["mean"],
            )
        )
        best_stats, best_transform = candidate_scores[0]

        topk = max(1, min(self.mesh_alignment_candidate_topk, len(candidate_scores)))
        for _, candidate in candidate_scores[:topk]:
            refined_transform, _, _ = trimesh.registration.icp(
                source_points.astype(np.float64, copy=False),
                target_points.astype(np.float64, copy=False),
                initial=candidate,
                threshold=1e-6,
                max_iterations=self.mesh_alignment_icp_iterations,
                reflection=True,
                scale=True,
                translation=True,
            )
            refined_stats = _mesh_to_point_cloud_distance_summary(
                mesh_points=score_points,
                transform=refined_transform,
                target_points=target_points.astype(np.float64, copy=False),
            )
            if (
                refined_stats["median"],
                refined_stats["p90"],
                refined_stats["mean"],
            ) < (
                best_stats["median"],
                best_stats["p90"],
                best_stats["mean"],
            ):
                best_stats = refined_stats
                best_transform = refined_transform

        coarse_nn = _nearest_distance_summary(
            _apply_similarity_transform(source_points.astype(np.float64, copy=False), best_transform),
            target_points.astype(np.float64, copy=False),
        )
        return best_transform.astype(np.float64, copy=False), {
            "mesh_to_object_mean": best_stats["mean"],
            "mesh_to_object_median": best_stats["median"],
            "mesh_to_object_p90": best_stats["p90"],
            "mesh_to_object_p95": best_stats["p95"],
            "nn_mean": coarse_nn["mean"],
            "nn_median": coarse_nn["median"],
            "nn_p95": coarse_nn["p95"],
        }

    def _prepare_world_aligned_mesh(
        self,
        scene_name: str,
        mesh_member: str,
        object_points_world: np.ndarray,
    ) -> Tuple[Path, np.ndarray, Dict[str, object]]:
        raw_mesh_path = _extract_zip_member(
            zip_path=self.mesh_gt_zip_path,
            member_name=mesh_member,
            output_path=self.mesh_extract_root / "raw" / mesh_member,
        )
        aligned_mesh_path = (
            self.mesh_extract_root
            / "aligned"
            / _NERO_GLOSSYREAL_MESH_ALIGNMENT_CACHE_VERSION
            / f"{scene_name}-colmap-world.ply"
        )
        if aligned_mesh_path.exists():
            aligned_vertices = _load_binary_point_ply(aligned_mesh_path.read_bytes())
            return aligned_mesh_path, aligned_vertices, {"cache_hit": True}

        mesh = _load_trimesh_mesh(raw_mesh_path)
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)

        transform = None
        stats: Dict[str, object]
        if self.mesh_alignment_map is not None:
            scene_entry = self.mesh_alignment_map["scenes"].get(scene_name)
            if scene_entry is not None:
                transform = np.asarray(scene_entry["transform"], dtype=np.float64)
                stats = dict(scene_entry)
                stats.pop("transform", None)
                stats.update({
                    "cache_hit": False,
                    "alignment_source": "precomputed_map",
                    "alignment_map_path": str(self.mesh_alignment_map_path),
                })
            else:
                stats = {}
        else:
            stats = {}

        if transform is None:
            transform, estimated_stats = self._estimate_mesh_alignment_transform(
                scene_name=scene_name,
                mesh_vertices=mesh_vertices,
                object_points_world=object_points_world,
            )
            stats.update(estimated_stats)
            stats["alignment_source"] = "runtime_3d_registration"

        aligned_vertices = _export_aligned_mesh(mesh, transform, aligned_mesh_path)
        stats["cache_hit"] = False
        return aligned_mesh_path, aligned_vertices, stats

    def _load_scene_gt_points(self, tf: tarfile.TarFile, member_name: str) -> torch.Tensor:
        points_bytes = tf.extractfile(member_name).read()
        return torch.from_numpy(_load_binary_point_ply(points_bytes))

    def _load_frame_image(
        self,
        tf: tarfile.TarFile,
        image_member: str,
        intrinsic: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_rgb = np.asarray(Image.open(io.BytesIO(tf.extractfile(image_member).read())).convert("RGB"))
        resized_intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
        if image_rgb.shape[0] != self.image_size or image_rgb.shape[1] != self.image_size:
            src_h, src_w = image_rgb.shape[:2]
            scale_x = self.image_size / float(src_w)
            scale_y = self.image_size / float(src_h)
            image_rgb = _resize_image_rgb(image_rgb, self.image_size)
            resized_intrinsic[0, 0] *= scale_x
            resized_intrinsic[1, 1] *= scale_y
            resized_intrinsic[0, 2] *= scale_x
            resized_intrinsic[1, 2] *= scale_y
        return image_rgb.astype(np.uint8, copy=False), resized_intrinsic

    def iter_samples(self) -> Iterator[BenchmarkSample]:
        with tarfile.open(self.archive_path, "r:*") as tf:
            for scene in self.scenes:
                available_frames = list(scene["frames"])
                selected_positions = _select_frame_ids(
                    num_frames_total=len(available_frames),
                    num_frames_target=self.num_frames,
                    strategy=self.frame_selection,
                )
                selected_frames = [available_frames[int(idx)] for idx in selected_positions.tolist()]

                images = []
                intrinsics = []
                extrinsics = []
                frame_ids = []
                frame_names = []
                for frame in selected_frames:
                    image, intrinsic = self._load_frame_image(
                        tf=tf,
                        image_member=str(frame["image_member"]),
                        intrinsic=np.asarray(frame["intrinsic"], dtype=np.float32),
                    )
                    images.append(image)
                    intrinsics.append(intrinsic)
                    extrinsics.append(np.asarray(frame["extrinsic"], dtype=np.float32))
                    frame_ids.append(frame["frame_id"])
                    frame_names.append(str(frame["image_name"]))

                image_tensor = torch.from_numpy(np.stack(images).astype(np.float32)).permute(0, 3, 1, 2) / 255.0
                extrinsic_tensor = torch.from_numpy(np.stack(extrinsics).astype(np.float32))
                intrinsic_tensor = torch.from_numpy(np.stack(intrinsics).astype(np.float32))
                object_points_member = scene.get("object_points_member")
                object_points = (
                    self._load_scene_gt_points(tf, str(object_points_member))
                    if object_points_member is not None
                    else None
                )
                object_points_np = (
                    object_points.cpu().numpy().astype(np.float32, copy=False)
                    if object_points is not None
                    else None
                )

                gt_mesh_path = None
                raster_mesh_vertices = None
                raster_mesh_faces = None
                if self.reconstruction_gt_source == "mesh_gt_zip":
                    mesh_member = scene.get("mesh_member")
                    if mesh_member is None:
                        raise FileNotFoundError(
                            f"Mesh GT for scene `{scene['name']}` was not found in {self.mesh_gt_zip_path}."
                        )
                    if object_points_np is None:
                        raise ValueError(
                            f"Scene `{scene['name']}` is missing object_point_cloud.ply, "
                            "which is required to align mesh GT into COLMAP world coordinates."
                        )
                    mesh_path, aligned_vertices, alignment_stats = self._prepare_world_aligned_mesh(
                        scene_name=str(scene["name"]),
                        mesh_member=str(mesh_member),
                        object_points_world=object_points_np,
                    )
                    gt_mesh_path = str(mesh_path)
                    gt_points = torch.from_numpy(aligned_vertices)
                    object_points_np = aligned_vertices
                    aligned_mesh = _load_trimesh_mesh(Path(gt_mesh_path))
                    aligned_mesh_vertices = np.asarray(aligned_mesh.vertices, dtype=np.float64)
                    aligned_mesh_faces = np.asarray(aligned_mesh.faces, dtype=np.int32)
                    bbox_diag = float(
                        np.linalg.norm(
                            aligned_mesh_vertices.max(axis=0) - aligned_mesh_vertices.min(axis=0)
                        )
                    )
                    cluster_size = bbox_diag * self.mesh_mask_cluster_ratio
                    raster_mesh_vertices, raster_mesh_faces = _simplify_mesh_vertex_clustering(
                        vertices=aligned_mesh_vertices,
                        faces=aligned_mesh_faces,
                        cluster_size=cluster_size,
                    )
                else:
                    gt_points = self._load_scene_gt_points(tf, str(scene["gt_points_member"]))
                    if object_points is None:
                        object_points = gt_points
                        object_points_np = gt_points.cpu().numpy().astype(np.float32, copy=False)
                    alignment_stats = None
                normalized_extrinsics, avg_scale = _normalize_camera_extrinsics_from_reference_points(
                    extrinsics=extrinsic_tensor,
                    reference_world_points=gt_points.to(torch.float32),
                )
                camera_pair_indices = _shared_sparse_point_pairs(
                    frame_records=selected_frames,
                    min_shared_points=self.camera_min_shared_sparse_points,
                )
                object_masks = []
                for extrinsic, intrinsic in zip(extrinsics, intrinsics):
                    if gt_mesh_path is not None:
                        object_mask = _rasterize_mesh_mask(
                            vertices_world=np.asarray(raster_mesh_vertices, dtype=np.float64),
                            faces=np.asarray(raster_mesh_faces, dtype=np.int32),
                            extrinsic=np.asarray(extrinsic, dtype=np.float32),
                            intrinsic=np.asarray(intrinsic, dtype=np.float32),
                            image_hw=(self.image_size, self.image_size),
                        )
                    else:
                        object_mask = _project_object_points_to_mask(
                            points_world=object_points_np,
                            extrinsic=np.asarray(extrinsic, dtype=np.float32),
                            intrinsic=np.asarray(intrinsic, dtype=np.float32),
                            image_hw=(self.image_size, self.image_size),
                            point_radius=self.object_mask_point_radius,
                            dilation_kernel_size=self.object_mask_dilation_kernel_size,
                            projection_trim_quantile=self.object_mask_projection_trim_quantile,
                        )
                    object_masks.append(object_mask)
                object_mask_tensor = torch.from_numpy(np.stack(object_masks))
                object_mask_coverage = [
                    float(mask.astype(np.float32, copy=False).mean())
                    for mask in object_masks
                ]

                sample_id = f"{self.subset}/{scene['name']}"
                yield BenchmarkSample(
                    sample_id=sample_id,
                    images=image_tensor,
                    masks=None,
                    depths=None,
                    cam_points=None,
                    world_points=None,
                    point_masks=object_mask_tensor,
                    extrinsics=normalized_extrinsics,
                    intrinsics=intrinsic_tensor,
                    raw_depths=None,
                    raw_extrinsics=extrinsic_tensor,
                    raw_intrinsics=intrinsic_tensor,
                    normalization_scale=avg_scale,
                    normalization_reference_extrinsic=extrinsic_tensor[0],
                    gt_mesh_path=gt_mesh_path,
                    gt_point_cloud=None if gt_mesh_path is not None else gt_points,
                    metadata={
                        "seq_name": str(scene["name"]),
                        "frame_ids": frame_ids,
                        "frame_names": frame_names,
                        "object_mask_coverage": object_mask_coverage,
                        "mesh_alignment": alignment_stats,
                    },
                    protocol={
                        "camera_pair_indices": camera_pair_indices,
                        "camera_min_shared_sparse_points": self.camera_min_shared_sparse_points,
                        "pred_point_sample_points": self.pred_point_sample_points,
                        "gt_point_sample_points": self.gt_point_sample_points,
                        "small_translation_epsilon": self.small_translation_epsilon,
                        "tsdf_resolution": self.tsdf_resolution,
                        "tsdf_sdf_trunc_factor": self.tsdf_sdf_trunc_factor,
                    },
                )
