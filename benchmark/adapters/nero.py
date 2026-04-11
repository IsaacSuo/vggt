from __future__ import annotations

import io
import pickle
import struct
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from benchmark.adapters.base import BenchmarkDatasetAdapter, BenchmarkSample
from benchmark.plan import DatasetSpec


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
        self.max_scenes = config.get("max_scenes")
        self.gt_point_sample_points = int(config.get("gt_point_sample_points", 20000))
        self.pred_point_sample_points = int(config.get("pred_point_sample_points", 20000))
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
                    scene_assets[scene_name]["gt_points_member"] = member.name

            scenes: List[Dict[str, object]] = []
            for scene_name in sorted(scene_assets):
                assets = scene_assets[scene_name]
                if (
                    not assets["image_members"]
                    or assets["cameras_member"] is None
                    or assets["images_member"] is None
                    or assets["gt_points_member"] is None
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
                        "gt_points_member": assets["gt_points_member"],
                    }
                )

        if self.max_scenes is not None:
            scenes = scenes[: int(self.max_scenes)]
        return scenes

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

                gt_points = self._load_scene_gt_points(tf, str(scene["gt_points_member"]))
                normalized_extrinsics, avg_scale = _normalize_camera_extrinsics_from_reference_points(
                    extrinsics=extrinsic_tensor,
                    reference_world_points=gt_points.to(torch.float32),
                )
                camera_pair_indices = _shared_sparse_point_pairs(
                    frame_records=selected_frames,
                    min_shared_points=self.camera_min_shared_sparse_points,
                )

                sample_id = f"{self.subset}/{scene['name']}"
                yield BenchmarkSample(
                    sample_id=sample_id,
                    images=image_tensor,
                    masks=None,
                    depths=None,
                    cam_points=None,
                    world_points=None,
                    point_masks=None,
                    extrinsics=normalized_extrinsics,
                    intrinsics=intrinsic_tensor,
                    raw_depths=None,
                    raw_extrinsics=extrinsic_tensor,
                    raw_intrinsics=intrinsic_tensor,
                    normalization_scale=avg_scale,
                    normalization_reference_extrinsic=extrinsic_tensor[0],
                    gt_point_cloud=gt_points,
                    metadata={
                        "seq_name": str(scene["name"]),
                        "frame_ids": frame_ids,
                        "frame_names": frame_names,
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
