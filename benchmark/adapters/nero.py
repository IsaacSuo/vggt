from __future__ import annotations

import io
import pickle
import struct
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from benchmark.adapters.base import BenchmarkDatasetAdapter, BenchmarkSample
from benchmark.plan import DatasetSpec


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
    type_map = {
        "char": "b",
        "uchar": "B",
        "short": "h",
        "ushort": "H",
        "int": "i",
        "uint": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }

    while True:
        line = file_obj.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading NeRO PLY header.")
        decoded = line.decode("ascii", errors="strict").strip()
        if decoded == "ply" or decoded == "format binary_little_endian 1.0" or decoded.startswith("comment "):
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
            if prop_type not in type_map:
                raise ValueError(f"Unsupported PLY property type `{prop_type}`")
            vertex_properties.append((prop_name, type_map[prop_type]))
            continue
        if decoded == "end_header":
            break

    if vertex_count is None or not vertex_properties:
        raise ValueError("NeRO PLY is missing vertex metadata.")

    struct_fmt = "<" + "".join(fmt for _, fmt in vertex_properties)
    struct_size = struct.calcsize(struct_fmt)
    raw_rows = [struct.unpack(struct_fmt, file_obj.read(struct_size)) for _ in range(vertex_count)]
    raw = np.asarray(raw_rows)
    prop_names = [name for name, _ in vertex_properties]
    xyz = np.stack(
        [
            raw[:, prop_names.index("x")],
            raw[:, prop_names.index("y")],
            raw[:, prop_names.index("z")],
        ],
        axis=-1,
    )
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
