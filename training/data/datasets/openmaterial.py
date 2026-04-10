# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenMaterial Dataset for Visual-Hull-Aware VGGT Training.

This dataset loads synthetic data rendered with Blender in NeRF-style format,
including images, masks, and camera parameters from transforms.json files.

Data structure expected:
    OpenMaterial/
        {hash}/
            {scene_name}/
                transforms_train.json
                transforms_test.json
                train/
                    images/
                        000.png, 001.png, ...
                    masks/
                        000.png, 001.png, ...
                test/
                    images/
                    masks/
"""

import json
import logging
import os
import os.path as osp
import random
from collections import OrderedDict
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data.dataset_util import read_image_cv2
from data.base_dataset import BaseDataset


logger = logging.getLogger(__name__)


_PLY_DTYPE_MAP = {
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


def _resolve_mask_path(image_path: str) -> Optional[str]:
    """
    Resolve the corresponding mask path for an image.

    Supports both `mask/` and `masks/` directory names because the
    OpenMaterial subsets in the wild are not fully consistent here.
    """
    candidates = [
        image_path.replace("/images/", "/masks/"),
        image_path.replace("/images/", "/mask/"),
    ]

    for candidate in candidates:
        if osp.exists(candidate):
            return candidate

    return None


def _load_scene_name_filter(path: str) -> set[str]:
    """Load scene identifiers from a newline-delimited manifest file."""
    scene_names = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            scene_names.add(line)
    return scene_names


def _scene_matches_filter(
    scene_name: str,
    hash_id: str,
    allowed_scene_names: Optional[set[str]],
) -> bool:
    """Match either `scene_name` or `hash_id/scene_name` against a manifest set."""
    if allowed_scene_names is None:
        return True
    return scene_name in allowed_scene_names or f"{hash_id}/{scene_name}" in allowed_scene_names


def _replace_image_dir_and_extension(path: str, replacement_dir: str, extension: str) -> str:
    root, _ = osp.splitext(path)
    replaced = root.replace("/images/", f"/{replacement_dir}/")
    if replaced == root:
        replaced = root.replace("\\images\\", f"\\{replacement_dir}\\")
    return replaced + extension


def _resolve_precomputed_depth_path(
    image_path: str,
    depth_subdir: str = "depth_mesh",
    scene_dir: Optional[str] = None,
    cache_root: Optional[str] = None,
) -> str:
    """Map an image path to the offline mesh-depth cache path."""
    if cache_root is None:
        return _replace_image_dir_and_extension(image_path, depth_subdir, ".npy")

    if scene_dir is None:
        raise ValueError("scene_dir must be provided when cache_root is set")

    rel_image_path = osp.relpath(image_path, scene_dir)
    rel_depth_path = _replace_image_dir_and_extension(rel_image_path, depth_subdir, ".npy")
    scene_name = osp.basename(scene_dir)
    hash_id = osp.basename(osp.dirname(scene_dir))
    return osp.join(cache_root, hash_id, scene_name, rel_depth_path)


def _resolve_gt_mesh_path(data_dir: str, hash_id: str) -> Optional[str]:
    """Resolve the GT mesh path for one OpenMaterial object hash."""
    candidates = [
        osp.join(data_dir, "groundtruth_ablation", hash_id, f"clean_{hash_id}.ply"),
        osp.join(data_dir, "groundtruth", hash_id, f"clean_{hash_id}.ply"),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return None


def _load_ply_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load vertices and triangle faces from a binary little-endian PLY mesh.
    """
    vertex_count = None
    face_count = None
    vertex_properties = []
    current_element = None

    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")

            decoded = line.decode("ascii", errors="strict").strip()
            if decoded == "ply":
                continue
            if decoded == "format binary_little_endian 1.0":
                continue
            if decoded.startswith("comment "):
                continue
            if decoded.startswith("element "):
                parts = decoded.split()
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                    vertex_properties = []
                elif current_element == "face":
                    face_count = int(parts[2])
                continue
            if decoded.startswith("property "):
                parts = decoded.split()
                if current_element == "vertex":
                    _, prop_type, prop_name = parts
                    if prop_type not in _PLY_DTYPE_MAP:
                        raise ValueError(f"Unsupported PLY property type `{prop_type}` in {path}")
                    vertex_properties.append((prop_name, "<" + _PLY_DTYPE_MAP[prop_type]))
                continue
            if decoded == "end_header":
                break

        if vertex_count is None or face_count is None or not vertex_properties:
            raise ValueError(f"PLY mesh metadata missing in {path}")

        vertex_dtype = np.dtype(vertex_properties)
        vertices = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)

        triangles: List[List[int]] = []
        for _ in range(face_count):
            degree = np.fromfile(f, dtype=np.uint8, count=1)
            if degree.size == 0:
                raise ValueError(f"Unexpected EOF while reading PLY faces: {path}")
            degree = int(degree[0])
            face = np.fromfile(f, dtype=np.int32, count=degree)
            if face.size != degree:
                raise ValueError(f"Unexpected EOF while reading PLY face indices: {path}")
            if degree < 3:
                continue
            if degree == 3:
                triangles.append(face.tolist())
            else:
                for i in range(1, degree - 1):
                    triangles.append([int(face[0]), int(face[i]), int(face[i + 1])])

    required = ("x", "y", "z")
    if any(name not in vertices.dtype.names for name in required):
        raise ValueError(f"PLY file {path} does not contain XYZ coordinates")

    vertices_xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    return vertices_xyz, faces


def _rasterize_mesh_depth(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_shape_hw: Tuple[int, int],
    near_plane: float = 1e-3,
) -> np.ndarray:
    """
    Rasterize a triangle mesh into a depth map with a CPU z-buffer.
    """
    height, width = image_shape_hw
    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    if vertices_world.size == 0 or faces.size == 0:
        depth_map.fill(0.0)
        return depth_map

    vertices_cam = vertices_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    eps = 1e-8

    def clip_triangle_against_near_plane(triangle_cam: np.ndarray) -> List[np.ndarray]:
        """
        Clip one camera-space triangle against the near plane z = near_plane.

        Triangles that cross the plane are split into up to two triangles, which
        avoids the large holes caused by dropping them entirely.
        """
        clipped_polygon: List[np.ndarray] = []
        prev = triangle_cam[-1]
        prev_inside = prev[2] >= near_plane

        for curr in triangle_cam:
            curr_inside = curr[2] >= near_plane

            if curr_inside != prev_inside:
                t = (near_plane - prev[2]) / (curr[2] - prev[2] + eps)
                clipped_polygon.append(prev + t * (curr - prev))

            if curr_inside:
                clipped_polygon.append(curr)

            prev = curr
            prev_inside = curr_inside

        if len(clipped_polygon) < 3:
            return []

        base = clipped_polygon[0]
        triangles_out = []
        for idx in range(1, len(clipped_polygon) - 1):
            triangles_out.append(
                np.stack([base, clipped_polygon[idx], clipped_polygon[idx + 1]], axis=0)
            )
        return triangles_out

    for face in faces:
        triangle_cam = vertices_cam[face]
        for clipped_triangle in clip_triangle_against_near_plane(triangle_cam):
            x = clipped_triangle[:, 0]
            y = clipped_triangle[:, 1]
            z = clipped_triangle[:, 2]
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
            pts = np.stack([u, v], axis=-1)
            z_inv = 1.0 / z

            # Expand the bbox by 1 pixel to reduce boundary holes from rounding.
            min_x = max(int(np.floor(np.min(pts[:, 0]))) - 1, 0)
            max_x = min(int(np.ceil(np.max(pts[:, 0]))) + 1, width - 1)
            min_y = max(int(np.floor(np.min(pts[:, 1]))) - 1, 0)
            max_y = min(int(np.ceil(np.max(pts[:, 1]))) + 1, height - 1)

            if min_x > max_x or min_y > max_y:
                continue

            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < eps:
                continue

            xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
            ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
            xx, yy = np.meshgrid(xs, ys)

            w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
            w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not np.any(inside):
                continue

            depth = 1.0 / (w0 * z_inv[0] + w1 * z_inv[1] + w2 * z_inv[2] + eps)
            region = depth_map[min_y : max_y + 1, min_x : max_x + 1]
            update = inside & (depth > near_plane) & (depth < region)
            if np.any(update):
                region[update] = depth[update].astype(np.float32)

    depth_map[~np.isfinite(depth_map)] = 0.0
    return depth_map


class OpenMaterialDataset(BaseDataset):
    """
    Dataset for OpenMaterial synthetic data with mask support.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        data_dir: str = None,
        min_num_images: int = 8,
        len_train: int = 100000,
        len_test: int = 10000,
        load_mask: bool = True,
        load_mesh_depth: bool = True,
        scene_list_path: Optional[str] = None,
    ):
        """
        Initialize the OpenMaterial dataset.

        Args:
            common_conf: Configuration object with common settings.
            split: Dataset split, either 'train' or 'test'.
            data_dir: Root directory containing OpenMaterial scenes.
            min_num_images: Minimum number of images per scene.
            len_train: Length of the training dataset.
            len_test: Length of the test dataset.
            load_mask: Whether to load masks.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.load_mask = getattr(common_conf, "load_mask", load_mask)
        self.load_depth = getattr(common_conf, "load_depth", False)
        self.load_mesh_depth = load_mesh_depth
        self.mesh_near_plane = float(getattr(common_conf, "mesh_near_plane", 1e-3))
        self.depth_precompute_dir = getattr(common_conf, "depth_precompute_dir", None)
        self.depth_precompute_subdir = str(getattr(common_conf, "depth_precompute_subdir", "depth_mesh"))
        self.prefer_precomputed_depth = bool(getattr(common_conf, "prefer_precomputed_depth", True))
        self.require_precomputed_depth = bool(getattr(common_conf, "require_precomputed_depth", False))
        self._mesh_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._depth_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._logged_precomputed_depth_hit = False
        self._logged_raster_fallback = False
        depth_cache_max_mb = int(getattr(common_conf, "depth_cache_max_mb", 512))
        self._depth_cache_max_bytes = depth_cache_max_mb * 1024 * 1024
        self._depth_cache_total_bytes = 0

        if data_dir is None:
            raise ValueError("data_dir must be specified.")

        self.data_dir = data_dir
        self.split = split
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.scene_list_path = scene_list_path
        self.allowed_scene_names = None

        if self.scene_list_path is not None:
            if not osp.exists(self.scene_list_path):
                raise FileNotFoundError(
                    f"OpenMaterial scene list not found: {self.scene_list_path}"
                )
            self.allowed_scene_names = _load_scene_name_filter(self.scene_list_path)
            logger.info(
                "OpenMaterial [%s]: restricting scenes using `%s` (%d entries)",
                split,
                self.scene_list_path,
                len(self.allowed_scene_names),
            )

        # Find all scenes
        self.scenes = self._find_scenes()
        logger.info(f"OpenMaterial [{split}]: Found {len(self.scenes)} scenes")

    def _find_scenes(self) -> List[dict]:
        """
        Find all valid scenes in the data directory.

        Returns:
            List of scene dictionaries with metadata.
        """
        scenes = []
        transform_file = f"transforms_{self.split}.json"

        # Search pattern: data_dir/{hash}/{scene_name}/transforms_{split}.json
        pattern = osp.join(self.data_dir, "*", "*", transform_file)
        transform_files = glob(pattern)

        for tf_path in transform_files:
            try:
                with open(tf_path, 'r') as f:
                    data = json.load(f)

                frames = data.get('frames', [])
                if len(frames) < self.min_num_images:
                    continue

                scene_dir = osp.dirname(tf_path)
                scene_name = osp.basename(scene_dir)
                hash_id = osp.basename(osp.dirname(scene_dir))

                if not _scene_matches_filter(
                    scene_name=scene_name,
                    hash_id=hash_id,
                    allowed_scene_names=self.allowed_scene_names,
                ):
                    continue

                # Extract camera intrinsics
                intrinsics = {
                    'fl_x': data.get('fl_x'),
                    'fl_y': data.get('fl_y'),
                    'cx': data.get('cx'),
                    'cy': data.get('cy'),
                    'w': data.get('w'),
                    'h': data.get('h'),
                }

                scenes.append({
                    'name': scene_name,
                    'hash_id': hash_id,
                    'dir': scene_dir,
                    'transform_path': tf_path,
                    'frames': frames,
                    'intrinsics': intrinsics,
                })

            except Exception as e:
                logger.warning(f"Failed to load {tf_path}: {e}")
                continue

        return scenes

    def _resolve_gt_mesh_path(self, scene: dict) -> Optional[str]:
        """Resolve the GT mesh path for the current scene hash."""
        hash_id = scene.get("hash_id")
        if hash_id is None:
            return None
        return _resolve_gt_mesh_path(self.data_dir, hash_id)

    def _get_scene_mesh(self, scene: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load and cache the GT mesh if mesh supervision is enabled."""
        if not (self.load_depth and self.load_mesh_depth):
            return None

        mesh_path = self._resolve_gt_mesh_path(scene)
        if mesh_path is None:
            raise FileNotFoundError(
                f"GT mesh not found for OpenMaterial scene `{scene['name']}` (hash `{scene['hash_id']}`)"
            )

        if mesh_path not in self._mesh_cache:
            self._mesh_cache[mesh_path] = _load_ply_mesh(mesh_path)

        return self._mesh_cache[mesh_path]

    def _process_mask_with_image_geometry(
        self,
        mask: np.ndarray,
        extri_opencv: np.ndarray,
        intri_opencv: np.ndarray,
        original_size: np.ndarray,
        target_image_shape: np.ndarray,
        filepath: str,
    ) -> np.ndarray:
        """
        Apply the exact same geometric transforms as `process_one_image` to a mask.

        We replay the same random augmentation by restoring the RNG state around the
        image preprocessing call inside `get_data`.
        """
        mask_rgb = np.repeat((mask > 0.5).astype(np.uint8)[..., None] * 255, 3, axis=2)
        processed_mask_rgb, _, _, _, _, _, _, _ = self.process_one_image(
            mask_rgb,
            np.zeros(mask.shape[:2], dtype=np.float32),
            extri_opencv,
            intri_opencv,
            original_size,
            target_image_shape,
            filepath=filepath,
        )
        return (processed_mask_rgb[..., 0] > 127).astype(np.float32)

    def _get_cached_depth(
        self,
        cache_key: str,
        render_fn,
    ) -> np.ndarray:
        """Byte-budgeted LRU cache for expensive per-frame depth renders."""
        if cache_key in self._depth_cache:
            depth = self._depth_cache.pop(cache_key)
            self._depth_cache[cache_key] = depth
            return depth.copy()

        depth = render_fn()

        if depth.nbytes > self._depth_cache_max_bytes:
            return depth.copy()

        while (
            self._depth_cache
            and self._depth_cache_total_bytes + depth.nbytes > self._depth_cache_max_bytes
        ):
            _, evicted = self._depth_cache.popitem(last=False)
            self._depth_cache_total_bytes -= evicted.nbytes

        self._depth_cache[cache_key] = depth
        self._depth_cache_total_bytes += depth.nbytes
        return depth.copy()

    def _load_precomputed_depth(
        self,
        scene_dir: str,
        image_path: str,
        cache_key: str,
        expected_shape_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Load offline rasterized depth if present on disk."""
        if not (self.load_depth and self.load_mesh_depth and self.prefer_precomputed_depth):
            return None

        depth_path = _resolve_precomputed_depth_path(
            image_path,
            depth_subdir=self.depth_precompute_subdir,
            scene_dir=scene_dir,
            cache_root=self.depth_precompute_dir,
        )
        if not osp.exists(depth_path):
            if self.require_precomputed_depth:
                raise FileNotFoundError(
                    f"Precomputed OpenMaterial depth not found at {depth_path}. "
                    f"Run the offline precompute step first or disable require_precomputed_depth."
                )
            return None

        def load_fn() -> np.ndarray:
            depth = np.load(depth_path)
            if depth.shape != expected_shape_hw:
                raise ValueError(
                    f"Precomputed depth shape {depth.shape} does not match image shape "
                    f"{expected_shape_hw} for {depth_path}"
                )
            return depth.astype(np.float32, copy=False)

        depth = self._get_cached_depth(f"precomputed::{cache_key}", load_fn)
        if not self._logged_precomputed_depth_hit:
            logger.info(
                "OpenMaterial: using precomputed mesh depths from `%s` when available",
                depth_path if self.depth_precompute_dir else self.depth_precompute_subdir,
            )
            self._logged_precomputed_depth_hit = True
        return depth

    def _get_frame_depth(
        self,
        scene: dict,
        frame: dict,
        image_path: str,
        image_shape_hw: Tuple[int, int],
        extri_opencv_raw: np.ndarray,
        intri_opencv_raw: np.ndarray,
        scene_mesh: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Load precomputed depth or fall back to on-the-fly rasterization."""
        cache_key = f"{scene['hash_id']}::{self.split}::{frame['file_path']}"
        depth = self._load_precomputed_depth(scene["dir"], image_path, cache_key, image_shape_hw)
        if depth is not None:
            return depth

        if scene_mesh is None:
            return np.zeros(image_shape_hw, dtype=np.float32)

        if not self._logged_raster_fallback:
            logger.info(
                "OpenMaterial: falling back to online CPU mesh rasterization for missing `%s` depth caches",
                self.depth_precompute_subdir,
            )
            self._logged_raster_fallback = True

        vertices_world, faces = scene_mesh
        return self._get_cached_depth(
            f"raster::{cache_key}",
            lambda: _rasterize_mesh_depth(
                vertices_world,
                faces,
                extri_opencv_raw,
                intri_opencv_raw,
                image_shape_hw,
                near_plane=self.mesh_near_plane,
            ),
        )

    def __len__(self):
        return self.len_train

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific scene.

        Args:
            seq_index: Index of the scene.
            img_per_seq: Number of images to sample.
            seq_name: Name of the scene (unused, kept for API compatibility).
            ids: Specific frame indices to use.
            aspect_ratio: Target aspect ratio.

        Returns:
            Batch dictionary with images, masks, cameras, etc.
        """
        # Random scene selection
        if seq_index is None:
            seq_index = random.randint(0, len(self.scenes) - 1)
        else:
            seq_index = seq_index % len(self.scenes)

        scene = self.scenes[seq_index]
        frames = scene['frames']
        intrinsics = scene['intrinsics']
        scene_mesh = self._get_scene_mesh(scene)

        # Sample frame indices
        if ids is None:
            if img_per_seq is None:
                img_per_seq = min(16, len(frames))
            ids = np.random.choice(len(frames), img_per_seq, replace=False)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        masks = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics_list = []
        intrinsics_list = []
        original_sizes = []

        for idx in ids:
            frame = frames[idx]

            # Load image
            image_path = osp.join(scene['dir'], frame['file_path'])
            if not osp.exists(image_path):
                # Try alternative path
                image_path = osp.join(scene['dir'], osp.basename(frame['file_path']))

            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])

            # Load mask if available
            mask = None
            if self.load_mask:
                mask_path = _resolve_mask_path(image_path)
                if mask_path is not None:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 128).astype(np.float32)

            # Build intrinsic matrix
            fx = intrinsics['fl_x']
            fy = intrinsics['fl_y']
            cx = intrinsics['cx']
            cy = intrinsics['cy']

            intri_opencv = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float64)

            # Convert camera-to-world (NeRF) to world-to-camera (OpenCV)
            c2w = np.array(frame['transform_matrix'], dtype=np.float64)

            # NeRF uses OpenGL convention (Y up, -Z forward)
            # OpenCV uses (Y down, Z forward)
            # Apply conversion: flip Y and Z
            c2w_opencv = c2w.copy()
            c2w_opencv[:3, 1:3] *= -1  # Flip Y and Z columns

            # Invert to get world-to-camera
            extri_opencv = np.linalg.inv(c2w_opencv)
            extri_opencv_raw = extri_opencv.copy()
            intri_opencv_raw = intri_opencv.copy()

            depth_map = self._get_frame_depth(
                scene=scene,
                frame=frame,
                image_path=image_path,
                image_shape_hw=tuple(image.shape[:2]),
                extri_opencv_raw=extri_opencv_raw,
                intri_opencv_raw=intri_opencv_raw,
                scene_mesh=scene_mesh,
            )
            if mask is not None:
                depth_map[mask < 0.5] = 0.0

            # Process image
            rng_state = np.random.get_state()
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            if mask is not None:
                post_image_rng_state = np.random.get_state()
                np.random.set_state(rng_state)
                mask = self._process_mask_with_image_geometry(
                    mask,
                    extri_opencv_raw,
                    intri_opencv_raw,
                    original_size,
                    target_image_shape,
                    image_path,
                )
                np.random.set_state(post_image_rng_state)

            images.append(image)
            masks.append(mask)
            depths.append(depth_map)
            extrinsics_list.append(extri_opencv)
            intrinsics_list.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        batch = {
            "seq_name": f"openmaterial_{scene['name']}",
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "masks": masks,  # Visual hull masks for training
            "depths": depths,
            "extrinsics": extrinsics_list,
            "intrinsics": intrinsics_list,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        return batch
