from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List

import numpy as np
import torch

from benchmark.adapters.base import BenchmarkDatasetAdapter, BenchmarkSample
from benchmark.plan import DatasetSpec
from data.datasets.openmaterial import OpenMaterialDataset, _resolve_gt_mesh_path
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


def _build_common_conf(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        img_size=int(config.get("img_size", 518)),
        patch_size=int(config.get("patch_size", 14)),
        debug=bool(config.get("debug", False)),
        training=False,
        load_mask=bool(config.get("load_mask", True)),
        load_depth=bool(config.get("load_depth", True)),
        mesh_near_plane=float(config.get("mesh_near_plane", 1e-3)),
        depth_cache_max_mb=int(config.get("depth_cache_max_mb", 512)),
        depth_precompute_dir=config.get("depth_precompute_dir"),
        depth_precompute_subdir=str(config.get("depth_precompute_subdir", "depth_mesh")),
        prefer_precomputed_depth=bool(config.get("prefer_precomputed_depth", True)),
        require_precomputed_depth=bool(config.get("require_precomputed_depth", False)),
        rescale=bool(config.get("rescale", True)),
        rescale_aug=bool(config.get("rescale_aug", False)),
        landscape_check=bool(config.get("landscape_check", True)),
        augs=SimpleNamespace(scales=None),
    )


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

    raise ValueError(f"Unsupported OpenMaterial frame selection strategy: {strategy}")


def _convert_raw_batch_to_sample(raw_batch: dict, sample_id: str) -> BenchmarkSample:
    images = torch.from_numpy(np.stack(raw_batch["images"]).astype(np.float32)).contiguous()
    images = images.permute(0, 3, 1, 2).to(torch.float32).div(255.0)

    depths = torch.from_numpy(np.stack(raw_batch["depths"]).astype(np.float32))
    extrinsics = torch.from_numpy(np.stack(raw_batch["extrinsics"]).astype(np.float32))
    intrinsics = torch.from_numpy(np.stack(raw_batch["intrinsics"]).astype(np.float32))
    cam_points = torch.from_numpy(np.stack(raw_batch["cam_points"]).astype(np.float32))
    world_points = torch.from_numpy(np.stack(raw_batch["world_points"]).astype(np.float32))
    point_masks = torch.from_numpy(np.stack(raw_batch["point_masks"]))

    first_extrinsic = extrinsics[0]
    first_rotation = first_extrinsic[:3, :3]
    first_translation = first_extrinsic[:3, 3]
    world_points_first_cam = (
        world_points @ first_rotation.transpose(-1, -2).unsqueeze(0).unsqueeze(0)
    ) + first_translation.view(1, 1, 1, 3)
    world_point_dist = world_points_first_cam.norm(dim=-1)
    valid_mask_float = point_masks.to(torch.float32)
    dist_sum = (world_point_dist * valid_mask_float).sum()
    valid_count = valid_mask_float.sum()
    avg_scale = float((dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6).item())

    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics.unsqueeze(0),
        cam_points=cam_points.unsqueeze(0),
        world_points=world_points.unsqueeze(0),
        depths=depths.unsqueeze(0),
        point_masks=point_masks.unsqueeze(0),
    )

    masks = None
    if "masks" in raw_batch and all(mask is not None for mask in raw_batch["masks"]):
        masks = torch.from_numpy(np.stack(raw_batch["masks"]).astype(np.float32))

    return BenchmarkSample(
        sample_id=sample_id,
        images=images,
        masks=masks,
        depths=normalized_depths.squeeze(0),
        cam_points=normalized_cam_points.squeeze(0),
        world_points=normalized_world_points.squeeze(0),
        point_masks=point_masks,
        extrinsics=normalized_extrinsics.squeeze(0),
        intrinsics=intrinsics,
        raw_depths=depths,
        raw_extrinsics=extrinsics,
        raw_intrinsics=intrinsics,
        normalization_scale=avg_scale,
        normalization_reference_extrinsic=first_extrinsic,
        gt_mesh_path=raw_batch.get("gt_mesh_path"),
        metadata={
            "seq_name": raw_batch["seq_name"],
            "frame_ids": raw_batch["ids"].tolist() if hasattr(raw_batch["ids"], "tolist") else list(raw_batch["ids"]),
        },
        protocol={
            "gt_mesh_sample_points": raw_batch.get("gt_mesh_sample_points", 20000),
            "pred_point_sample_points": raw_batch.get("pred_point_sample_points", 20000),
            "covisibility_min_overlap_ratio": raw_batch.get("covisibility_min_overlap_ratio", 0.05),
            "covisibility_min_visible_points": raw_batch.get("covisibility_min_visible_points", 256),
            "covisibility_depth_abs_tol": raw_batch.get("covisibility_depth_abs_tol", 0.01),
            "covisibility_depth_rel_tol": raw_batch.get("covisibility_depth_rel_tol", 0.01),
            "small_translation_epsilon": raw_batch.get("small_translation_epsilon", 1e-4),
            "tsdf_resolution": raw_batch.get("tsdf_resolution", 256),
            "tsdf_sdf_trunc_factor": raw_batch.get("tsdf_sdf_trunc_factor", 4.0),
        },
    )


class OpenMaterialBenchmarkAdapter(BenchmarkDatasetAdapter):
    def __init__(self, spec: DatasetSpec):
        super().__init__(spec)
        config = spec.config
        data_dir = Path(config["data_dir"])
        if not data_dir.exists():
            raise FileNotFoundError(f"OpenMaterial benchmark data_dir not found: {data_dir}")

        common_conf = _build_common_conf(config)
        split = str(config.get("split", "test"))
        self.dataset = OpenMaterialDataset(
            common_conf=common_conf,
            split=split,
            data_dir=str(data_dir),
            min_num_images=int(config.get("min_num_images", 8)),
            len_train=int(config.get("len_train", 100000)),
            len_test=int(config.get("len_test", 10000)),
            load_mask=bool(config.get("load_mask", True)),
            load_mesh_depth=bool(config.get("load_mesh_depth", True)),
            scene_list_path=config.get("scene_list_path"),
        )
        self.num_frames = int(config.get("num_frames", 16))
        self.aspect_ratio = float(config.get("aspect_ratio", 1.0))
        self.frame_selection = str(config.get("frame_selection", "evenly_spaced"))
        self.max_scenes = config.get("max_scenes")
        self.gt_mesh_sample_points = int(config.get("gt_mesh_sample_points", 20000))
        self.pred_point_sample_points = int(config.get("pred_point_sample_points", 20000))
        self.covisibility_min_overlap_ratio = float(config.get("covisibility_min_overlap_ratio", 0.05))
        self.covisibility_min_visible_points = int(config.get("covisibility_min_visible_points", 256))
        self.covisibility_depth_abs_tol = float(config.get("covisibility_depth_abs_tol", 0.01))
        self.covisibility_depth_rel_tol = float(config.get("covisibility_depth_rel_tol", 0.01))
        self.small_translation_epsilon = float(config.get("small_translation_epsilon", 1e-4))
        self.tsdf_resolution = int(config.get("tsdf_resolution", 256))
        self.tsdf_sdf_trunc_factor = float(config.get("tsdf_sdf_trunc_factor", 4.0))

    def iter_samples(self) -> Iterator[BenchmarkSample]:
        scene_indices: List[int] = list(range(len(self.dataset.scenes)))
        if self.max_scenes is not None:
            scene_indices = scene_indices[: int(self.max_scenes)]

        for scene_idx in scene_indices:
            scene = self.dataset.scenes[scene_idx]
            frame_ids = _select_frame_ids(
                num_frames_total=len(scene["frames"]),
                num_frames_target=self.num_frames,
                strategy=self.frame_selection,
            )
            sample_id = f"{scene['hash_id']}/{scene['name']}"
            raw_batch = self.dataset.get_data(
                seq_index=scene_idx,
                ids=frame_ids,
                aspect_ratio=self.aspect_ratio,
            )
            raw_batch["gt_mesh_path"] = _resolve_gt_mesh_path(str(self.dataset.data_dir), scene["hash_id"])
            raw_batch["gt_mesh_sample_points"] = self.gt_mesh_sample_points
            raw_batch["pred_point_sample_points"] = self.pred_point_sample_points
            raw_batch["covisibility_min_overlap_ratio"] = self.covisibility_min_overlap_ratio
            raw_batch["covisibility_min_visible_points"] = self.covisibility_min_visible_points
            raw_batch["covisibility_depth_abs_tol"] = self.covisibility_depth_abs_tol
            raw_batch["covisibility_depth_rel_tol"] = self.covisibility_depth_rel_tol
            raw_batch["small_translation_epsilon"] = self.small_translation_epsilon
            raw_batch["tsdf_resolution"] = self.tsdf_resolution
            raw_batch["tsdf_sdf_trunc_factor"] = self.tsdf_sdf_trunc_factor
            yield _convert_raw_batch_to_sample(raw_batch, sample_id=sample_id)
