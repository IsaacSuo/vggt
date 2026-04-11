from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch

from benchmark.adapters.base import BenchmarkSample
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


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


def _load_ply_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
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
            if decoded == "ply" or decoded == "format binary_little_endian 1.0" or decoded.startswith("comment "):
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

        triangles = []
        for _ in range(face_count):
            degree = np.fromfile(f, dtype=np.uint8, count=1)
            if degree.size == 0:
                raise ValueError(f"Unexpected EOF while reading PLY faces: {path}")
            degree_int = int(degree[0])
            face = np.fromfile(f, dtype=np.int32, count=degree_int)
            if face.size != degree_int:
                raise ValueError(f"Unexpected EOF while reading PLY face indices: {path}")
            if degree_int < 3:
                continue
            if degree_int == 3:
                triangles.append(face.tolist())
            else:
                for i in range(1, degree_int - 1):
                    triangles.append([int(face[0]), int(face[i]), int(face[i + 1])])

    required = ("x", "y", "z")
    if any(name not in vertices.dtype.names for name in required):
        raise ValueError(f"PLY file {path} does not contain XYZ coordinates")

    vertices_xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    return vertices_xyz, faces


def _pairwise_indices(num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(num_frames, num_frames, offset=1)


def _to_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    num_frames = extrinsics.shape[0]
    bottom = torch.zeros((num_frames, 1, 4), dtype=extrinsics.dtype, device=extrinsics.device)
    bottom[..., -1] = 1.0
    return torch.cat([extrinsics, bottom], dim=1)


def _denormalize_extrinsics(normalized_extrinsics: torch.Tensor, sample: BenchmarkSample) -> torch.Tensor:
    if sample.normalization_scale is None or sample.normalization_reference_extrinsic is None:
        raise ValueError("Sample is missing normalization metadata for camera denormalization.")

    extrinsics = normalized_extrinsics.clone()
    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * float(sample.normalization_scale)

    extrinsics_h = _to_homogeneous(extrinsics)
    reference_extrinsic = sample.normalization_reference_extrinsic.to(
        device=extrinsics.device,
        dtype=extrinsics.dtype,
    )
    reference_extrinsic_h = _to_homogeneous(reference_extrinsic.unsqueeze(0))[0]
    denormalized = extrinsics_h @ reference_extrinsic_h.unsqueeze(0)
    return denormalized[:, :3, :]


def _denormalize_world_points(normalized_world_points: torch.Tensor, sample: BenchmarkSample) -> torch.Tensor:
    if sample.normalization_scale is None or sample.normalization_reference_extrinsic is None:
        raise ValueError("Sample is missing normalization metadata for point denormalization.")

    first_extrinsic = sample.normalization_reference_extrinsic.to(
        device=normalized_world_points.device,
        dtype=normalized_world_points.dtype,
    )
    rotation = first_extrinsic[:3, :3]
    translation = first_extrinsic[:3, 3]
    points_first_camera = normalized_world_points * float(sample.normalization_scale)
    return (points_first_camera - translation.view(1, 1, 1, 3)) @ rotation


def _rotation_error_deg(pred_rotation: torch.Tensor, gt_rotation: torch.Tensor) -> torch.Tensor:
    rotation_delta = pred_rotation @ gt_rotation.transpose(-1, -2)
    trace = torch.diagonal(rotation_delta, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(min=-1.0, max=1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _translation_direction_error_deg(pred_translation: torch.Tensor, gt_translation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_norm = torch.norm(pred_translation, dim=-1, keepdim=True).clamp(min=eps)
    gt_norm = torch.norm(gt_translation, dim=-1, keepdim=True).clamp(min=eps)
    pred_unit = pred_translation / pred_norm
    gt_unit = gt_translation / gt_norm
    cos_theta = (pred_unit * gt_unit).sum(dim=-1).clamp(min=-1.0, max=1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _project_world_points(world_points: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam_points = world_points @ extrinsic[:3, :3].transpose(-1, -2) + extrinsic[:3, 3]
    z = cam_points[:, 2]
    z_safe = z.clamp(min=eps)
    u = intrinsic[0, 0] * (cam_points[:, 0] / z_safe) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (cam_points[:, 1] / z_safe) + intrinsic[1, 2]
    return u, v, z


def _visible_overlap_ratio(
    src_world_points: torch.Tensor,
    dst_depth: torch.Tensor,
    dst_valid_mask: torch.Tensor,
    dst_extrinsic: torch.Tensor,
    dst_intrinsic: torch.Tensor,
    depth_abs_tol: float,
    depth_rel_tol: float,
) -> Tuple[float, int]:
    if src_world_points.numel() == 0:
        return 0.0, 0

    u, v, z = _project_world_points(src_world_points, dst_extrinsic, dst_intrinsic)
    h, w = dst_depth.shape
    u_idx = torch.round(u).to(torch.long)
    v_idx = torch.round(v).to(torch.long)
    in_front = z > 0
    in_bounds = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
    projected = in_front & in_bounds
    if projected.sum().item() == 0:
        return 0.0, 0

    dst_depth_samples = dst_depth[v_idx[projected], u_idx[projected]]
    dst_valid_samples = dst_valid_mask[v_idx[projected], u_idx[projected]]
    proj_depth = z[projected]
    depth_tol = torch.maximum(
        torch.full_like(dst_depth_samples, depth_abs_tol),
        dst_depth_samples.abs() * depth_rel_tol,
    )
    depth_consistent = (proj_depth - dst_depth_samples).abs() <= depth_tol
    visible = dst_valid_samples & torch.isfinite(dst_depth_samples) & (dst_depth_samples > 0) & depth_consistent
    visible_count = int(visible.sum().item())
    return float(visible_count / max(src_world_points.shape[0], 1)), visible_count


def _covisible_pair_indices(sample: BenchmarkSample) -> Tuple[torch.Tensor, torch.Tensor]:
    explicit_pairs = sample.protocol.get("camera_pair_indices")
    if explicit_pairs is not None:
        device = sample.images.device
        if len(explicit_pairs) == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        pair_tensor = torch.as_tensor(explicit_pairs, dtype=torch.long, device=device)
        if pair_tensor.ndim != 2 or pair_tensor.shape[1] != 2:
            raise ValueError("camera_pair_indices must have shape [N, 2].")
        return pair_tensor[:, 0], pair_tensor[:, 1]

    if (
        sample.world_points is None
        or sample.depths is None
        or sample.point_masks is None
        or sample.extrinsics is None
        or sample.intrinsics is None
    ):
        return _pairwise_indices(sample.images.shape[0])

    world_points = sample.world_points.to(torch.float32)
    depths = sample.depths.to(torch.float32)
    valid_masks = sample.point_masks.bool()
    extrinsics = sample.extrinsics.to(torch.float32)
    intrinsics = sample.intrinsics.to(torch.float32)

    min_overlap_ratio = float(sample.protocol.get("covisibility_min_overlap_ratio", 0.05))
    min_visible_points = int(sample.protocol.get("covisibility_min_visible_points", 256))
    depth_abs_tol = float(sample.protocol.get("covisibility_depth_abs_tol", 0.01))
    depth_rel_tol = float(sample.protocol.get("covisibility_depth_rel_tol", 0.01))

    pair_i, pair_j = _pairwise_indices(world_points.shape[0])
    covisible_i = []
    covisible_j = []
    for idx in range(pair_i.numel()):
        i = int(pair_i[idx].item())
        j = int(pair_j[idx].item())
        src_points_i = world_points[i][valid_masks[i] & torch.isfinite(depths[i]) & (depths[i] > 0)]
        src_points_j = world_points[j][valid_masks[j] & torch.isfinite(depths[j]) & (depths[j] > 0)]
        overlap_ij, visible_ij = _visible_overlap_ratio(
            src_world_points=src_points_i,
            dst_depth=depths[j],
            dst_valid_mask=valid_masks[j],
            dst_extrinsic=extrinsics[j],
            dst_intrinsic=intrinsics[j],
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
        )
        overlap_ji, visible_ji = _visible_overlap_ratio(
            src_world_points=src_points_j,
            dst_depth=depths[i],
            dst_valid_mask=valid_masks[i],
            dst_extrinsic=extrinsics[i],
            dst_intrinsic=intrinsics[i],
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
        )
        if (
            min(overlap_ij, overlap_ji) >= min_overlap_ratio
            and min(visible_ij, visible_ji) >= min_visible_points
        ):
            covisible_i.append(i)
            covisible_j.append(j)

    device = sample.images.device
    if not covisible_i:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return (
        torch.tensor(covisible_i, dtype=torch.long, device=device),
        torch.tensor(covisible_j, dtype=torch.long, device=device),
    )


def _relative_pose_errors(
    pred_extrinsics: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    small_translation_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_h = _to_homogeneous(pred_extrinsics)
    gt_h = _to_homogeneous(gt_extrinsics)
    pred_rel = pred_h[pair_j] @ torch.linalg.inv(pred_h[pair_i])
    gt_rel = gt_h[pair_j] @ torch.linalg.inv(gt_h[pair_i])

    rotation_errors = _rotation_error_deg(pred_rel[:, :3, :3], gt_rel[:, :3, :3])
    gt_translation = gt_rel[:, :3, 3]
    translation_valid = gt_translation.norm(dim=-1) >= small_translation_epsilon
    translation_errors = torch.full_like(rotation_errors, float("nan"))
    if translation_valid.any():
        translation_errors[translation_valid] = _translation_direction_error_deg(
            pred_rel[translation_valid, :3, 3],
            gt_translation[translation_valid],
        )
    return rotation_errors, translation_errors, translation_valid


def _pose_auc(joint_pose_errors_deg: torch.Tensor, max_threshold_deg: float) -> float:
    if joint_pose_errors_deg.numel() == 0:
        return 0.0

    joint_pose_errors_deg = joint_pose_errors_deg.to(torch.float32)
    total_count = joint_pose_errors_deg.numel()
    below_threshold = joint_pose_errors_deg[joint_pose_errors_deg < max_threshold_deg]
    if below_threshold.numel() == 0:
        return 0.0

    sorted_errors, _ = torch.sort(below_threshold)
    accuracy = torch.arange(
        1,
        sorted_errors.numel() + 1,
        device=sorted_errors.device,
        dtype=sorted_errors.dtype,
    ) / float(total_count)

    x = torch.cat([
        torch.zeros(1, device=sorted_errors.device, dtype=sorted_errors.dtype),
        sorted_errors,
        torch.tensor([max_threshold_deg], device=sorted_errors.device, dtype=sorted_errors.dtype),
    ])
    y = torch.cat([
        torch.zeros(1, device=accuracy.device, dtype=accuracy.dtype),
        accuracy,
        accuracy[-1:].clone(),
    ])
    return float((torch.trapz(y, x) / max_threshold_deg).item())


def compute_camera_metrics(predictions: Dict[str, torch.Tensor], sample: BenchmarkSample) -> Dict[str, float]:
    if sample.raw_extrinsics is None or "pose_enc" not in predictions:
        return {}

    image_hw = tuple(sample.images.shape[-2:])
    pred_pose = predictions["pose_enc"].detach()
    pred_extrinsics_norm, _ = pose_encoding_to_extri_intri(pred_pose, image_size_hw=image_hw)
    pred_extrinsics = _denormalize_extrinsics(pred_extrinsics_norm.squeeze(0), sample)
    gt_extrinsics = sample.raw_extrinsics.to(pred_extrinsics.device, dtype=pred_extrinsics.dtype)

    pair_i, pair_j = _covisible_pair_indices(sample)
    if pair_i.numel() == 0:
        return {
            "auc3": 0.0,
            "auc30": 0.0,
        }

    rotation_errors_deg, translation_direction_errors_deg, translation_valid = _relative_pose_errors(
        pred_extrinsics=pred_extrinsics,
        gt_extrinsics=gt_extrinsics,
        pair_i=pair_i.to(pred_extrinsics.device),
        pair_j=pair_j.to(pred_extrinsics.device),
        small_translation_epsilon=float(sample.protocol.get("small_translation_epsilon", 1e-4)),
    )
    joint_pose_errors_deg = rotation_errors_deg.clone()
    if translation_valid.any():
        joint_pose_errors_deg[translation_valid] = torch.maximum(
            rotation_errors_deg[translation_valid],
            translation_direction_errors_deg[translation_valid],
        )

    return {
        "auc3": _pose_auc(joint_pose_errors_deg, max_threshold_deg=3.0),
        "auc30": _pose_auc(joint_pose_errors_deg, max_threshold_deg=30.0),
    }


def _sample_seed(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _downsample_points(points: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    indices = torch.randperm(points.shape[0], generator=generator, device=points.device)[:max_points]
    return points[indices]


def _chunked_nearest_distances(src: torch.Tensor, dst: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    distances = []
    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        cdist = torch.cdist(src[start:end], dst)
        distances.append(cdist.min(dim=1).values)
    return torch.cat(distances, dim=0)


def _points_bbox_diagonal(points: torch.Tensor) -> float:
    if points.numel() == 0:
        return 0.0
    bbox_min = points.min(dim=0).values
    bbox_max = points.max(dim=0).values
    return float(torch.linalg.norm(bbox_max - bbox_min).item())


@lru_cache(maxsize=64)
def _load_mesh_vertices_faces(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return _load_ply_mesh(mesh_path)


@lru_cache(maxsize=64)
def _mesh_bbox_diagonal(mesh_path: str) -> float:
    vertices, _ = _load_mesh_vertices_faces(mesh_path)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return float(np.linalg.norm(bbox_max - bbox_min))


@lru_cache(maxsize=64)
def _sample_gt_mesh_points(mesh_path: str, num_points: int, seed: int) -> np.ndarray:
    vertices, faces = _load_mesh_vertices_faces(mesh_path)
    if faces.size == 0:
        raise ValueError(f"Mesh at {mesh_path} has no faces.")

    triangles = vertices[faces]
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_02 = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_01, edge_02), axis=-1) * 0.5
    area_sum = float(areas.sum())
    if area_sum <= 0:
        raise ValueError(f"Mesh at {mesh_path} has zero total area.")

    probabilities = areas / area_sum
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(faces), size=num_points, replace=True, p=probabilities)
    chosen = triangles[face_indices]

    u = rng.random((num_points, 1), dtype=np.float32)
    v = rng.random((num_points, 1), dtype=np.float32)
    sqrt_u = np.sqrt(u)
    bary_a = 1.0 - sqrt_u
    bary_b = sqrt_u * (1.0 - v)
    bary_c = sqrt_u * v
    sampled = bary_a * chosen[:, 0] + bary_b * chosen[:, 1] + bary_c * chosen[:, 2]
    return sampled.astype(np.float32, copy=False)


def _prepare_gt_reference_points(
    sample: BenchmarkSample,
    device: torch.device,
    dtype: torch.dtype,
    point_budget: int,
    sample_seed: int,
) -> Tuple[torch.Tensor, float]:
    if sample.gt_point_cloud is not None:
        gt_points = sample.gt_point_cloud.to(dtype=torch.float32)
        if gt_points.ndim != 2 or gt_points.shape[-1] != 3:
            raise ValueError("gt_point_cloud must have shape [N, 3].")
        bbox_diagonal = _points_bbox_diagonal(gt_points)
        gt_points = _downsample_points(gt_points, point_budget, sample_seed)
        return gt_points.to(device=device, dtype=dtype), bbox_diagonal

    if sample.gt_mesh_path is not None:
        gt_points_np = _sample_gt_mesh_points(sample.gt_mesh_path, point_budget, sample_seed)
        gt_points = torch.from_numpy(gt_points_np).to(device=device, dtype=dtype)
        bbox_diagonal = _mesh_bbox_diagonal(sample.gt_mesh_path)
        return gt_points, bbox_diagonal

    raise ValueError("Sample is missing both gt_mesh_path and gt_point_cloud.")


def _tsdf_fused_points(
    pred_depth: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    pred_intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    voxel_length: float,
    sdf_trunc: float,
    num_sample_points: int,
) -> torch.Tensor:
    import open3d as o3d

    device = pred_depth.device
    pred_depth = pred_depth.to(torch.float32)
    pred_extrinsics = pred_extrinsics.to(torch.float64)
    pred_intrinsics = pred_intrinsics.to(torch.float64)
    valid_mask = valid_mask.bool()
    height, width = pred_depth.shape[-2:]
    depth_values = pred_depth[valid_mask & torch.isfinite(pred_depth) & (pred_depth > 0)]
    if depth_values.numel() == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)

    depth_trunc = float(depth_values.max().item() + sdf_trunc)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    color_np = np.zeros((height, width, 3), dtype=np.uint8)
    color = o3d.geometry.Image(color_np)
    for frame_idx in range(pred_depth.shape[0]):
        frame_depth = pred_depth[frame_idx].clone()
        frame_valid = valid_mask[frame_idx] & torch.isfinite(frame_depth) & (frame_depth > 0)
        if frame_valid.sum().item() == 0:
            continue
        frame_depth[~frame_valid] = 0.0
        depth_np = frame_depth.detach().cpu().numpy().astype(np.float32, copy=False)
        depth = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic_np = pred_intrinsics[frame_idx].detach().cpu().numpy()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsic_np[0, 0]),
            float(intrinsic_np[1, 1]),
            float(intrinsic_np[0, 2]),
            float(intrinsic_np[1, 2]),
        )
        extrinsic_np = _to_homogeneous(pred_extrinsics[frame_idx : frame_idx + 1]).detach().cpu().numpy()[0]
        volume.integrate(rgbd, intrinsic, extrinsic_np)

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=int(num_sample_points))
    else:
        pcd = volume.extract_point_cloud()
        if len(pcd.points) > num_sample_points:
            pcd = pcd.farthest_point_down_sample(num_sample_points=int(num_sample_points))

    points_np = np.asarray(pcd.points, dtype=np.float32)
    if points_np.size == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    return torch.from_numpy(points_np).to(device=device, dtype=torch.float32)


def compute_reconstruction_metrics(predictions: Dict[str, torch.Tensor], sample: BenchmarkSample) -> Dict[str, float]:
    if (
        ("depth" not in predictions)
        or ("pose_enc" not in predictions)
        or (sample.gt_point_cloud is None and sample.gt_mesh_path is None)
    ):
        return {}

    pred_depth = predictions["depth"].detach().squeeze(0).squeeze(-1)
    image_hw = tuple(sample.images.shape[-2:])
    pred_pose = predictions["pose_enc"].detach()
    pred_extrinsics_norm, pred_intrinsics = pose_encoding_to_extri_intri(pred_pose, image_size_hw=image_hw)
    pred_extrinsics = _denormalize_extrinsics(pred_extrinsics_norm.squeeze(0), sample)
    pred_intrinsics = pred_intrinsics.squeeze(0).to(device=pred_depth.device, dtype=pred_depth.dtype)
    pred_depth = pred_depth.to(torch.float32) * float(sample.normalization_scale or 1.0)
    if sample.point_masks is not None:
        valid_mask = sample.point_masks.to(device=pred_depth.device).bool()
    elif sample.masks is not None:
        valid_mask = sample.masks.to(device=pred_depth.device) > 0.5
    else:
        valid_mask = torch.ones_like(pred_depth, dtype=torch.bool, device=pred_depth.device)

    sample_seed = _sample_seed(sample.sample_id)
    pred_point_budget = int(sample.protocol.get("pred_point_sample_points", 20000))
    gt_point_budget = int(
        sample.protocol.get(
            "gt_point_sample_points",
            sample.protocol.get("gt_mesh_sample_points", 20000),
        )
    )
    gt_points, bbox_diagonal = _prepare_gt_reference_points(
        sample=sample,
        device=pred_depth.device,
        dtype=pred_depth.dtype,
        point_budget=gt_point_budget,
        sample_seed=sample_seed,
    )
    if bbox_diagonal <= 0:
        return {}
    tsdf_resolution = int(sample.protocol.get("tsdf_resolution", 256))
    tsdf_sdf_trunc_factor = float(sample.protocol.get("tsdf_sdf_trunc_factor", 4.0))
    voxel_length = bbox_diagonal / max(tsdf_resolution, 1)
    sdf_trunc = voxel_length * tsdf_sdf_trunc_factor

    pred_points = _tsdf_fused_points(
        pred_depth=pred_depth,
        pred_extrinsics=pred_extrinsics,
        pred_intrinsics=pred_intrinsics,
        valid_mask=valid_mask,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        num_sample_points=pred_point_budget,
    )
    if pred_points.numel() == 0:
        return {}

    pred_points = _downsample_points(pred_points, pred_point_budget, sample_seed)
    gt_points = gt_points.to(device=pred_points.device, dtype=pred_points.dtype)

    pred_to_gt = _chunked_nearest_distances(pred_points, gt_points)
    gt_to_pred = _chunked_nearest_distances(gt_points, pred_points)

    chamfer_l1 = 0.5 * (pred_to_gt.mean() + gt_to_pred.mean()) / bbox_diagonal

    def compute_f1(threshold_ratio: float) -> float:
        threshold = bbox_diagonal * threshold_ratio
        precision = (pred_to_gt <= threshold).to(torch.float32).mean()
        recall = (gt_to_pred <= threshold).to(torch.float32).mean()
        if (precision + recall).item() == 0:
            return 0.0
        return float((2 * precision * recall / (precision + recall)).item())

    return {
        "cd_l1": float(chamfer_l1.item()),
        "f1@1%": compute_f1(0.01),
        "f1@5%": compute_f1(0.05),
    }


def compute_depth_metrics(predictions: Dict[str, torch.Tensor], sample: BenchmarkSample) -> Dict[str, float]:
    if sample.raw_depths is None or sample.point_masks is None or "depth" not in predictions:
        return {}

    pred_depth = predictions["depth"].detach().squeeze(0).squeeze(-1) * float(sample.normalization_scale or 1.0)
    gt_depth = sample.raw_depths.to(device=pred_depth.device, dtype=pred_depth.dtype)
    valid_mask = sample.point_masks.to(device=pred_depth.device).bool() & torch.isfinite(gt_depth) & (gt_depth > 0)
    if valid_mask.sum().item() == 0:
        return {}

    pred_valid = pred_depth[valid_mask].clamp(min=1e-6)
    gt_valid = gt_depth[valid_mask].clamp(min=1e-6)
    ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    abs_diff = (pred_valid - gt_valid).abs()

    return {
        "delta1": float((ratio < 1.25).to(torch.float32).mean().item()),
        "absrel": float((abs_diff / gt_valid).mean().item()),
    }


def compute_sample_metrics(predictions: Dict[str, torch.Tensor], sample: BenchmarkSample) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(compute_camera_metrics(predictions, sample))
    metrics.update(compute_reconstruction_metrics(predictions, sample))
    metrics.update(compute_depth_metrics(predictions, sample))
    return metrics
