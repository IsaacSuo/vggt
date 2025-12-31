# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visual Hull computation utilities.

Visual Hull is the maximal shape consistent with a set of silhouettes.
It provides an outer bound on the object's geometry, useful for:
1. Providing a "safety layer" point cloud on smooth surfaces
2. Filtering out points that violate silhouette consistency
3. Hybrid initialization with depth-based point clouds
"""

import logging
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_visual_hull_points(
    masks: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: int = 128,
    threshold: float = 0.9,
    bbox_scale: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Visual Hull points from multi-view silhouettes.

    Uses voxel carving: creates a 3D voxel grid and marks voxels as occupied
    if they project inside the silhouette in all views.

    Args:
        masks: Binary masks [S, H, W] where 1=foreground, 0=background
        extrinsics: Camera extrinsics [S, 4, 4] (world to camera transform)
        intrinsics: Camera intrinsics [S, 3, 3]
        resolution: Voxel grid resolution (resolution^3 voxels)
        threshold: Fraction of views that must agree for a voxel to be valid
        bbox_scale: Scale factor for bounding box estimation

    Returns:
        points: Visual Hull surface points [N, 3] in world coordinates
        occupancy: Voxel occupancy grid [resolution, resolution, resolution]
    """
    device = masks.device
    S, H, W = masks.shape

    # Estimate bounding box from depth predictions or use default
    # For now, use a simple heuristic based on camera positions
    camera_centers = get_camera_centers(extrinsics)  # [S, 3]

    # Compute bounding box that encompasses all cameras with margin
    center = camera_centers.mean(dim=0)
    max_dist = (camera_centers - center).norm(dim=1).max()
    bbox_size = max_dist * bbox_scale

    # Create voxel grid
    x = torch.linspace(-bbox_size/2, bbox_size/2, resolution, device=device)
    y = torch.linspace(-bbox_size/2, bbox_size/2, resolution, device=device)
    z = torch.linspace(-bbox_size/2, bbox_size/2, resolution, device=device)

    # Offset to scene center
    xx, yy, zz = torch.meshgrid(x + center[0], y + center[1], z + center[2], indexing='ij')
    voxel_centers = torch.stack([xx, yy, zz], dim=-1)  # [R, R, R, 3]

    # Flatten for projection
    voxel_centers_flat = voxel_centers.reshape(-1, 3)  # [R^3, 3]
    num_voxels = voxel_centers_flat.shape[0]

    # Count how many views each voxel projects inside the silhouette
    vote_count = torch.zeros(num_voxels, device=device)

    for view_idx in range(S):
        # Project voxels to this view
        ext = extrinsics[view_idx]  # [4, 4]
        intr = intrinsics[view_idx]  # [3, 3]
        mask = masks[view_idx]  # [H, W]

        # World to camera: p_cam = R @ p_world + t
        R = ext[:3, :3]  # [3, 3]
        t = ext[:3, 3]   # [3]

        # Transform to camera coordinates
        p_cam = (R @ voxel_centers_flat.T).T + t  # [N, 3]

        # Project to image plane
        p_img = (intr @ p_cam.T).T  # [N, 3]

        # Normalize by depth
        depth = p_img[:, 2:3]
        p_2d = p_img[:, :2] / (depth + 1e-8)  # [N, 2]

        # Check which voxels are in front of camera
        valid_depth = depth.squeeze() > 0.1

        # Check which voxels project inside image bounds
        in_bounds = (
            (p_2d[:, 0] >= 0) & (p_2d[:, 0] < W) &
            (p_2d[:, 1] >= 0) & (p_2d[:, 1] < H)
        )

        # Sample mask at projected locations
        valid = valid_depth & in_bounds
        p_2d_valid = p_2d[valid]

        # Normalize to [-1, 1] for grid_sample
        p_2d_norm = torch.zeros_like(p_2d_valid)
        p_2d_norm[:, 0] = 2 * p_2d_valid[:, 0] / (W - 1) - 1
        p_2d_norm[:, 1] = 2 * p_2d_valid[:, 1] / (H - 1) - 1

        # Sample mask values
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid = p_2d_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, N_valid, 2]
        sampled = F.grid_sample(
            mask_expanded,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )  # [1, 1, 1, N_valid]
        sampled = sampled.squeeze()  # [N_valid]

        # Vote: count if inside silhouette
        votes = torch.zeros(num_voxels, device=device)
        votes[valid] = (sampled > 0.5).float()
        vote_count += votes

    # Voxels are occupied if they project inside silhouette in most views
    occupancy = (vote_count / S) >= threshold
    occupancy = occupancy.reshape(resolution, resolution, resolution)

    # Extract surface points (voxels on the boundary)
    surface_mask = extract_surface_voxels(occupancy)
    surface_points = voxel_centers[surface_mask]

    return surface_points, occupancy


def extract_surface_voxels(occupancy: torch.Tensor) -> torch.Tensor:
    """
    Extract surface voxels from an occupancy grid.

    Surface voxels are occupied voxels that have at least one empty neighbor.

    Args:
        occupancy: Binary occupancy grid [R, R, R]

    Returns:
        surface_mask: Boolean mask [R, R, R] indicating surface voxels
    """
    # Pad occupancy grid
    padded = F.pad(occupancy.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), value=0)

    # Count neighbors using 3D convolution
    kernel = torch.ones(1, 1, 3, 3, 3, device=occupancy.device)
    kernel[0, 0, 1, 1, 1] = 0  # Don't count self
    neighbor_count = F.conv3d(padded, kernel, padding=0).squeeze()

    # Surface voxels: occupied AND have less than 26 occupied neighbors
    max_neighbors = 26  # 3x3x3 - 1
    surface_mask = occupancy & (neighbor_count < max_neighbors * occupancy.float())

    return surface_mask.bool()


def get_camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Get camera centers in world coordinates from extrinsic matrices.

    Args:
        extrinsics: Camera extrinsics [S, 4, 4] (world to camera)

    Returns:
        centers: Camera centers [S, 3] in world coordinates
    """
    # For world-to-camera transform: p_cam = R @ p_world + t
    # Camera center: R @ center + t = 0 => center = -R^T @ t
    R = extrinsics[:, :3, :3]  # [S, 3, 3]
    t = extrinsics[:, :3, 3]   # [S, 3]

    centers = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
    return centers


def masks_to_point_cloud(
    masks: torch.Tensor,
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    min_confidence: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert masked depth maps to 3D point cloud.

    Only includes points that are inside the mask and above confidence threshold.

    Args:
        masks: Binary masks [S, H, W]
        depth: Depth maps [S, H, W] or [S, 1, H, W]
        extrinsics: Camera extrinsics [S, 4, 4]
        intrinsics: Camera intrinsics [S, 3, 3]
        confidence: Optional confidence maps [S, H, W]
        min_confidence: Minimum confidence threshold

    Returns:
        points: 3D points [N, 3] in world coordinates
        colors: Optional RGB colors [N, 3] if images provided
    """
    S = masks.shape[0]
    device = masks.device

    if len(depth.shape) == 4:
        depth = depth.squeeze(1)  # [S, H, W]

    H, W = depth.shape[1:]

    all_points = []

    for view_idx in range(S):
        mask = masks[view_idx]  # [H, W]
        d = depth[view_idx]     # [H, W]
        ext = extrinsics[view_idx]  # [4, 4]
        intr = intrinsics[view_idx]  # [3, 3]

        # Apply confidence threshold
        valid = mask > 0.5
        if confidence is not None:
            valid = valid & (confidence[view_idx] >= min_confidence)

        # Get valid pixel coordinates
        v, u = torch.where(valid)  # [N], [N]
        z = d[valid]  # [N]

        if len(z) == 0:
            continue

        # Unproject to camera coordinates
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]

        x_cam = (u.float() - cx) * z / fx
        y_cam = (v.float() - cy) * z / fy
        z_cam = z

        p_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)  # [N, 3]

        # Transform to world coordinates
        R = ext[:3, :3]  # [3, 3]
        t = ext[:3, 3]   # [3]

        # p_cam = R @ p_world + t => p_world = R^T @ (p_cam - t)
        p_world = (R.T @ (p_cam - t).T).T  # [N, 3]

        all_points.append(p_world)

    if not all_points:
        return torch.empty(0, 3, device=device), torch.empty(0, 3, device=device)

    points = torch.cat(all_points, dim=0)
    colors = torch.ones_like(points)  # Placeholder colors

    return points, colors


def sample_visual_hull_surface(
    occupancy: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    num_points: int = 10000,
) -> torch.Tensor:
    """
    Sample points on the Visual Hull surface.

    Uses marching cubes-like approach to extract surface, then samples points.

    Args:
        occupancy: Voxel occupancy [R, R, R]
        bbox_min: Minimum corner of bounding box [3]
        bbox_max: Maximum corner of bounding box [3]
        num_points: Number of points to sample

    Returns:
        points: Surface points [num_points, 3]
    """
    device = occupancy.device
    R = occupancy.shape[0]

    # Extract surface voxels
    surface_mask = extract_surface_voxels(occupancy)

    # Get voxel indices
    surface_indices = torch.where(surface_mask)
    num_surface_voxels = surface_indices[0].shape[0]

    if num_surface_voxels == 0:
        logger.warning("No surface voxels found")
        return torch.empty(0, 3, device=device)

    # Convert indices to world coordinates
    voxel_size = (bbox_max - bbox_min) / R

    # Sample voxels (with replacement if needed)
    if num_surface_voxels < num_points:
        sample_idx = torch.randint(0, num_surface_voxels, (num_points,), device=device)
    else:
        sample_idx = torch.randperm(num_surface_voxels, device=device)[:num_points]

    # Get voxel centers
    i = surface_indices[0][sample_idx].float()
    j = surface_indices[1][sample_idx].float()
    k = surface_indices[2][sample_idx].float()

    # Add random offset within voxel
    offset = torch.rand(num_points, 3, device=device) - 0.5

    points = torch.stack([i, j, k], dim=1) + offset
    points = bbox_min + points * voxel_size

    return points


def hybrid_point_cloud(
    vggt_points: torch.Tensor,
    visual_hull_points: torch.Tensor,
    vggt_confidence: Optional[torch.Tensor] = None,
    visual_hull_weight: float = 0.3,
) -> torch.Tensor:
    """
    Create hybrid point cloud by combining VGGT predictions with Visual Hull.

    P_init = P_VGGT ∪ P_VisualHull

    Args:
        vggt_points: Points from VGGT [N1, 3]
        visual_hull_points: Points from Visual Hull [N2, 3]
        vggt_confidence: Optional confidence for VGGT points [N1]
        visual_hull_weight: Weight for Visual Hull points in final cloud

    Returns:
        combined_points: Union of both point sets [N1 + N2, 3]
    """
    # Filter VGGT points by confidence if provided
    if vggt_confidence is not None:
        valid = vggt_confidence > 0.5
        vggt_points = vggt_points[valid]

    # Subsample Visual Hull points based on weight
    num_vh_points = int(visual_hull_points.shape[0] * visual_hull_weight)
    if num_vh_points > 0 and num_vh_points < visual_hull_points.shape[0]:
        idx = torch.randperm(visual_hull_points.shape[0], device=visual_hull_points.device)[:num_vh_points]
        visual_hull_points = visual_hull_points[idx]

    # Combine
    combined = torch.cat([vggt_points, visual_hull_points], dim=0)

    return combined
