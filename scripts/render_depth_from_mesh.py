#!/usr/bin/env python3
"""
从GT Mesh渲染深度图

使用方法:
    python scripts/render_depth_from_mesh.py \
        --mesh_dir datasets/groundtruth \
        --data_dir datasets/openmaterial \
        --output_dir datasets/openmaterial_depth \
        --scene_id 5c4ae9c4a3cb47a4b6273eb2839a7b8c

或批量处理:
    python scripts/render_depth_from_mesh.py \
        --mesh_dir datasets/groundtruth \
        --data_dir datasets/openmaterial \
        --output_dir datasets/openmaterial_depth \
        --all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_mesh(mesh_path: str):
    """
    加载PLY mesh

    Args:
        mesh_path: PLY文件路径

    Returns:
        trimesh.Trimesh 对象
    """
    import trimesh

    print(f"[Mesh] Loading: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')

    print(f"[Mesh] Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"[Mesh] Bounds: {mesh.bounds}")

    return mesh


def load_transforms(json_path: str) -> dict:
    """
    加载NeRF格式的transforms.json

    Args:
        json_path: transforms_train.json 或 transforms_test.json 路径

    Returns:
        解析后的相机参数字典
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"[Camera] Loaded {len(data['frames'])} frames from {json_path}")
    print(f"[Camera] Image size: {data['w']} x {data['h']}")
    print(f"[Camera] Focal: fx={data['fl_x']:.2f}, fy={data['fl_y']:.2f}")

    return data


def nerf_c2w_to_opencv_w2c(c2w_nerf: np.ndarray) -> np.ndarray:
    """
    将NeRF的camera-to-world矩阵转换为OpenCV的world-to-camera矩阵

    NeRF/OpenGL约定: Y向上, Z向后 (相机看向-Z)
    OpenCV约定: Y向下, Z向前 (相机看向+Z)

    Args:
        c2w_nerf: (4, 4) NeRF格式的camera-to-world矩阵

    Returns:
        w2c_opencv: (3, 4) OpenCV格式的world-to-camera矩阵
    """
    # 1. OpenGL -> OpenCV: 翻转Y和Z
    # 变换矩阵
    flip = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)

    c2w_opencv = c2w_nerf @ flip

    # 2. camera-to-world -> world-to-camera: 求逆
    w2c_opencv = np.linalg.inv(c2w_opencv)

    return w2c_opencv[:3, :]


def get_camera_rays(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成相机光线

    Args:
        width, height: 图像尺寸
        fx, fy, cx, cy: 相机内参
        c2w: (4, 4) camera-to-world矩阵 (NeRF格式)

    Returns:
        ray_origins: (H*W, 3) 光线起点
        ray_directions: (H*W, 3) 光线方向 (单位向量)
    """
    # 生成像素网格
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 像素坐标 -> 相机坐标系下的方向
    # 注意: NeRF相机看向-Z
    dirs_cam = np.stack([
        (u - cx) / fx,
        -(v - cy) / fy,  # NeRF Y轴翻转
        -np.ones_like(u)  # 看向-Z
    ], axis=-1)  # (H, W, 3)

    # 归一化
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # 相机坐标系 -> 世界坐标系
    R = c2w[:3, :3]
    t = c2w[:3, 3]

    # 旋转方向向量
    dirs_world = dirs_cam @ R.T  # (H, W, 3)

    # 光线起点 (相机位置)
    origins = np.broadcast_to(t, dirs_world.shape)

    # 展平
    ray_origins = origins.reshape(-1, 3)
    ray_directions = dirs_world.reshape(-1, 3)

    return ray_origins, ray_directions


def render_depth_raycast(
    mesh,
    c2w: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float = 100.0
) -> np.ndarray:
    """
    使用光线追踪渲染深度图

    Args:
        mesh: trimesh.Trimesh 对象
        c2w: (4, 4) camera-to-world矩阵 (NeRF格式)
        width, height: 图像尺寸
        fx, fy, cx, cy: 相机内参
        max_depth: 最大深度值

    Returns:
        depth: (H, W) 深度图
    """
    import trimesh

    # 生成光线
    ray_origins, ray_directions = get_camera_rays(
        width, height, fx, fy, cx, cy, c2w
    )

    # 创建RayMeshIntersector
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # 批量光线求交
    # 返回: locations, index_ray, index_tri
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False
    )

    # 计算深度
    depth = np.full((height * width,), max_depth, dtype=np.float32)

    if len(locations) > 0:
        # 计算每条光线的深度 (沿光线方向的距离)
        hit_origins = ray_origins[index_ray]
        hit_distances = np.linalg.norm(locations - hit_origins, axis=-1)
        depth[index_ray] = hit_distances

    # 重塑为图像
    depth = depth.reshape(height, width)

    return depth


def render_depth_for_scene(
    mesh_path: str,
    transforms_path: str,
    output_dir: str,
    split: str = "train",
    downsample: int = 1,
    max_depth: float = 100.0
) -> List[str]:
    """
    为一个场景渲染所有帧的深度图

    Args:
        mesh_path: GT mesh路径
        transforms_path: transforms.json路径
        output_dir: 输出目录
        split: "train" 或 "test"
        downsample: 下采样因子 (1=原始分辨率, 2=1/2分辨率)
        max_depth: 最大深度值

    Returns:
        生成的深度图路径列表
    """
    # 加载mesh
    mesh = load_mesh(mesh_path)

    # 加载相机参数
    transforms = load_transforms(transforms_path)

    # 提取相机内参
    width = transforms['w'] // downsample
    height = transforms['h'] // downsample
    fx = transforms['fl_x'] / downsample
    fy = transforms['fl_y'] / downsample
    cx = transforms['cx'] / downsample
    cy = transforms['cy'] / downsample

    # 创建输出目录
    depth_output_dir = Path(output_dir) / split / "depths"
    depth_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    # 渲染每一帧
    frames = transforms['frames']
    print(f"\n[Render] Rendering {len(frames)} frames at {width}x{height}...")

    for frame in tqdm(frames, desc=f"Rendering {split}"):
        # 提取帧信息
        file_path = frame['file_path']
        c2w = np.array(frame['transform_matrix'], dtype=np.float64)

        # 从file_path提取帧编号
        # 例如: "train/images/000.png" -> "000"
        frame_name = Path(file_path).stem

        # 渲染深度
        depth = render_depth_raycast(
            mesh, c2w, width, height, fx, fy, cx, cy, max_depth
        )

        # 保存深度图
        depth_path = depth_output_dir / f"{frame_name}.npy"
        np.save(depth_path, depth.astype(np.float32))
        output_paths.append(str(depth_path))

        # 可选: 保存可视化PNG
        depth_vis_path = depth_output_dir / f"{frame_name}_vis.png"
        save_depth_visualization(depth, depth_vis_path, max_depth)

    print(f"[Render] Saved {len(output_paths)} depth maps to {depth_output_dir}")

    return output_paths


def save_depth_visualization(
    depth: np.ndarray,
    output_path: str,
    max_depth: float = 100.0
):
    """
    保存深度图的可视化

    Args:
        depth: (H, W) 深度图
        output_path: 输出路径
        max_depth: 最大深度值 (用于归一化)
    """
    import cv2

    # 归一化到0-255
    depth_vis = depth.copy()
    depth_vis[depth_vis >= max_depth] = 0  # 无效区域设为0

    valid_mask = depth_vis > 0
    if valid_mask.any():
        min_d = depth_vis[valid_mask].min()
        max_d = depth_vis[valid_mask].max()
        depth_vis[valid_mask] = (depth_vis[valid_mask] - min_d) / (max_d - min_d + 1e-6) * 255

    depth_vis = depth_vis.astype(np.uint8)

    # 应用colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

    # 无效区域设为黑色
    depth_colored[~valid_mask] = 0

    cv2.imwrite(str(output_path), depth_colored)


def process_scene(
    scene_id: str,
    mesh_dir: str,
    data_dir: str,
    output_dir: str,
    splits: List[str] = ["train", "test"],
    downsample: int = 1
):
    """
    处理单个场景

    Args:
        scene_id: 场景ID
        mesh_dir: groundtruth目录
        data_dir: openmaterial目录
        output_dir: 输出目录
        splits: 要处理的splits
        downsample: 下采样因子
    """
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_id}")
    print(f"{'='*60}")

    # 构建路径
    mesh_path = Path(mesh_dir) / scene_id / f"clean_{scene_id}.ply"
    scene_data_dir = Path(data_dir) / scene_id
    scene_output_dir = Path(output_dir) / scene_id

    # 检查mesh是否存在
    if not mesh_path.exists():
        print(f"[Error] Mesh not found: {mesh_path}")
        return

    # 处理每个split
    for split in splits:
        transforms_path = scene_data_dir / f"transforms_{split}.json"

        if not transforms_path.exists():
            print(f"[Warning] Transforms not found: {transforms_path}")
            continue

        render_depth_for_scene(
            mesh_path=str(mesh_path),
            transforms_path=str(transforms_path),
            output_dir=str(scene_output_dir),
            split=split,
            downsample=downsample
        )


def find_all_scenes(mesh_dir: str, data_dir: str) -> List[str]:
    """
    查找所有有效的场景ID

    Args:
        mesh_dir: groundtruth目录
        data_dir: openmaterial目录

    Returns:
        场景ID列表
    """
    mesh_dir = Path(mesh_dir)
    data_dir = Path(data_dir)

    # 获取所有mesh目录
    mesh_scenes = set()
    if mesh_dir.exists():
        for d in mesh_dir.iterdir():
            if d.is_dir():
                mesh_scenes.add(d.name)

    # 获取所有数据目录
    data_scenes = set()
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and (d / "transforms_train.json").exists():
                data_scenes.add(d.name)

    # 交集
    valid_scenes = sorted(mesh_scenes & data_scenes)

    print(f"[Discovery] Found {len(mesh_scenes)} mesh scenes")
    print(f"[Discovery] Found {len(data_scenes)} data scenes")
    print(f"[Discovery] Valid scenes (intersection): {len(valid_scenes)}")

    return valid_scenes


def main():
    parser = argparse.ArgumentParser(description="Render depth maps from GT mesh")

    parser.add_argument("--mesh_dir", type=str, required=True,
                        help="Path to groundtruth directory containing PLY files")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to openmaterial directory containing transforms.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for depth maps")

    parser.add_argument("--scene_id", type=str, default=None,
                        help="Process single scene by ID")
    parser.add_argument("--all", action="store_true",
                        help="Process all available scenes")

    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"],
                        help="Splits to process (default: train test)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor (default: 1, no downsampling)")

    args = parser.parse_args()

    # 验证参数
    if not args.scene_id and not args.all:
        parser.error("Must specify --scene_id or --all")

    if args.all:
        # 处理所有场景
        scenes = find_all_scenes(args.mesh_dir, args.data_dir)

        if not scenes:
            print("[Error] No valid scenes found!")
            return

        for scene_id in scenes:
            process_scene(
                scene_id=scene_id,
                mesh_dir=args.mesh_dir,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                splits=args.splits,
                downsample=args.downsample
            )
    else:
        # 处理单个场景
        process_scene(
            scene_id=args.scene_id,
            mesh_dir=args.mesh_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            splits=args.splits,
            downsample=args.downsample
        )

    print("\n[Done] Depth rendering complete!")


if __name__ == "__main__":
    main()
