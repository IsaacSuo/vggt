# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenMaterial Dataset for Phong Training

数据集结构:
    datasets/openmaterial/{scene_id}/
    ├── train/
    │   ├── images/*.png      # RGB图像
    │   └── depths/*.npy      # 预渲染深度图 (可选)
    ├── test/
    │   ├── images/*.png
    │   └── depths/*.npy
    ├── mask/                  # 物体掩码
    ├── transforms_train.json
    └── transforms_test.json
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OpenMaterialDataset(Dataset):
    """
    OpenMaterial 数据集加载器

    特点:
    - 支持NeRF格式的相机参数 (自动转换为OpenCV格式)
    - 支持预渲染的GT深度图
    - 支持物体掩码
    - 兼容VGGT训练接口
    """

    def __init__(
        self,
        data_dir: str,
        scene_ids: Optional[List[str]] = None,
        split: str = "train",
        img_size: int = 518,
        num_frames: int = 4,
        load_depth: bool = True,
        load_mask: bool = True,
        random_sample: bool = True,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: openmaterial数据集根目录
            scene_ids: 要加载的场景ID列表 (None=所有场景)
            split: "train" 或 "test"
            img_size: 输出图像大小 (正方形)
            num_frames: 每个样本的帧数
            load_depth: 是否加载深度图
            load_mask: 是否加载物体掩码
            random_sample: 是否随机采样帧
            augment: 是否进行数据增强
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.num_frames = num_frames
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.random_sample = random_sample
        self.augment = augment

        # 发现所有场景
        if scene_ids is None:
            scene_ids = self._discover_scenes()

        self.scene_ids = scene_ids
        print(f"[OpenMaterialDataset] Found {len(self.scene_ids)} scenes for {split}")

        # 加载所有场景的元数据
        self.scenes_data = {}
        self._load_all_scenes()

        # 创建索引 (scene_id, frame_indices)
        self._build_index()

        print(f"[OpenMaterialDataset] Total samples: {len(self)}")

    def _discover_scenes(self) -> List[str]:
        """发现所有有效的场景"""
        scenes = []
        for d in self.data_dir.iterdir():
            if d.is_dir():
                transforms_path = d / f"transforms_{self.split}.json"
                if transforms_path.exists():
                    scenes.append(d.name)
        return sorted(scenes)

    def _load_all_scenes(self):
        """加载所有场景的元数据"""
        for scene_id in self.scene_ids:
            scene_dir = self.data_dir / scene_id
            transforms_path = scene_dir / f"transforms_{self.split}.json"

            with open(transforms_path, 'r') as f:
                transforms = json.load(f)

            # 验证帧数据
            valid_frames = []
            for frame in transforms['frames']:
                img_path = scene_dir / frame['file_path']
                if img_path.exists():
                    valid_frames.append(frame)

            if len(valid_frames) < self.num_frames:
                print(f"[Warning] Scene {scene_id} has only {len(valid_frames)} frames, skipping")
                continue

            self.scenes_data[scene_id] = {
                'transforms': transforms,
                'frames': valid_frames,
                'scene_dir': scene_dir,
            }

    def _build_index(self):
        """构建数据索引"""
        self.samples = []

        for scene_id, data in self.scenes_data.items():
            num_frames_total = len(data['frames'])

            if self.random_sample:
                # 随机采样模式: 每个场景作为一个样本
                # 实际帧在__getitem__时随机选择
                num_samples = max(1, num_frames_total // self.num_frames)
                for _ in range(num_samples):
                    self.samples.append((scene_id, None))
            else:
                # 顺序采样模式: 滑动窗口
                for start_idx in range(0, num_frames_total - self.num_frames + 1, self.num_frames):
                    frame_indices = list(range(start_idx, start_idx + self.num_frames))
                    self.samples.append((scene_id, frame_indices))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            dict:
                images: (S, 3, H, W) RGB图像 [0, 1]
                depths: (S, H, W) 深度图 (如果load_depth=True)
                masks: (S, H, W) 物体掩码 (如果load_mask=True)
                extrinsics: (S, 3, 4) world-to-camera (OpenCV)
                intrinsics: (S, 3, 3) 相机内参
                scene_id: 场景ID
        """
        scene_id, frame_indices = self.samples[idx]
        data = self.scenes_data[scene_id]

        # 选择帧索引
        if frame_indices is None:
            # 随机采样
            all_indices = list(range(len(data['frames'])))
            frame_indices = sorted(random.sample(all_indices, self.num_frames))

        # 加载数据
        images = []
        depths = []
        masks = []
        extrinsics = []
        intrinsics = []

        transforms = data['transforms']
        scene_dir = data['scene_dir']

        # 相机内参 (全局)
        K = self._build_intrinsic_matrix(transforms)
        orig_w, orig_h = transforms['w'], transforms['h']

        for frame_idx in frame_indices:
            frame = data['frames'][frame_idx]

            # 加载图像
            img_path = scene_dir / frame['file_path']
            img = self._load_image(img_path)
            images.append(img)

            # 加载深度
            if self.load_depth:
                depth = self._load_depth(scene_dir, frame, orig_w, orig_h)
                depths.append(depth)

            # 加载掩码
            if self.load_mask:
                mask = self._load_mask(scene_dir, frame, orig_w, orig_h)
                masks.append(mask)

            # 相机外参
            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            w2c = self._nerf_c2w_to_opencv_w2c(c2w)
            extrinsics.append(w2c)

            # 内参 (需要根据resize调整)
            K_scaled = self._scale_intrinsic(K, transforms['w'], transforms['h'])
            intrinsics.append(K_scaled)

        # 转换为tensor
        images = torch.stack([torch.from_numpy(img) for img in images])  # (S, 3, H, W)
        extrinsics = torch.from_numpy(np.stack(extrinsics))  # (S, 3, 4)
        intrinsics = torch.from_numpy(np.stack(intrinsics))  # (S, 3, 3)

        result = {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'scene_id': scene_id,
        }

        if self.load_depth and depths:
            result['depths'] = torch.stack([torch.from_numpy(d) for d in depths])

        if self.load_mask and masks:
            result['masks'] = torch.stack([torch.from_numpy(m) for m in masks])

        return result

    def _build_intrinsic_matrix(self, transforms: dict) -> np.ndarray:
        """构建相机内参矩阵"""
        fx = transforms['fl_x']
        fy = transforms['fl_y']
        cx = transforms['cx']
        cy = transforms['cy']

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        return K

    def _scale_intrinsic(
        self,
        K: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> np.ndarray:
        """
        根据resize和中心裁剪调整内参

        处理流程:
        1. 按短边缩放到img_size (保持宽高比)
        2. 中心裁剪到img_size x img_size
        """
        # 计算缩放比例 (按短边)
        short_side = min(orig_w, orig_h)
        scale = self.img_size / short_side

        K_scaled = K.copy()
        K_scaled[0, :] *= scale  # fx, cx
        K_scaled[1, :] *= scale  # fy, cy

        # 计算中心裁剪的偏移
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        crop_x = (new_w - self.img_size) // 2
        crop_y = (new_h - self.img_size) // 2

        # 调整主点
        K_scaled[0, 2] -= crop_x  # cx
        K_scaled[1, 2] -= crop_y  # cy

        return K_scaled

    def _nerf_c2w_to_opencv_w2c(self, c2w: np.ndarray) -> np.ndarray:
        """
        将NeRF的camera-to-world转换为OpenCV的world-to-camera

        NeRF/OpenGL: Y向上, Z向后
        OpenCV: Y向下, Z向前
        """
        # OpenGL -> OpenCV: 翻转Y和Z
        flip = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float32)

        c2w_4x4 = np.eye(4, dtype=np.float32)
        c2w_4x4[:4, :4] = c2w

        c2w_opencv = c2w_4x4 @ flip

        # camera-to-world -> world-to-camera
        w2c_opencv = np.linalg.inv(c2w_opencv)

        return w2c_opencv[:3, :].astype(np.float32)

    def _load_image(self, path: Path) -> np.ndarray:
        """
        加载并预处理图像

        处理流程:
        1. 按短边缩放到img_size (保持宽高比)
        2. 中心裁剪到img_size x img_size

        Returns:
            img: (3, H, W) float32 [0, 1]
        """
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # 按短边缩放
        short_side = min(w, h)
        scale = self.img_size / short_side

        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 中心裁剪
        crop_x = (new_w - self.img_size) // 2
        crop_y = (new_h - self.img_size) // 2
        img = img[crop_y:crop_y + self.img_size, crop_x:crop_x + self.img_size]

        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        return img

    def _load_depth(self, scene_dir: Path, frame: dict, orig_w: int, orig_h: int) -> np.ndarray:
        """
        加载深度图

        处理流程与图像一致:
        1. 按短边缩放
        2. 中心裁剪

        Returns:
            depth: (H, W) float32
        """
        # 从file_path推断深度路径
        # "train/images/000.png" -> "train/depths/000.npy"
        file_path = Path(frame['file_path'])
        frame_name = file_path.stem
        split_name = file_path.parts[0]  # "train" or "test"

        depth_path = scene_dir / split_name / "depths" / f"{frame_name}.npy"

        if depth_path.exists():
            depth = np.load(depth_path)

            h, w = depth.shape[:2]

            # 按短边缩放
            short_side = min(w, h)
            scale = self.img_size / short_side

            new_w = int(w * scale)
            new_h = int(h * scale)
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # 中心裁剪
            crop_x = (new_w - self.img_size) // 2
            crop_y = (new_h - self.img_size) // 2
            depth = depth[crop_y:crop_y + self.img_size, crop_x:crop_x + self.img_size]
        else:
            # 深度图不存在，返回零深度
            depth = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        return depth.astype(np.float32)

    def _load_mask(self, scene_dir: Path, frame: dict, orig_w: int, orig_h: int) -> np.ndarray:
        """
        加载物体掩码

        处理流程与图像一致:
        1. 按短边缩放
        2. 中心裁剪

        Returns:
            mask: (H, W) float32, 1=物体, 0=背景
        """
        # 从file_path推断掩码路径
        # "train/images/000.png" -> "mask/000.png"
        file_path = Path(frame['file_path'])
        frame_name = file_path.stem

        # 尝试多种可能的掩码路径
        mask_paths = [
            scene_dir / "mask" / f"{frame_name}.png",
            scene_dir / "masks" / f"{frame_name}.png",
            scene_dir / frame['file_path'].replace('/images/', '/mask/'),
        ]

        mask = None
        for mask_path in mask_paths:
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            # 掩码不存在，返回全1
            mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
        else:
            h, w = mask.shape[:2]

            # 按短边缩放
            short_side = min(w, h)
            scale = self.img_size / short_side

            new_w = int(w * scale)
            new_h = int(h * scale)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # 中心裁剪
            crop_x = (new_w - self.img_size) // 2
            crop_y = (new_h - self.img_size) // 2
            mask = mask[crop_y:crop_y + self.img_size, crop_x:crop_x + self.img_size]

            # 归一化到[0, 1]
            mask = (mask > 127).astype(np.float32)

        return mask


def create_openmaterial_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 2,
    num_frames: int = 4,
    img_size: int = 518,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    创建OpenMaterial数据加载器

    Args:
        data_dir: 数据集目录
        split: "train" 或 "test"
        batch_size: 批次大小
        num_frames: 每个样本的帧数
        img_size: 图像大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱

    Returns:
        DataLoader
    """
    dataset = OpenMaterialDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        num_frames=num_frames,
        **kwargs
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    dataset = OpenMaterialDataset(
        data_dir=args.data_dir,
        split=args.split,
        img_size=518,
        num_frames=4,
    )

    print(f"\nDataset size: {len(dataset)}")

    # 测试加载
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"Extrinsics shape: {sample['extrinsics'].shape}")
    print(f"Intrinsics shape: {sample['intrinsics'].shape}")

    if 'depths' in sample:
        print(f"Depths shape: {sample['depths'].shape}")
    if 'masks' in sample:
        print(f"Masks shape: {sample['masks'].shape}")
