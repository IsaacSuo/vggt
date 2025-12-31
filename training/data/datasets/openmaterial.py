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
from glob import glob
from typing import List, Optional

import cv2
import numpy as np

from data.dataset_util import read_image_cv2
from data.base_dataset import BaseDataset


logger = logging.getLogger(__name__)


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
        self.load_mask = getattr(common_conf, 'load_mask', load_mask)

        if data_dir is None:
            raise ValueError("data_dir must be specified.")

        self.data_dir = data_dir
        self.split = split
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test

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
                    'dir': scene_dir,
                    'transform_path': tf_path,
                    'frames': frames,
                    'intrinsics': intrinsics,
                })

            except Exception as e:
                logger.warning(f"Failed to load {tf_path}: {e}")
                continue

        return scenes

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
                mask_path = image_path.replace('/images/', '/masks/')
                if osp.exists(mask_path):
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

            # No depth in this dataset
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            # Process image
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

            # Resize mask to match processed image
            if mask is not None:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

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
