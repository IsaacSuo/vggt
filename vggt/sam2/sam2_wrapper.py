# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM 2 (Segment Anything Model 2) wrapper for video mask propagation.

This module provides utilities to use SAM 2's video predictor mode for
generating temporally consistent masks across multiple views.

Requirements:
    pip install segment-anything-2
    # or install from: https://github.com/facebookresearch/segment-anything-2
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import SAM2
try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor as _SAM2VideoPredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning(
        "SAM2 not installed. Install with: pip install segment-anything-2 "
        "or from https://github.com/facebookresearch/segment-anything-2"
    )


class SAM2VideoPredictor:
    """
    Wrapper for SAM2 video predictor that handles multi-view mask propagation.

    This class simplifies the process of:
    1. Loading a SAM2 video predictor model
    2. Providing prompts (points/boxes) on a reference frame
    3. Propagating masks to all other frames

    Example:
        >>> predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
        >>> predictor.set_images(images)  # [S, 3, H, W] or list of PIL images
        >>> predictor.add_point_prompt(frame_idx=0, point=(100, 200), label=1)
        >>> masks = predictor.propagate()  # Returns [S, H, W] binary masks
    """

    # Model checkpoints
    CHECKPOINTS = {
        "sam2-hiera-tiny": "facebook/sam2-hiera-tiny",
        "sam2-hiera-small": "facebook/sam2-hiera-small",
        "sam2-hiera-base-plus": "facebook/sam2-hiera-base-plus",
        "sam2-hiera-large": "facebook/sam2-hiera-large",
    }

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_l.yaml",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize SAM2 video predictor.

        Args:
            model_cfg: Path to model config file
            checkpoint_path: Path to model checkpoint
            device: Device to run model on
        """
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Please install with: "
                "pip install segment-anything-2"
            )

        self.device = device
        self.predictor = build_sam2_video_predictor(
            model_cfg,
            checkpoint_path,
            device=device,
        )

        self.inference_state = None
        self.images = None
        self.prompts = []

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "sam2-hiera-large",
        device: str = "cuda",
    ) -> "SAM2VideoPredictor":
        """
        Load a pretrained SAM2 model.

        Args:
            model_name: One of "sam2-hiera-tiny", "sam2-hiera-small",
                       "sam2-hiera-base-plus", "sam2-hiera-large"
            device: Device to run model on

        Returns:
            Initialized SAM2VideoPredictor
        """
        if model_name not in cls.CHECKPOINTS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.CHECKPOINTS.keys())}")

        # The actual loading is handled by build_sam2_video_predictor
        # with automatic HuggingFace download
        cfg_map = {
            "sam2-hiera-tiny": "sam2_hiera_t.yaml",
            "sam2-hiera-small": "sam2_hiera_s.yaml",
            "sam2-hiera-base-plus": "sam2_hiera_b+.yaml",
            "sam2-hiera-large": "sam2_hiera_l.yaml",
        }

        return cls(
            model_cfg=cfg_map[model_name],
            checkpoint_path=cls.CHECKPOINTS[model_name],
            device=device,
        )

    def set_images(
        self,
        images: Union[torch.Tensor, List[Image.Image], List[np.ndarray]],
    ) -> None:
        """
        Set the images/frames for video prediction.

        Args:
            images: Either:
                - torch.Tensor of shape [S, 3, H, W] in range [0, 1]
                - List of PIL Images
                - List of numpy arrays [H, W, 3] in range [0, 255]
        """
        if isinstance(images, torch.Tensor):
            # Convert to list of numpy arrays
            images = images.cpu().numpy()
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            # [S, 3, H, W] -> list of [H, W, 3]
            images = [img.transpose(1, 2, 0) for img in images]
        elif isinstance(images[0], Image.Image):
            images = [np.array(img) for img in images]

        self.images = images
        self.inference_state = self.predictor.init_state(video_path=images)
        self.prompts = []

    def add_point_prompt(
        self,
        frame_idx: int,
        point: Tuple[float, float],
        label: int = 1,
        obj_id: int = 1,
    ) -> None:
        """
        Add a point prompt on a specific frame.

        Args:
            frame_idx: Index of the frame to add prompt to
            point: (x, y) coordinates of the point
            label: 1 for foreground, 0 for background
            obj_id: Object ID for tracking multiple objects
        """
        self.prompts.append({
            "type": "point",
            "frame_idx": frame_idx,
            "point": point,
            "label": label,
            "obj_id": obj_id,
        })

    def add_box_prompt(
        self,
        frame_idx: int,
        box: Tuple[float, float, float, float],
        obj_id: int = 1,
    ) -> None:
        """
        Add a bounding box prompt on a specific frame.

        Args:
            frame_idx: Index of the frame to add prompt to
            box: (x1, y1, x2, y2) bounding box coordinates
            obj_id: Object ID for tracking multiple objects
        """
        self.prompts.append({
            "type": "box",
            "frame_idx": frame_idx,
            "box": box,
            "obj_id": obj_id,
        })

    def propagate(self) -> torch.Tensor:
        """
        Propagate masks from prompts to all frames.

        Returns:
            torch.Tensor: Binary masks of shape [S, H, W]
        """
        if self.inference_state is None:
            raise RuntimeError("Call set_images() first")

        if not self.prompts:
            raise RuntimeError("No prompts added. Call add_point_prompt() or add_box_prompt()")

        # Group prompts by frame and object
        prompts_by_frame = {}
        for prompt in self.prompts:
            frame_idx = prompt["frame_idx"]
            if frame_idx not in prompts_by_frame:
                prompts_by_frame[frame_idx] = []
            prompts_by_frame[frame_idx].append(prompt)

        # Add prompts to predictor
        for frame_idx, frame_prompts in prompts_by_frame.items():
            points = []
            labels = []
            boxes = []

            for prompt in frame_prompts:
                obj_id = prompt["obj_id"]

                if prompt["type"] == "point":
                    points.append(prompt["point"])
                    labels.append(prompt["label"])
                elif prompt["type"] == "box":
                    boxes.append(prompt["box"])

            # Prepare inputs
            point_coords = np.array(points) if points else None
            point_labels = np.array(labels) if labels else None
            box = np.array(boxes[0]) if boxes else None  # SAM2 takes one box

            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=point_coords,
                labels=point_labels,
                box=box,
            )

        # Propagate through video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Combine all object masks into single mask per frame
        num_frames = len(self.images)
        H, W = self.images[0].shape[:2]
        masks = torch.zeros(num_frames, H, W, dtype=torch.bool)

        for frame_idx in range(num_frames):
            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    masks[frame_idx] = masks[frame_idx] | mask.squeeze()

        return masks.float()

    def reset(self) -> None:
        """Reset the predictor state."""
        self.inference_state = None
        self.images = None
        self.prompts = []


def propagate_masks_from_first_frame(
    images: torch.Tensor,
    first_frame_points: List[Tuple[float, float]],
    first_frame_labels: Optional[List[int]] = None,
    model_name: str = "sam2-hiera-large",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Convenience function to propagate masks from point prompts on the first frame.

    Args:
        images: Input images [S, 3, H, W] in range [0, 1]
        first_frame_points: List of (x, y) points on the first frame
        first_frame_labels: Labels for each point (1=foreground, 0=background).
                           Default: all foreground
        model_name: SAM2 model to use
        device: Device to run on

    Returns:
        torch.Tensor: Binary masks [S, H, W]

    Example:
        >>> images = load_images()  # [S, 3, H, W]
        >>> points = [(100, 200), (150, 250)]  # Click on object
        >>> masks = propagate_masks_from_first_frame(images, points)
    """
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2 is required for mask propagation")

    if first_frame_labels is None:
        first_frame_labels = [1] * len(first_frame_points)

    predictor = SAM2VideoPredictor.from_pretrained(model_name, device=device)
    predictor.set_images(images)

    for point, label in zip(first_frame_points, first_frame_labels):
        predictor.add_point_prompt(frame_idx=0, point=point, label=label)

    masks = predictor.propagate()
    return masks
