#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visual-Hull-Aware VGGT Demo

This demo implements the complete pipeline for generating high-quality point clouds
from multi-view images of reflective/specular objects:

1. SAM2 Mask Propagation: Generate consistent masks across views
2. Visual-Hull-Aware VGGT: Run VGGT with mask-guided attention
3. Visual Hull Computation: Compute visual hull from masks
4. Hybrid Seeding: Combine VGGT predictions with Visual Hull surface

Usage:
    # With interactive point prompts
    python demo_visual_hull.py --image_folder path/to/images --interactive

    # With pre-computed masks
    python demo_visual_hull.py --image_folder path/to/images --mask_folder path/to/masks

    # With LoRA fine-tuned model
    python demo_visual_hull.py --image_folder path/to/images --lora_path path/to/lora.pt
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Visual-Hull-Aware VGGT Demo")

    # Input
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing input images")
    parser.add_argument("--mask_folder", type=str, default=None,
                        help="Path to folder containing pre-computed masks (optional)")

    # SAM2 options
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive mode for SAM2 prompts")
    parser.add_argument("--sam2_model", type=str, default="sam2-hiera-large",
                        choices=["sam2-hiera-tiny", "sam2-hiera-small",
                                 "sam2-hiera-base-plus", "sam2-hiera-large"],
                        help="SAM2 model to use")
    parser.add_argument("--prompt_points", type=str, default=None,
                        help="Comma-separated list of x,y points, e.g., '100,200,150,250'")

    # VGGT options
    parser.add_argument("--vggt_model", type=str, default="facebook/VGGT-1B",
                        help="VGGT model to use")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA weights (optional)")

    # Visual Hull options
    parser.add_argument("--vh_resolution", type=int, default=128,
                        help="Visual Hull voxel resolution")
    parser.add_argument("--vh_threshold", type=float, default=0.9,
                        help="Visual Hull voting threshold")
    parser.add_argument("--vh_weight", type=float, default=0.3,
                        help="Weight of Visual Hull points in hybrid cloud")

    # Output
    parser.add_argument("--output_dir", type=str, default="output_visual_hull",
                        help="Output directory")
    parser.add_argument("--save_ply", action="store_true",
                        help="Save point cloud as PLY file")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results with viser")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    return parser.parse_args()


def load_images(folder: str) -> Tuple[torch.Tensor, List[str]]:
    """Load images from folder."""
    folder = Path(folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    image_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise ValueError(f"No images found in {folder}")

    logger.info(f"Found {len(image_files)} images")

    # Load and preprocess
    from vggt.utils.load_fn import load_and_preprocess_images
    images = load_and_preprocess_images([str(f) for f in image_files])

    return images, [f.name for f in image_files]


def load_masks(folder: str, image_names: List[str]) -> torch.Tensor:
    """Load pre-computed masks from folder."""
    folder = Path(folder)
    masks = []

    for name in image_names:
        # Try different mask naming conventions
        mask_candidates = [
            folder / name,
            folder / f"{Path(name).stem}_mask.png",
            folder / f"mask_{name}",
        ]

        mask_path = None
        for candidate in mask_candidates:
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            raise ValueError(f"Mask not found for {name}")

        mask = np.array(Image.open(mask_path).convert('L'))
        masks.append(torch.from_numpy(mask > 128).float())

    return torch.stack(masks)


def generate_masks_with_sam2(
    images: torch.Tensor,
    prompt_points: Optional[List[Tuple[float, float]]] = None,
    model_name: str = "sam2-hiera-large",
    device: str = "cuda",
    interactive: bool = False,
) -> torch.Tensor:
    """Generate masks using SAM2."""
    from vggt.sam2 import SAM2VideoPredictor

    logger.info("Initializing SAM2...")
    predictor = SAM2VideoPredictor.from_pretrained(model_name, device=device)
    predictor.set_images(images)

    if interactive:
        # Interactive mode - show first frame and get click
        logger.info("Interactive mode: Click on the object in the first frame")
        # For now, use center of image as default
        H, W = images.shape[2:]
        prompt_points = [(W // 2, H // 2)]
        logger.info(f"Using default center point: {prompt_points[0]}")

    if prompt_points is None:
        raise ValueError("No prompt points provided. Use --prompt_points or --interactive")

    for point in prompt_points:
        predictor.add_point_prompt(frame_idx=0, point=point, label=1)

    logger.info("Propagating masks...")
    masks = predictor.propagate()

    return masks


def run_vggt_with_mask(
    images: torch.Tensor,
    masks: torch.Tensor,
    model_path: str = "facebook/VGGT-1B",
    lora_path: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """Run VGGT with visual hull mask."""
    from vggt.models.vggt import VGGT

    logger.info("Loading VGGT model...")
    model = VGGT.from_pretrained(model_path).to(device)

    # Load LoRA weights if provided
    if lora_path:
        logger.info(f"Loading LoRA weights from {lora_path}")
        model.enable_lora()  # Initialize LoRA layers
        model.load_lora(lora_path)

    model.eval()

    # Determine precision
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    logger.info("Running VGGT inference with mask...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(
                images.to(device),
                visual_hull_mask=masks.to(device),
            )

    return predictions


def compute_visual_hull(
    masks: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: int = 128,
    threshold: float = 0.9,
) -> torch.Tensor:
    """Compute Visual Hull surface points."""
    from vggt.sam2.visual_hull import compute_visual_hull_points

    logger.info("Computing Visual Hull...")
    surface_points, _ = compute_visual_hull_points(
        masks=masks,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        resolution=resolution,
        threshold=threshold,
    )

    logger.info(f"Visual Hull: {surface_points.shape[0]} surface points")
    return surface_points


def create_hybrid_point_cloud(
    vggt_points: torch.Tensor,
    visual_hull_points: torch.Tensor,
    vggt_confidence: Optional[torch.Tensor] = None,
    visual_hull_weight: float = 0.3,
) -> torch.Tensor:
    """Create hybrid point cloud: P_init = P_VGGT ∪ P_VisualHull."""
    from vggt.sam2.visual_hull import hybrid_point_cloud

    logger.info("Creating hybrid point cloud...")
    combined = hybrid_point_cloud(
        vggt_points=vggt_points,
        visual_hull_points=visual_hull_points,
        vggt_confidence=vggt_confidence,
        visual_hull_weight=visual_hull_weight,
    )

    logger.info(f"Hybrid point cloud: {combined.shape[0]} points")
    return combined


def save_point_cloud_ply(points: torch.Tensor, path: str, colors: Optional[torch.Tensor] = None):
    """Save point cloud as PLY file."""
    import struct

    points = points.cpu().numpy()
    n_points = points.shape[0]

    with open(path, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
"""
        if colors is not None:
            header += """property uchar red
property uchar green
property uchar blue
"""
        header += "end_header\n"
        f.write(header.encode())

        # Data
        if colors is not None:
            colors = (colors.cpu().numpy() * 255).astype(np.uint8)
            for i in range(n_points):
                f.write(struct.pack('<fff', *points[i]))
                f.write(struct.pack('<BBB', *colors[i]))
        else:
            for i in range(n_points):
                f.write(struct.pack('<fff', *points[i]))

    logger.info(f"Saved point cloud to {path}")


def visualize_with_viser(points: torch.Tensor, colors: Optional[torch.Tensor] = None):
    """Visualize point cloud with viser."""
    import viser

    server = viser.ViserServer()
    logger.info(f"Viser server started at {server.request_share_url()}")

    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy() if colors is not None else np.ones_like(points_np) * 0.5

    server.scene.add_point_cloud(
        "/points",
        points=points_np,
        colors=colors_np,
        point_size=0.01,
    )

    input("Press Enter to close visualization...")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    logger.info(f"Loading images from {args.image_folder}")
    images, image_names = load_images(args.image_folder)
    logger.info(f"Loaded {len(image_names)} images")

    # Get or generate masks
    if args.mask_folder:
        logger.info(f"Loading masks from {args.mask_folder}")
        masks = load_masks(args.mask_folder, image_names)
    else:
        # Parse prompt points
        prompt_points = None
        if args.prompt_points:
            coords = [float(x) for x in args.prompt_points.split(',')]
            prompt_points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

        masks = generate_masks_with_sam2(
            images=images,
            prompt_points=prompt_points,
            model_name=args.sam2_model,
            device=args.device,
            interactive=args.interactive,
        )

    # Save masks
    for i, (mask, name) in enumerate(zip(masks, image_names)):
        mask_path = output_dir / f"mask_{name}"
        Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).save(mask_path)

    # Run VGGT with mask
    predictions = run_vggt_with_mask(
        images=images,
        masks=masks,
        model_path=args.vggt_model,
        lora_path=args.lora_path,
        device=args.device,
    )

    # Get VGGT point cloud
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    pose_enc = predictions["pose_enc"]
    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]

    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Unproject depth to points
    # depth shape: [B, S, H, W, 1] or [B, S, 1, H, W]
    # After squeeze(0): [S, H, W, 1] or [S, 1, H, W]
    depth_squeezed = depth.squeeze(0)
    if depth_squeezed.dim() == 4 and depth_squeezed.shape[1] == 1:
        depth_squeezed = depth_squeezed.squeeze(1)  # [S, H, W]
    elif depth_squeezed.dim() == 4 and depth_squeezed.shape[-1] == 1:
        depth_squeezed = depth_squeezed.squeeze(-1)  # [S, H, W]

    vggt_points = unproject_depth_map_to_point_map(
        depth_squeezed,
        extrinsics.squeeze(0),
        intrinsics.squeeze(0),
    )

    # Flatten to point cloud
    # vggt_points shape: [S, H, W, 3]
    S, H, W, _ = vggt_points.shape
    vggt_points_flat = vggt_points.reshape(-1, 3)
    vggt_conf_flat = depth_conf.squeeze(0).reshape(-1)  # [S, H, W] -> [S*H*W]

    # Filter by confidence and mask
    masks_resized = torch.nn.functional.interpolate(
        masks.unsqueeze(1), size=(H, W), mode='nearest'
    ).squeeze(1)
    mask_flat = masks_resized.reshape(-1) > 0.5

    valid = (vggt_conf_flat > 0.5) & mask_flat
    vggt_points_filtered = vggt_points_flat[valid]

    logger.info(f"VGGT points: {vggt_points_filtered.shape[0]} (filtered from {vggt_points_flat.shape[0]})")

    # Compute Visual Hull
    vh_points = compute_visual_hull(
        masks=masks.to(args.device),
        extrinsics=extrinsics.squeeze(0).to(args.device),
        intrinsics=intrinsics.squeeze(0).to(args.device),
        resolution=args.vh_resolution,
        threshold=args.vh_threshold,
    )

    # Create hybrid point cloud
    hybrid_points = create_hybrid_point_cloud(
        vggt_points=vggt_points_filtered.to(args.device),
        visual_hull_points=vh_points,
        visual_hull_weight=args.vh_weight,
    )

    # Save results
    if args.save_ply:
        save_point_cloud_ply(
            vggt_points_filtered,
            str(output_dir / "vggt_points.ply"),
        )
        save_point_cloud_ply(
            vh_points,
            str(output_dir / "visual_hull_points.ply"),
        )
        save_point_cloud_ply(
            hybrid_points,
            str(output_dir / "hybrid_points.ply"),
        )

    # Visualize
    if args.visualize:
        visualize_with_viser(hybrid_points)

    logger.info("Done!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
