# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer) is a feed-forward neural network that infers 3D scene attributes from images, including camera parameters, point maps, depth maps, and 3D point tracks. It processes single to hundreds of views in seconds.

## Common Commands

### Installation
```bash
pip install -r requirements.txt           # Core dependencies
pip install -r requirements_demo.txt      # Demo/visualization dependencies
pip install -e .                          # Install as editable package
```

### Running Demos
```bash
python demo_gradio.py                                    # Gradio web interface
python demo_viser.py --image_folder path/to/images       # Viser 3D viewer
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/       # Export to COLMAP format
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba  # With bundle adjustment
```

### Training
Training uses Hydra for configuration and requires torchrun for distributed training:
```bash
cd training
torchrun --nproc_per_node=NUM_GPUS launch.py --config=default
```

Before training:
1. Set `CO3D_DIR` and `CO3D_ANNOTATION_DIR` in `training/config/default.yaml`
2. Set `resume_checkpoint_path` to the pretrained model path
3. Adjust `frozen_module_names` to control which modules to freeze

## Architecture

### Core Model Structure (`vggt/models/`)

**VGGT** (`vggt.py`): Main model with four prediction heads:
- `Aggregator`: Processes images through alternating frame/global attention
- `CameraHead`: Predicts camera pose encodings (extrinsic/intrinsic)
- `DPTHead` (depth_head): Predicts depth maps with confidence
- `DPTHead` (point_head): Predicts 3D world points with confidence
- `TrackHead`: Predicts 3D point tracks across frames

**Aggregator** (`aggregator.py`): Uses alternating attention mechanism:
- `frame_blocks`: Attention within each frame (shape: B*S, P, C)
- `global_blocks`: Attention across all frames (shape: B, S*P, C)
- Uses DINOv2 ViT-L as default patch embedder
- Special tokens: `camera_token` and `register_token` (differentiated for first vs. other frames)

### Key Utilities (`vggt/utils/`)
- `pose_enc.py`: `pose_encoding_to_extri_intri()` converts pose encoding to OpenCV-convention extrinsic/intrinsic matrices
- `geometry.py`: `unproject_depth_map_to_point_map()` constructs 3D points from depth + camera params
- `load_fn.py`: `load_and_preprocess_images()` for image loading

### Training System (`training/`)
- `trainer.py`: DDP trainer with Hydra config, gradient accumulation, AMP support
- `loss.py`: MultitaskLoss combining camera, depth, and point losses
- `data/`: Dataset implementations (Co3D, VKitti) with dynamic batching
- Config files in `training/config/` use Hydra with `_target_` instantiation

### Input/Output Conventions
- Images: `[B, S, 3, H, W]` in range `[0, 1]` (B=batch, S=sequence)
- Cameras: OpenCV convention (camera from world)
- Default image size: 518x518, patch size: 14

### Precision
- Use `torch.bfloat16` on Ampere+ GPUs (Compute Capability 8.0+), otherwise `torch.float16`
- Training uses gradient checkpointing when `model.train()` is called
