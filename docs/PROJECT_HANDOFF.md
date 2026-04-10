# Project Handoff

This document is the durable context record for future agents.
Its purpose is to make full-repo rescans unnecessary.

Read this before reading code.

## 1. What This Repository Is

At a high level, this repository contains two layers:

1. Original VGGT
2. A fork for reflective / specular / transparent material adaptation

Original VGGT is a feed-forward multi-view geometry model.
Given one or more images, it predicts:

- camera parameters
- depth maps
- point maps
- optional point tracks

The fork adds:

- LoRA injection for lightweight fine-tuning
- mask-guided global attention
- an OpenMaterial dataset loader
- a visual-hull-aware demo and training route

So the correct mental model is:

- base repo = generic multi-view geometry model
- fork = targeted adaptation path for difficult materials where cross-view consistency breaks

## 2. The Real Project Goal Of This Fork

The main fork goal is not "general cleanup" or "generic finetuning".
It is:

- improve VGGT on reflective, specular, and transparent objects
- assume foreground masks are available
- modify as little of the original model as possible

The guiding hypothesis is:

- those material types mostly damage cross-view matching and aggregation
- therefore the most important place to intervene is the backbone's global attention
- not the dense heads
- not a full retraining of the model

That is why the fork centers on:

- visual-hull / foreground masks
- suppression of background tokens during global attention
- LoRA on global attention blocks

## 3. Top-Level Repository Layout

### `vggt/`

Main package.

Contains:

- `models/`: the top-level model and backbone
- `heads/`: camera, depth, point, and track heads
- `layers/`: transformer building blocks
- `utils/`: image loading, geometry, pose encoding
- `dependency/`: tracker / COLMAP-related support code
- `lora/`: fork-specific LoRA support
- `sam2/`: fork-specific mask and visual-hull utilities

### `training/`

Hydra + DDP training stack.

Contains:

- launch entrypoint
- trainer
- loss functions
- configs
- datasets and dataloading

### `demo_*.py`

User-facing scripts for inference and visualization.

Important split:

- `demo_gradio.py`, `demo_viser.py`, `demo_colmap.py`: mostly original VGGT usage
- `demo_visual_hull.py`: fork-specific pipeline

### `docs/`

Project notes.

Important documents:

- this handoff file
- `main_route_specular_transparent_vggt.md`: research route note for the hard-material fork

## 4. Core Model Architecture

## 4.1 Top-Level Model

Main file:

- [vggt/models/vggt.py](/home/fangsuo/py/vggt/vggt/models/vggt.py)

`VGGT` is a wrapper around:

- `Aggregator`
- `CameraHead`
- `DPTHead` for depth
- `DPTHead` for point maps
- `TrackHead`

The model accepts either:

- `[S, 3, H, W]`
- `[B, S, 3, H, W]`

It returns a prediction dictionary.

Standard outputs:

- `pose_enc`
- `pose_enc_list`
- `depth`
- `depth_conf`
- `world_points`
- `world_points_conf`
- `images` during inference

Optional tracking outputs if query points are given:

- `track`
- `vis`
- `conf`

Fork-specific addition:

- `visual_hull_mask` can be passed to `forward`

## 4.2 Backbone: Aggregator

Main file:

- [vggt/models/aggregator.py](/home/fangsuo/py/vggt/vggt/models/aggregator.py)

This is the most important file in the repository.

The backbone works like this:

1. Normalize images with ImageNet / ResNet statistics.
2. Patchify each image with a ViT-style patch embedder.
3. Add special tokens:
   - one camera token
   - register tokens
4. Alternate between:
   - `frame_blocks`: attention inside a single frame
   - `global_blocks`: attention across all frames
5. Collect intermediate outputs from both streams.
6. Concatenate frame and global intermediates channel-wise.
7. Send those features to the heads.

Important interpretation:

- `frame attention` learns single-view structure
- `global attention` learns cross-view correspondence and multi-view fusion

That is the core reason the fork targets global attention for reflective / transparent data.

## 4.3 Heads

### Camera head

Main file:

- [vggt/heads/camera_head.py](/home/fangsuo/py/vggt/vggt/heads/camera_head.py)

It predicts a 9D pose encoding:

- translation: 3
- quaternion rotation: 4
- field of view: 2

It uses iterative refinement over the camera tokens.

### Dense heads

Main file:

- [vggt/heads/dpt_head.py](/home/fangsuo/py/vggt/vggt/heads/dpt_head.py)

The same DPT-style head class is used for:

- depth prediction
- point-map prediction
- feature extraction for the tracking head

It fuses multiple intermediate transformer layers into dense outputs.

### Track head

Main file:

- [vggt/heads/track_head.py](/home/fangsuo/py/vggt/vggt/heads/track_head.py)

This head first extracts dense features with a feature-only DPT head, then runs a tracker predictor.

Important maturity note:

- tracking inference exists
- tracking training is not the stable path of this fork

## 5. Conventions And Data Shapes

### Images

- Usually float tensors in `[0, 1]`
- Shape `[B, S, 3, H, W]`

### Camera convention

- OpenCV convention
- extrinsics are world-to-camera
- this is sometimes described in comments as `camera from world`

### Common resolution assumptions

- default image size is `518`
- patch size is `14`

### Output shape conventions

- depth: `[B, S, H, W, 1]`
- depth confidence: `[B, S, H, W]`
- point map: `[B, S, H, W, 3]` plus confidence

## 6. Inference Data Flow

The standard forward path is:

1. load and preprocess images
2. call `model(images)`
3. backbone produces aggregated tokens
4. camera head predicts pose encoding
5. dense heads predict depth and / or point maps
6. `pose_encoding_to_extri_intri()` turns pose encoding into usable camera matrices
7. `unproject_depth_map_to_point_map()` can build world points from depth + cameras

Important utility files:

- [vggt/utils/load_fn.py](/home/fangsuo/py/vggt/vggt/utils/load_fn.py)
- [vggt/utils/pose_enc.py](/home/fangsuo/py/vggt/vggt/utils/pose_enc.py)
- [vggt/utils/geometry.py](/home/fangsuo/py/vggt/vggt/utils/geometry.py)

Important caveat:

- the geometry utilities are not perfectly uniform in torch vs numpy behavior
- some demo paths convert tensors to numpy early

## 7. Demo Scripts And What They Are For

### `demo_gradio.py`

Browser UI.

Use when:

- user wants an upload-and-reconstruct workflow
- user wants GLB export and browser inspection

### `demo_viser.py`

3D point cloud and camera visualization with `viser`.

Use when:

- user wants a lightweight interactive viewer
- debugging point clouds and camera poses

### `demo_colmap.py`

Exports VGGT predictions to COLMAP format, optionally with BA.

Use when:

- user wants downstream Gaussian splatting or NeRF tooling

Important caveats:

- still has TODOs for masks, iterative BA, and distortion support
- BA path uses external tracking support code

### `demo_visual_hull.py`

Fork-specific end-to-end pipeline for hard materials.

Pipeline:

1. get masks from SAM2 or load masks
2. run VGGT with `visual_hull_mask`
3. compute visual hull from silhouettes
4. combine VGGT points with visual hull points
5. optionally save / visualize result

Use this script when the task is specifically about the hard-material adaptation route.

## 8. Training Stack

Main files:

- [training/launch.py](/home/fangsuo/py/vggt/training/launch.py)
- [training/trainer.py](/home/fangsuo/py/vggt/training/trainer.py)
- [training/loss.py](/home/fangsuo/py/vggt/training/loss.py)

### Entry model

`training/launch.py` uses Hydra to load a config and instantiate `Trainer`.

### Trainer responsibilities

`training/trainer.py` handles:

- env setup
- DDP initialization
- model / loss / optimizer construction
- optional checkpoint resume
- optional LoRA injection
- training and validation loops
- AMP
- scheduler stepping
- checkpoint saving

### Important training behavior

- batch data is normalized before forward
- if visual-hull mode is enabled and masks exist in batch, masks are passed to model forward
- the trainer supports gradient accumulation and DDP no-sync

### Losses

Current stable losses:

- camera loss
- depth loss
- optional point loss

Current unstable / incomplete area:

- track loss is explicitly not cleaned up and is not the maintained path

## 9. Data Pipeline

Important files:

- [training/data/base_dataset.py](/home/fangsuo/py/vggt/training/data/base_dataset.py)
- [training/data/composed_dataset.py](/home/fangsuo/py/vggt/training/data/composed_dataset.py)
- [training/data/dynamic_dataloader.py](/home/fangsuo/py/vggt/training/data/dynamic_dataloader.py)

### General pattern

The training stack does not use a simple fixed-shape dataset.

It uses:

- dynamic image counts per sample
- dynamic aspect ratios
- a composed dataset wrapper
- distributed batching constrained by `max_img_per_gpu`

### What the batch usually contains

Typical keys:

- `images`
- `depths`
- `extrinsics`
- `intrinsics`
- `cam_points`
- `world_points`
- `point_masks`
- optional `masks`
- optional tracking data

### Why this matters

If something fails in training, the bug may not be in the model.
It may come from:

- batch construction
- geometric transforms during preprocessing
- mask alignment with image transforms
- normalization of cameras and points

## 10. Dataset-Specific Notes

## 10.1 Co3D

This is the original main training dataset path for default VGGT fine-tuning.

Use it when:

- reproducing the default training setup
- validating the generic training stack

## 10.2 VKitti

Auxiliary dataset for geometry diversity.

Use it when:

- mixing datasets
- augmenting the training distribution

## 10.3 OpenMaterial

This is the fork-specific dataset for the hard-material direction.

Main file:

- [training/data/datasets/openmaterial.py](/home/fangsuo/py/vggt/training/data/datasets/openmaterial.py)

What it does:

- reads NeRF-style `transforms_train.json` / `transforms_test.json`
- loads RGB images
- loads foreground masks if available
- converts NeRF camera convention to OpenCV world-to-camera
- loads a GT mesh
- loads precomputed mesh-depth `.npy` files if available
- otherwise rasterizes the GT mesh into depth maps on CPU
- caches rendered depth maps with an LRU byte budget
- applies the same image geometry transforms to image, depth, and mask

Why it exists:

- reflective / transparent objects need stronger geometry supervision than ordinary image-only cues
- masks matter
- GT mesh-derived depth is the supervision source

Important constraints:

- rasterization is CPU-side and correctness-oriented, not production-fast
- GT mesh is required in the current implementation
- mask paths are mildly inconsistent in the wild, so the loader supports both `mask/` and `masks/`
- the preferred fast path is offline mesh-depth precompute via `training/data/preprocess/openmaterial_depth_cache.py`
- training can read those caches either from sidecar `depth_mesh/` folders under each scene or from a separate cache root via `depth_precompute_dir`
- the precompute script now supports GPU rasterization backends, including `nvdiffrast`

What was verified on this fork:

- the pre-trained checkpoint at `checkpoints/model.pt` loads successfully
- LoRA GPU smoke training works at `img_size=518`
- offline OpenMaterial depth caches were generated successfully for the smoke subset
- training was verified to consume those offline depth caches instead of falling back to online CPU mesh rasterization
- the `nvdiffrast` GPU backend successfully generated one full OpenMaterial scene cache on a server with CUDA 13.0 and RTX 5090

How the cache-path verification was done:

- use the smoke subset under `/tmp/vggt_lora_smoke_subset`
- point `depth_precompute_dir` at `/tmp/vggt_lora_smoke_depth_cache`
- force `require_precomputed_depth=True` for both train and val
- run a 1-GPU LoRA smoke training with a small dynamic batch
- confirm training reaches `Train Epoch` and `Val Epoch` and saves checkpoints
- separately patch the online rasterizer to raise immediately and verify `OpenMaterialDataset.get_data(...)` still succeeds

Practical implication:

- the main remaining blocker for full OpenMaterial training startup is no longer "does the dataset read cache files"
- it is now "finish full-dataset GPU precompute and choose a GPU-safe training batch shape"

Operational note from server bring-up:

- `nvdiffrast` is not available on PyPI; install it from GitHub
- if `nvcc` exists but install fails with `cicc: not found`, check whether `CUDA_HOME` points at the full toolkit root such as `/usr/local/cuda-13.0`
- one real server had:
  - project at `/opt/data/private/fyp/vggt`
  - dataset at `/opt/data/private/dataset/OpenMaterial_ablation`
  - cache output at `/opt/data/private/dataset/OpenMaterial_ablation_depth_cache`
- on that server, a Python environment mismatch also occurred during probe launch:
  - `python` in the active env was `/root/om/bin/python`
  - but `torchrun` resolved to `/usr/local/bin/torchrun`
  - that `torchrun` spawned `/usr/bin/python`, which caused `ModuleNotFoundError: No module named 'hydra'` even though `hydra` was installed in the active env
  - on that server, the reliable launch path was `python -m torch.distributed.run ...` from the active environment instead of the bare `torchrun` on `PATH`
- to avoid repeating long server CLI overrides, this repo now includes:
  - config `training/config/openmaterial_probe_server.yaml`
  - entry script `training/run_openmaterial_probe_server.sh`
  - config `training/config/openmaterial_train_server.yaml`
  - entry script `training/run_openmaterial_train_server.sh`
  - config `training/config/openmaterial_probe_server_disjoint.yaml`
  - entry script `training/run_openmaterial_probe_server_disjoint.sh`
  - config `training/config/openmaterial_train_server_disjoint.yaml`
  - entry script `training/run_openmaterial_train_server_disjoint.sh`
  - split helper `training/data/preprocess/openmaterial_scene_split.py`
  - split helper entry script `training/run_openmaterial_scene_split_server.sh`
- on that server, the `nvdiffrast` backend processed one 90-frame scene in under 10 seconds, which is dramatically faster than the old CPU rasterizer

## 11. Fork-Specific Mechanisms

## 11.1 LoRA

Main files:

- [vggt/lora/lora_utils.py](/home/fangsuo/py/vggt/vggt/lora/lora_utils.py)
- [vggt/lora/lora_layers.py](/home/fangsuo/py/vggt/vggt/lora/lora_layers.py)

How it works:

- wrap selected attention `nn.Linear` modules with `LoRALinear`
- freeze the base model if requested
- keep only LoRA parameters trainable

Targeting options:

- `global` blocks
- `frame` blocks
- `both`

Module options:

- `qkv`
- `proj`

Current default strategy of this fork:

- LoRA on `global` blocks only
- `qkv` only
- all global blocks unless narrowed by config

Reason:

- the adaptation target is cross-view fusion
- this is the lowest-risk intervention point

## 11.2 Visual-Hull-Aware Attention

Main files:

- [vggt/models/aggregator.py](/home/fangsuo/py/vggt/vggt/models/aggregator.py)
- [vggt/layers/attention.py](/home/fangsuo/py/vggt/vggt/layers/attention.py)
- [vggt/layers/block.py](/home/fangsuo/py/vggt/vggt/layers/block.py)

Mechanism:

1. user supplies `visual_hull_mask`
2. mask is downsampled to patch resolution
3. special tokens are kept always valid
4. a global attention mask is built
5. background keys are suppressed in global attention

Important nuance:

- this does not redesign the architecture
- it constrains the existing global attention
- it is a relatively conservative intervention

## 11.3 SAM2 And Visual Hull

Main files:

- [vggt/sam2/sam2_wrapper.py](/home/fangsuo/py/vggt/vggt/sam2/sam2_wrapper.py)
- [vggt/sam2/visual_hull.py](/home/fangsuo/py/vggt/vggt/sam2/visual_hull.py)
- [demo_visual_hull.py](/home/fangsuo/py/vggt/demo_visual_hull.py)

Role:

- generate or propagate masks
- estimate a visual hull from silhouettes and cameras
- merge hull points with VGGT points

This is not the generic path of the repository.
It is the fork's specialized route.

## 12. Configurations That Matter

## 12.1 Default training config

Main file:

- [training/config/default.yaml](/home/fangsuo/py/vggt/training/config/default.yaml)

Default philosophy:

- use Co3D
- freeze the aggregator
- train camera and depth heads
- point and track are not the default path

This is the stable generic fine-tuning route.

## 12.2 LoRA visual-hull config

Main file:

- [training/config/lora_finetune.yaml](/home/fangsuo/py/vggt/training/config/lora_finetune.yaml)

Fork philosophy:

- use OpenMaterial
- enable masks
- enable visual-hull training path
- inject LoRA into global attention
- keep base model frozen

This is the fork's main research configuration.

## 13. What Is Mature Vs What Is Prototype

More mature:

- core VGGT inference
- backbone + camera/depth/point heads
- generic demos
- default training scaffold

Moderately mature but fork-specific:

- LoRA support
- visual-hull-aware attention
- OpenMaterial loader

More prototype-like:

- full hard-material route as a polished product
- track training
- some BA / export edge cases
- mask support in every downstream path

## 14. Known Gaps And Traps

- Do not assume all demos share the same tensor / numpy conventions.
- Do not assume tracking is the maintained training path.
- Do not assume OpenMaterial supervision exists without a GT mesh.
- Do not assume the hard-material fork changed the whole model; most of the original model remains intact.
- Do not start modifying heads if the failure mode is clearly cross-view consistency; inspect global attention first.
- Do not confuse "visual hull" here with a full geometry replacement; it is used as guidance and hybridization.

## 15. Minimal Code Reading By Task

If you already read this document, use the following minimal slices.

### To change inference API

Read only:

- `vggt/models/vggt.py`
- `vggt/utils/load_fn.py`
- the one demo you are touching

### To change core multi-view reasoning

Read only:

- `vggt/models/aggregator.py`
- `vggt/layers/attention.py`
- `vggt/layers/block.py`

### To change training logic

Read only:

- `training/trainer.py`
- `training/loss.py`
- the specific Hydra config

### To change LoRA behavior

Read only:

- `vggt/lora/lora_utils.py`
- `vggt/lora/lora_layers.py`
- `training/trainer.py`
- `training/config/lora_finetune.yaml`

### To change OpenMaterial supervision

Read only:

- `training/data/datasets/openmaterial.py`
- `training/data/base_dataset.py`
- `training/data/composed_dataset.py`
- `training/data/preprocess/openmaterial_depth_cache.py`

### To change visual hull / SAM2 flow

Read only:

- `demo_visual_hull.py`
- `vggt/sam2/sam2_wrapper.py`
- `vggt/sam2/visual_hull.py`

## 16. Recommended Workflow For Future Agents

Before work:

1. Read `AGENT_START_HERE.md`
2. Read this file
3. Check `git status --short`
4. Read only the minimal code slice for the task

When responding:

- state whether you are dealing with generic VGGT or fork-specific hard-material logic
- identify whether the issue is in:
  - backbone
  - head
  - training loop
  - dataset / supervision
  - demo / export path

When modifying:

- preserve the distinction between original VGGT behavior and fork additions
- prefer small interventions on the hard-material route instead of broad architectural churn

After significant changes:

- update this document so future agents do not have to rediscover the same context

For the current OpenMaterial training route:

1. Verify whether the task is about dataset/cache throughput or model behavior.
2. If it is about startup slowness, check `depth_precompute_dir` and `require_precomputed_depth` first.
3. Keep `img_size=518` when loading the released pre-trained checkpoint.
4. Prefer GPU depth precompute with `nvdiffrast` when a suitable CUDA environment is available.
5. On this machine, start smoke or probe runs with conservative image count / batch settings before scaling up.
6. On this server, prefer `python -m torch.distributed.run ...` over the bare `torchrun` on `PATH` for OpenMaterial probe and training runs.
7. For any benchmark claim, do not train on all 105 scenes; generate a scene-disjoint manifest split first and use the `*_disjoint` server configs.

## 17. What To Update In This Handoff When The Project Changes

Update this document if any of these change:

- repository identity
- main research goal
- recommended intervention point
- training default path
- LoRA target strategy
- dataset assumptions
- visual-hull pipeline
- minimal reading map
- cache verification status for the OpenMaterial route

Do not let this become a stale essay.
Its job is to replace repeated repo-wide context gathering.
