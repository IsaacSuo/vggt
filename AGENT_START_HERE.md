# Agent Start Here

This repository is large enough that re-understanding it from code wastes time and tokens.
Do not start by scanning the whole repo.

## Read Order

1. Read this file.
2. Read [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md).
3. Only then read the minimal code for your task from the task map below.

If you follow that order, you should already understand:

- what the repo is
- what this fork is trying to do
- which parts are original VGGT vs fork additions
- where training, inference, LoRA, visual-hull masks, and OpenMaterial fit
- which files matter for a given task

## Repository Identity

This is not just the original VGGT repository.

It is:

- the original VGGT inference/training codebase
- plus a fork focused on reflective / specular / transparent objects
- with mask-guided global attention
- with LoRA fine-tuning support
- with an OpenMaterial dataset loader
- with a visual-hull-aware demo and training path

The current project direction is to improve multi-view geometry on hard materials while preserving the original VGGT prior as much as possible.

## 60-Second Mental Model

- `vggt/models/vggt.py`: top-level model wrapper
- `vggt/models/aggregator.py`: main backbone, alternating `frame` and `global` attention
- `vggt/heads/`: decoders for camera, depth, point map, track
- `training/`: Hydra + DDP training stack
- `training/data/datasets/openmaterial.py`: fork-specific masked dataset with mesh-rasterized depth supervision
- `vggt/lora/`: LoRA injection utilities
- `demo_visual_hull.py`: fork-specific demo for masks + visual hull + hybrid point cloud

The central idea of the fork is:

- reflective / transparent failure mostly corrupts cross-view aggregation
- therefore adapt `global attention`, not the whole model
- use masks to suppress background keys in global attention
- use LoRA for low-risk adaptation of the backbone

## Task Map

Read only the files listed for your task unless you have a concrete reason to go deeper.

### 1. General project understanding

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md)

Optional code:

- [vggt/models/vggt.py](/home/fangsuo/py/vggt/vggt/models/vggt.py)
- [vggt/models/aggregator.py](/home/fangsuo/py/vggt/vggt/models/aggregator.py)

### 2. Basic inference or model API usage

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md), sections on inference and demos

Then code:

- [vggt/models/vggt.py](/home/fangsuo/py/vggt/vggt/models/vggt.py)
- [vggt/utils/load_fn.py](/home/fangsuo/py/vggt/vggt/utils/load_fn.py)
- [vggt/utils/pose_enc.py](/home/fangsuo/py/vggt/vggt/utils/pose_enc.py)
- one demo only:
  - [demo_gradio.py](/home/fangsuo/py/vggt/demo_gradio.py)
  - or [demo_viser.py](/home/fangsuo/py/vggt/demo_viser.py)
  - or [demo_colmap.py](/home/fangsuo/py/vggt/demo_colmap.py)

### 3. Backbone or attention changes

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md), sections on backbone and visual-hull attention

Then code:

- [vggt/models/aggregator.py](/home/fangsuo/py/vggt/vggt/models/aggregator.py)
- [vggt/layers/attention.py](/home/fangsuo/py/vggt/vggt/layers/attention.py)
- [vggt/layers/block.py](/home/fangsuo/py/vggt/vggt/layers/block.py)

### 4. LoRA fine-tuning

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md), sections on LoRA and training

Then code:

- [vggt/lora/lora_utils.py](/home/fangsuo/py/vggt/vggt/lora/lora_utils.py)
- [vggt/lora/lora_layers.py](/home/fangsuo/py/vggt/vggt/lora/lora_layers.py)
- [vggt/models/vggt.py](/home/fangsuo/py/vggt/vggt/models/vggt.py)
- [training/trainer.py](/home/fangsuo/py/vggt/training/trainer.py)
- [training/config/lora_finetune.yaml](/home/fangsuo/py/vggt/training/config/lora_finetune.yaml)

### 5. Training bugs or new training features

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md), sections on training stack and loss

Then code:

- [training/launch.py](/home/fangsuo/py/vggt/training/launch.py)
- [training/trainer.py](/home/fangsuo/py/vggt/training/trainer.py)
- [training/loss.py](/home/fangsuo/py/vggt/training/loss.py)
- relevant config under [training/config](/home/fangsuo/py/vggt/training/config)

### 6. Dataset or supervision issues

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md), dataset sections

Then code:

- [training/data/base_dataset.py](/home/fangsuo/py/vggt/training/data/base_dataset.py)
- [training/data/composed_dataset.py](/home/fangsuo/py/vggt/training/data/composed_dataset.py)
- one dataset only:
  - [training/data/datasets/openmaterial.py](/home/fangsuo/py/vggt/training/data/datasets/openmaterial.py)
  - or [training/data/datasets/co3d.py](/home/fangsuo/py/vggt/training/data/datasets/co3d.py)
  - or [training/data/datasets/vkitti.py](/home/fangsuo/py/vggt/training/data/datasets/vkitti.py)

### 7. Specular / transparent object work

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md), entire file
- [docs/main_route_specular_transparent_vggt.md](docs/main_route_specular_transparent_vggt.md)

Then code:

- [vggt/models/aggregator.py](/home/fangsuo/py/vggt/vggt/models/aggregator.py)
- [training/config/lora_finetune.yaml](/home/fangsuo/py/vggt/training/config/lora_finetune.yaml)
- [training/data/datasets/openmaterial.py](/home/fangsuo/py/vggt/training/data/datasets/openmaterial.py)
- [demo_visual_hull.py](/home/fangsuo/py/vggt/demo_visual_hull.py)

### 8. Visual hull / SAM2 pipeline

Read:

- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md), visual-hull sections

Then code:

- [demo_visual_hull.py](/home/fangsuo/py/vggt/demo_visual_hull.py)
- [vggt/sam2/sam2_wrapper.py](/home/fangsuo/py/vggt/vggt/sam2/sam2_wrapper.py)
- [vggt/sam2/visual_hull.py](/home/fangsuo/py/vggt/vggt/sam2/visual_hull.py)

## Operating Rules For Future Agents

- Do not re-scan the full repo before reading the handoff docs.
- Distinguish clearly between original VGGT behavior and fork-added behavior.
- For any task, read the minimum code slice from the task map instead of everything nearby.
- If you make a meaningful architectural, training, or dataset change, update `docs/PROJECT_HANDOFF.md`.

## Maintenance Rule

When the project changes materially, update:

- `docs/PROJECT_HANDOFF.md` for durable knowledge
- this file only if the recommended read order or task map changes

The goal is to keep future context loading cheap and deterministic.
