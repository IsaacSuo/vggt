# Specular and Transparent Material Adaptation Follow-up

This document is the working note for the current fork on branch `main`.

- Repository branch: `main`
- Current HEAD at review time: `842f423`
- Scope of this note: understand the current fork, record the current engineering state, summarize required background knowledge, and define the project route for adapting VGGT to reflective and transparent objects with masks.

## 1. Project Goal

Primary goal:

- Adapt VGGT so that multi-view geometry is more stable on reflective, specular, and transparent materials.

Current assumption:

- Foreground masks are available.

This is important because with masks the problem becomes:

- suppress background and irrelevant context
- reduce cross-view aggregation errors caused by reflections
- adapt the geometric fusion stage rather than rebuilding the whole model

The recommended direction is therefore:

- use mask-guided global attention
- use minimal LoRA on the multi-view aggregation backbone
- keep the original VGGT prior as intact as possible

## 2. What VGGT Is

VGGT is a feed-forward multi-view geometry model.

Given one or more images, it predicts:

- camera parameters
- depth maps
- point maps
- optional tracks

The key design is:

- patchify images
- run alternating `frame attention` and `global attention`
- use the resulting tokens for geometry heads

In this repository, the main model entry is:

- `vggt/models/vggt.py`

The main backbone is:

- `vggt/models/aggregator.py`

The main output heads are:

- `vggt/heads/camera_head.py`
- `vggt/heads/dpt_head.py`
- `vggt/heads/track_head.py`

## 3. Core Code Structure

### 3.1 Backbone

`Aggregator` is the main geometry backbone.

Important details:

- input shape is `[B, S, 3, H, W]`
- images are normalized and patch embedded
- each frame gets special tokens:
  - camera token
  - register tokens
- the network alternates:
  - `frame_blocks`: within-frame attention
  - `global_blocks`: across-frame attention
- outputs from frame and global streams are concatenated along channels

This means:

- frame attention learns single-view structure
- global attention learns cross-view consistency

For reflective and transparent materials, the most relevant part is:

- `global attention`

because that is where cross-view matching and fusion can be corrupted by specular effects.

### 3.2 Camera Prediction

`CameraHead` predicts a 9D pose encoding:

- translation: 3
- quaternion: 4
- field of view: 2

The conversion utilities are in:

- `vggt/utils/pose_enc.py`

Important convention:

- the code uses OpenCV camera convention
- extrinsics are world-to-camera

### 3.3 Dense Geometry Prediction

`DPTHead` is used for:

- depth prediction
- point map prediction
- feature extraction for tracking

The head fuses several intermediate transformer layers.

### 3.4 Geometry Utilities

Relevant utility:

- `vggt/utils/geometry.py`

Important note:

- `unproject_depth_map_to_point_map()` returns `numpy.ndarray`, not `torch.Tensor`

This matters when building demos or pipelines that combine torch and numpy code.

## 4. Current Fork Additions Compared to the Original Mainline

The current fork has added several new components on top of the earlier VGGT training branch.

### 4.1 LoRA Support

Main files:

- `vggt/lora/lora_layers.py`
- `vggt/lora/lora_utils.py`
- `vggt/models/vggt.py`
- `training/trainer.py`

Current behavior:

- LoRA wraps attention linear layers with `LoRALinear`
- target locations can be:
  - `frame_blocks`
  - `global_blocks`
  - both
- target modules can be:
  - `qkv`
  - `proj`

Default LoRA config in this fork:

- only `global` blocks
- only `qkv`
- all global blocks unless `block_indices` is specified

### 4.2 Visual-Hull-Aware Attention

Main files:

- `vggt/models/aggregator.py`
- `vggt/layers/attention.py`
- `vggt/layers/block.py`

Current behavior:

- `visual_hull_mask` can be provided to `VGGT.forward()`
- the mask is downsampled to patch level
- global attention receives an additive mask
- background keys are suppressed in global attention

This is the most relevant fork addition for the target task.

### 4.3 SAM2 and Visual Hull Utilities

Main files:

- `vggt/sam2/sam2_wrapper.py`
- `vggt/sam2/visual_hull.py`
- `demo_visual_hull.py`

Current intent:

- generate or load masks
- run VGGT with mask-guided attention
- compute visual hull from silhouettes
- combine VGGT points and visual hull points into a hybrid point cloud

### 4.4 Added Training Datasets

Main files:

- `training/data/datasets/openmaterial.py`
- `training/data/datasets/vkitti.py`

Intended roles:

- `OpenMaterialDataset`: masked synthetic data for reflective/specular objects, now using GT mesh rasterization for depth supervision
- `VKittiDataset`: additional training source for geometry diversity

## 5. Background Knowledge Required

Anyone continuing this project should be comfortable with the following.

### 5.1 Multi-view Geometry

Minimum required concepts:

- camera intrinsics and extrinsics
- world-to-camera vs camera-to-world transforms
- reprojection
- depth unprojection
- silhouette consistency

### 5.2 Transformer Internals

Important concepts:

- patch embedding
- self-attention
- QKV projections
- multi-head attention
- why `qkv` is the most sensitive place for LoRA

### 5.3 LoRA

Need to understand:

- LoRA modifies a frozen linear layer by adding low-rank adapters
- applying LoRA changes parameter names
- checkpoint loading order matters

### 5.4 Reflective and Transparent Failure Modes

Important intuition:

- reflective objects break appearance consistency across views
- transparent objects can break both appearance and apparent surface location
- with masks, the main benefit is focusing aggregation on the object region
- masks help a lot, but they do not fully solve refractive ambiguity

### 5.5 Visual Hull

Need to understand:

- a visual hull is a silhouette-consistent outer shape
- it is usually more reliable as a geometric boundary than as exact surface geometry
- it can be used as:
  - a support prior
  - a filtering constraint
  - a hybrid seeding mechanism

## 6. Current Engineering Issues Found During Review

These were the major issues found in the current fork state.

### 6.1 LoRA Loading Order Was Wrong

Problem:

- LoRA wrapping changes parameter names from `attn.qkv.weight` to `attn.qkv.original_layer.weight` plus LoRA tensors
- loading a base VGGT checkpoint after LoRA injection causes missing and unexpected keys

Impact:

- LoRA fine-tuning from a base pretrained checkpoint was not actually restoring the original wrapped attention weights correctly

### 6.2 Visual-Hull Training Path Was Not Fully Connected

Problem:

- `OpenMaterialDataset` produced `masks`
- `ComposedDataset` dropped them
- `Trainer._step()` did not pass them to the model

Impact:

- the mask-guided training story in the config was not actually active end-to-end

### 6.3 `mask_penalty` Existed Only in Config

Problem:

- `training/config/lora_finetune.yaml` declared `mask_penalty`
- `training/loss.py` had no implementation for it

Impact:

- the config advertised behavior that did not exist

### 6.4 `demo_visual_hull.py` Mixed `numpy` and `torch`

Problem:

- the unprojection utility returns numpy
- later code assumed torch and called `.to()`, `.cpu()`, etc.

Impact:

- the demo could fail during hybrid cloud creation or PLY saving

### 6.5 Pre-computed Mask Path Was Misaligned

Problem:

- external masks loaded from disk were not preprocessed with the same geometry as model inputs

Impact:

- `--mask_folder` could feed masks at the wrong resolution/layout into the model

### 6.6 OpenMaterial Needed Real Geometry Supervision

Problem:

- `OpenMaterialDataset` originally had no real depth source in the training path
- using a simple point-cloud projection from `points3d.ply` was too sparse and too weak to count as meaningful depth supervision

Impact:

- the training objective looked denser than it really was
- projected point clouds left large holes and unstable boundaries
- this was not a credible long-term route for OpenMaterial fine-tuning

## 7. Fixes Already Applied Locally

The following fixes have been applied locally in the working tree.

### 7.1 Fixed LoRA Checkpoint Order

File:

- `training/trainer.py`

Current behavior after fix:

- resolve checkpoint path first
- if LoRA is requested and the checkpoint is a base checkpoint:
  - load base model weights first
  - do not load optimizer or training state yet
- inject LoRA after base weights are present
- only then construct optimizers
- if the checkpoint already contains LoRA parameter names, load it after LoRA injection

This preserves the original attention weights.

### 7.2 Connected `masks` to Training Forward Pass

Files:

- `training/data/composed_dataset.py`
- `training/trainer.py`

Current behavior after fix:

- `masks` are retained in the composed sample when available
- repeated batches include `masks`
- `Trainer._step()` passes `visual_hull_mask=batch["masks"]` when visual hull mode is enabled

### 7.3 Removed Fake `mask_penalty` Config

File:

- `training/config/lora_finetune.yaml`

Current behavior after fix:

- removed the unimplemented `mask_penalty` block
- removed logging of `loss_mask_penalty`

This avoids pretending a loss exists when it does not.

### 7.4 Fixed `demo_visual_hull.py`

File:

- `demo_visual_hull.py`

Current behavior after fix:

- pre-computed masks are resized/cropped/padded to match model input geometry
- unprojected point cloud is converted to torch before further tensor operations
- PLY save path now accepts both numpy and torch safely
- autocast only enables on CUDA

### 7.5 Switched OpenMaterial to GT Mesh Rasterization

Files:

- `training/data/datasets/openmaterial.py`
- `training/config/lora_finetune.yaml`

Current behavior after fix:

- `OpenMaterialDataset` resolves GT mesh from `datasets/groundtruth_ablation/<hash>/clean_<hash>.ply`
- depth supervision is generated by rasterizing the GT mesh per camera view
- there is no fallback to `points3d.ply`
- missing GT mesh now raises an explicit error instead of silently degrading supervision
- the rasterizer includes:
  - near-plane clipping for triangles crossing the camera near plane
  - byte-budgeted LRU caching for rendered depth maps
  - support for both `mask/` and `masks/` directories
  - replayed image geometry augmentation on masks so image/depth/mask stay aligned

## 8. What Was Verified

The following checks were run locally.

- `python -m compileall training vggt demo_visual_hull.py`
- LoRA order smoke test:
  - load base model weights
  - inject LoRA
  - verify wrapped original attention weight matches the base model weight
- lightweight `Aggregator` forward pass with `visual_hull_mask`
- `OpenMaterialDataset` import and sample loading in a newer environment with:
  - `cv2`
  - `torch`
  - `open3d`
- GT mesh rasterization produced dense-enough per-view supervision for OpenMaterial scenes

Not yet verified:

- full training run
- real LoRA fine-tuning run with actual checkpoints
- end-to-end SAM2 + visual hull demo
- throughput / CPU cost of the mesh rasterizer under long training jobs

## 9. Recommended Adaptation Strategy

With masks available, the recommended strategy is:

- adapt the multi-view fusion stage
- keep the single-view prior largely unchanged
- start with the smallest LoRA intervention that matches the target failure mode

### 9.1 Recommended First Configuration

Recommended first experiment:

- target block type: `global`
- target modules: `["qkv"]`
- block indices: late global blocks only

Suggested first block range:

- `[16, 17, 18, 19, 20, 21, 22, 23]`

Interpretation:

- only adapt the later cross-view aggregation layers
- do not touch frame attention yet
- do not touch output heads yet

Reason:

- reflective and transparent problems mainly corrupt cross-view consistency
- masks already help suppress irrelevant background regions
- later global blocks are where high-level geometry fusion is most likely to benefit

## 10. If I Were Choosing LoRA Variants

I would test in this order.

### Option A: Late Global + QKV

Recommended first choice.

Pros:

- minimal intervention
- best preservation of original model prior
- most targeted to the cross-view fusion problem

### Option B: All Global + QKV

Use if late-global adaptation is not enough.

Pros:

- stronger adaptation while still staying focused on cross-view fusion

Cons:

- more trainable parameters
- more risk of shifting the model too far

### Option C: All Global + QKV + PROJ

Use only if the first two are not enough.

Pros:

- more expressive adaptation of attention behavior

Cons:

- heavier
- higher overfitting risk

### Option D: Frame + Global

Not recommended as the first move.

Reason:

- this starts changing single-view representation too early
- with masks available, the main problem is more likely global fusion than frame encoding

### Option E: Heads Only

Not recommended as the primary route for this task.

Reason:

- reflective and transparent failures mostly happen before the heads, in multi-view aggregation

## 11. Project Route

The recommended route is staged and should stay conservative.

### Stage 0: Engineering Stabilization

Goal:

- make the fork behavior match the intended design

Checklist:

- LoRA load order fixed
- masks reach the model
- fake config removed
- visual hull demo no longer crashes on tensor type mismatches

### Stage 1: Baselines

Run three baselines:

1. Original VGGT, no mask-guided attention, no LoRA
2. Original VGGT with mask-guided attention only
3. LoRA version with mask-guided attention

Goal:

- separate the effect of masks from the effect of LoRA

### Stage 2: Minimal LoRA

Configuration:

- late global blocks only
- qkv only

Loss:

- camera + depth only

Depth source:

- OpenMaterial GT mesh rasterization, not point-cloud projection

Goal:

- verify that minimal backbone adaptation is already enough for reflective objects

### Stage 3: Stronger LoRA If Needed

Escalation order:

1. late global + qkv
2. all global + qkv
3. all global + qkv + proj

Only escalate if the previous stage is clearly insufficient.

### Stage 4: Add Proper Mask-Aware Geometry Losses

Only after the LoRA route is validated.

Recommended future directions:

- silhouette consistency loss
- confidence shaping inside vs outside mask

Not recommended:

- a naive generic `mask_penalty` with unclear geometry meaning

### Stage 5: Separate Mirror vs Transparent Analysis

Do not assume reflective and transparent objects will improve equally.

Expected pattern:

- reflective objects should benefit more directly
- transparent objects may improve, but usually remain harder

This stage should explicitly split failure cases by material type.

### Stage 6: Stable Inference Pipeline

Once the training route is validated, stabilize:

- mask input path
- LoRA loading path
- visual hull fusion path
- export path for downstream reconstruction or splatting

## 12. Practical Next Steps

Near-term recommended next steps:

1. Create a dedicated LoRA config for reflective/transparent adaptation using late global blocks only.
2. Run the three-stage baseline comparison.
3. Record qualitative failures by material type.
4. Decide whether `all global + qkv` is needed.
5. Measure the CPU cost of mesh rasterization and decide whether to keep it CPU-side or move it to a faster renderer.
6. Only then design a real mask-aware geometry loss.

## 13. Suggested Tracking Checklist

Use this as a quick working checklist.

- [x] Understand original VGGT structure
- [x] Understand fork additions
- [x] Fix LoRA checkpoint order
- [x] Connect masks to training forward
- [x] Remove fake `mask_penalty` config
- [x] Fix demo tensor type path
- [x] Replace point-cloud depth placeholder with GT mesh rasterization
- [ ] Add a dedicated late-global LoRA config
- [ ] Run baseline without masks
- [ ] Run baseline with masks only
- [ ] Run late-global LoRA with masks
- [ ] Evaluate mirror scenes
- [ ] Evaluate transparent scenes
- [ ] Decide whether to expand LoRA scope
- [ ] Add geometry-aware masked loss if needed

## 14. Files Most Worth Reading Next

For anyone continuing the work, the most important files are:

- `vggt/models/vggt.py`
- `vggt/models/aggregator.py`
- `vggt/lora/lora_layers.py`
- `vggt/lora/lora_utils.py`
- `training/trainer.py`
- `training/config/lora_finetune.yaml`
- `training/data/datasets/openmaterial.py`
- `demo_visual_hull.py`
- `vggt/sam2/visual_hull.py`

## 15. Summary

The best current path for this project is:

- use masks
- adapt only global attention first
- start with LoRA on late global `qkv`
- avoid broad changes until the minimal route is validated

This keeps the project disciplined:

- first make the engineering real
- then isolate gains from masks
- then add the smallest useful adaptation
- then only add stronger constraints if evidence says they are needed
