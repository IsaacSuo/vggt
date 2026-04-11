# Standalone Benchmark

This benchmark path is intentionally separate from `training/Trainer`.

Use it when you want to compare:

- multiple checkpoints
- across one or more benchmark datasets
- with a stable inference-first evaluation loop

instead of reusing `mode=val` from the training stack.

## How it works

1. Write a benchmark plan JSON.
2. Register each model explicitly, including LoRA settings if needed.
3. Register each dataset through a dataset adapter.
4. Run the benchmark once and collect:
   - `per_sample.json`
   - `per_sample.csv`
   - `summary.json`
   - `summary.csv`
   - `summary.md`

## Current adapter support

- `openmaterial`
- `nero_glossy_synthetic`
- `nero_glossy_real`

The OpenMaterial adapter reuses the existing dataset preprocessing path so images,
masks, depths, and camera targets stay aligned with the current training semantics.

The NeRO GlossySynthetic adapter reads the official `GlossySynthetic.tar.gz`
release directly, uses per-frame `camera.pkl` and `depth.png`, and evaluates
reconstruction against each scene's `eval_pts.ply` point cloud.

The NeRO GlossyReal adapter reads `images/*.jpg` plus COLMAP metadata from the
official `GlossyReal.tar.gz` release, uses COLMAP cameras as GT, filters camera
pairs by shared sparse-track support from `images.bin`, and evaluates
reconstruction against each scene's `object_point_cloud.ply`. For object-level
reconstruction filtering, it projects `object_point_cloud.ply` into each selected
frame to build benchmark masks, then applies those masks before TSDF fusion.
Because GlossyReal does not provide
per-frame GT depth, depth metrics are skipped for this dataset.

## OpenMaterial protocol

Camera:

- `auc3`
- `auc30`
- evaluate only covisible frame pairs
- relative rotation error in degrees
- relative translation direction error in degrees
- joint pair error = `max(rot_err, trans_dir_err)`
- if `|t_gt| < epsilon`, drop translation direction error and keep the rotation error for that pair

Reconstruction:

- use `depth + camera -> TSDF -> fused point cloud`
- apply the benchmark valid mask before fusion
- normalize prediction density through TSDF extraction plus fixed-count surface sampling
- `cd_l1` normalized by GT mesh bbox diagonal
- `f1@1%`
- `f1@5%`

Depth:

- `delta1`
- `absrel`

Aggregation and sampling:

- one OpenMaterial scene = one benchmark sample
- aggregate scene-level rows with macro-average
- frame sampling is fixed-count `evenly_spaced` unless the plan overrides it

Operational note:

- if the dataset tree does not already contain `depth_mesh/*.npy`, benchmark runs can fall back to online CPU mesh rasterization
- for real runs, precompute mesh-depth caches first with `training/data/preprocess/openmaterial_depth_cache.py`

## Example

Start from:

- [openmaterial_scene_disjoint_plan.json](/home/fangsuo/py/vggt/benchmark/examples/openmaterial_scene_disjoint_plan.json)

Run:

```bash
cd /opt/data/private/fyp/vggt
PYTHONPATH=/opt/data/private/fyp/vggt:/opt/data/private/fyp/vggt/training \
python benchmark/run.py \
  --plan benchmark/examples/openmaterial_scene_disjoint_plan.json \
  --output-dir /opt/data/private/fyp/vggt_runs/benchmark_eval
```

For NeRO GlossyReal qualitative figures from dataset viewpoints:

```bash
cd /opt/data/private/fyp/vggt
PYTHONPATH=/opt/data/private/fyp/vggt:/opt/data/private/fyp/vggt/training \
python benchmark/visualize_nero_glossyreal.py \
  --plan benchmark/examples/nero_glossyreal_plan.json \
  --scene bear \
  --models baseline_vggt lora_openmaterial \
  --output-dir /opt/data/private/fyp/vggt_runs/benchmark_viz/nero_glossyreal_bear
```

## Download helpers

This repo also includes benchmark dataset download helpers under `scripts/`:

- `scripts/download_benchmark_scene.py`
  - intended for downloading one scene locally for adapter development
  - supports `TransLab` partial scene downloads
  - for `NeRO`, the official release is not scene-granular, so the local helper downloads the requested official subset package instead
- `scripts/download_benchmark_all.py`
  - intended for downloading full datasets on a server for benchmark runs

Examples:

```bash
python scripts/download_benchmark_scene.py \
  --dataset translab \
  --scene scene_01 \
  --output-root /data/local_benchmarks

python scripts/download_benchmark_scene.py \
  --dataset nero \
  --nero-subset GlossySynthetic \
  --scene bell \
  --output-root /data/local_benchmarks

python scripts/download_benchmark_all.py \
  --dataset all \
  --output-root /data/server_benchmarks
```

## Extending to another dataset

Add a new adapter under `benchmark/adapters/` and register it in
`benchmark/registry.py`.
