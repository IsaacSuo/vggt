# Training

This is a re-implementation of our framework for training VGGT. This document shows how to set up the environment and run VGGT training. I have aimed to faithfully reproduce the original training framework, but please open an issue if anything looks off.

## 1. Prerequisites

Before you begin, ensure you have completed the following steps:

1. **Install VGGT as a package:**
   ```bash
   pip install -e .
   ```

2. **Install training-only dependencies used by the current training stack:**
   ```bash
   pip install iopath fvcore tensorboard
   ```

3. **Prepare the dataset and annotations:**
   - Download the Co3D dataset from the [official repository](https://github.com/facebookresearch/co3d).
   - Download the required annotation files from [Hugging Face](https://huggingface.co/datasets/JianyuanWang/co3d_anno/tree/main).

## 2. Configuration

After downloading the dataset and annotations, configure the paths in `training/config/default.yaml`.

### Required Path Configuration

1. Open `training/config/default.yaml`
2. Update the following paths with your absolute directory paths:
   - `CO3D_DIR`: Path to your Co3D dataset
   - `CO3D_ANNOTATION_DIR`: Path to your Co3D annotation files
   - `resume_checkpoint_path`: Path to your pre-trained VGGT checkpoint

### Configuration Example

```yaml
data:
  train:
    dataset:
      dataset_configs:
        - _target_: data.datasets.co3d.Co3dDataset
          split: train
          CO3D_DIR: /YOUR/PATH/TO/CO3D
          CO3D_ANNOTATION_DIR: /YOUR/PATH/TO/CO3D_ANNOTATION
# ... same for val ...

checkpoint:
  resume_checkpoint_path: /YOUR/PATH/TO/CKPT
```

## 3. Fine-tuning on Co3D

To fine-tune the provided pre-trained model on the Co3D dataset, run the following command. This example uses 4 GPUs with PyTorch Distributed Data Parallel (DDP):

```bash
torchrun --nproc_per_node=4 launch.py
```

The default configuration in `training/config/default.yaml` is set up for fine-tuning. It automatically resumes from a checkpoint and freezes the model's `aggregator` module during training.

## 4. OpenMaterial Fine-tuning

This fork also contains a dedicated config for reflective / transparent material adaptation:

- config: `training/config/lora_finetune.yaml`
- dataset class: `training/data/datasets/openmaterial.py`

### Expected OpenMaterial layout

The loader expects scene folders under:

```text
OpenMaterial/datasets/
  <hash>/
    <scene_name>/
      transforms_train.json
      transforms_test.json
      train/
        images/
        mask/      # or masks/
      test/
        images/
        mask/      # or masks/
  groundtruth_ablation/
    <hash>/
      clean_<hash>.ply
```

Important notes:

- `train/images` and `test/images` are read from `transforms_*.json`
- the loader supports both `mask/` and `masks/`
- depth supervision is generated from the GT mesh `clean_<hash>.ply`
- there is no point-cloud fallback anymore; if the GT mesh is missing, dataset loading raises an error

### What the OpenMaterial loader does

For each frame:

- read RGB image
- read foreground mask if available
- resolve camera intrinsics and extrinsics from `transforms_*.json`
- rasterize the GT mesh into a per-view depth map
- apply the same crop / resize / augmentation path to image, depth, and mask

The current rasterizer is CPU-side and includes:

- near-plane clipping for triangles crossing the camera near plane
- byte-budgeted LRU caching of rendered depth maps
- optional reuse of offline precomputed mesh-depth `.npy` files

The two most relevant knobs in `lora_finetune.yaml` are:

- `mesh_near_plane`
- `depth_cache_max_mb`
- `depth_precompute_dir`
- `depth_precompute_subdir`

### Required path configuration

Update the following fields in `training/config/lora_finetune.yaml`:

- `data.train.dataset.dataset_configs[0].data_dir`
- `data.val.dataset.dataset_configs[0].data_dir`
- `checkpoint.resume_checkpoint_path`

### Launch command

Run from the `training/` directory:

```bash
torchrun --nproc_per_node=4 launch.py --config lora_finetune
```

Single-GPU variant:

```bash
PYTHONPATH=/YOUR/PATH/TO/vggt \
torchrun --standalone --nnodes=1 --nproc_per_node=1 launch.py --config lora_finetune
```

Important launcher note:

- one real server had `python=/root/om/bin/python` but `torchrun=/usr/local/bin/torchrun`, and training failed with `ModuleNotFoundError: No module named 'hydra'`
- on that server, prefer launching with the active environment explicitly:

```bash
PYTHONPATH=/YOUR/PATH/TO/vggt \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 launch.py --config lora_finetune
```

For the current OpenMaterial server runs, this repo now also includes:

- config: `training/config/openmaterial_probe_server.yaml`
- entry script: `training/run_openmaterial_probe_server.sh`
- config: `training/config/openmaterial_train_server.yaml`
- entry script: `training/run_openmaterial_train_server.sh`
- config: `training/config/openmaterial_probe_server_disjoint.yaml`
- entry script: `training/run_openmaterial_probe_server_disjoint.sh`
- config: `training/config/openmaterial_train_server_disjoint.yaml`
- entry script: `training/run_openmaterial_train_server_disjoint.sh`
- split helper: `training/data/preprocess/openmaterial_scene_split.py`
- split helper entry script: `training/run_openmaterial_scene_split_server.sh`

These entries bake in the validated server paths, `img_size=518`, cached-depth settings, and the validated launch method. The probe config also keeps the conservative 1-GPU probe batch shape, while the train config inherits the normal `lora_finetune.yaml` training hyperparameters.

After the first 1-GPU disjoint training attempt on the RTX 5090, the inherited larger dynamic image count was still too aggressive and produced a CUDA OOM inside global attention. The fixed server train configs now default to a conservative `max_img_per_gpu=3` and `img_nums=[2,3]`, which is a safer starting point for real training bring-up.

If you want to override paths from the command line:

```bash
torchrun --nproc_per_node=4 launch.py --config lora_finetune \
  data.train.dataset.dataset_configs.0.data_dir=/YOUR/PATH/TO/OpenMaterial/datasets \
  data.val.dataset.dataset_configs.0.data_dir=/YOUR/PATH/TO/OpenMaterial/datasets \
  checkpoint.resume_checkpoint_path=/YOUR/PATH/TO/VGGT/model.pt
```

If you already downloaded the checkpoint into this fork, a concrete 1-GPU command looks like:

```bash
cd training
PYTHONPATH=/home/fangsuo/py/vggt \
torchrun --standalone --nnodes=1 --nproc_per_node=1 launch.py --config lora_finetune \
  data.train.dataset.dataset_configs.0.data_dir=/home/fangsuo/py/OpenMaterial/datasets \
  data.val.dataset.dataset_configs.0.data_dir=/home/fangsuo/py/OpenMaterial/datasets \
  checkpoint.resume_checkpoint_path=/home/fangsuo/py/vggt/checkpoints/model.pt
```

### Offline mesh-depth precompute

The slow part of the first OpenMaterial batch is usually the CPU mesh rasterizer.
If you want training startup to be much faster, precompute the raw mesh depth once and
let the dataset load `.npy` files from disk.

Default behavior after this change:

- the dataset first looks for precomputed depth files
- if found, it loads them directly
- if missing, it falls back to the old online CPU rasterizer unless `require_precomputed_depth=True`

By default the expected per-frame cache path is:

```text
<scene_dir>/
  train/
    images/000.png
    depth_mesh/000.npy
  test/
    images/000.png
    depth_mesh/000.npy
```

If you do not want to write into the dataset tree, you can place the cache under a separate root
with the same `<hash>/<scene_name>/train|test/depth_mesh/*.npy` structure and point
`depth_precompute_dir` at it.

Example precompute command:

```bash
python training/data/preprocess/openmaterial_depth_cache.py \
  --data_dir /home/fangsuo/py/OpenMaterial/datasets \
  --split both \
  --output_root /home/fangsuo/py/OpenMaterial/depth_cache
```

GPU variant with `nvdiffrast`:

```bash
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

CUDA_HOME=/usr/local/cuda-13.0 \
TORCH_CUDA_ARCH_LIST=12.0 \
PYTHONPATH=/home/fangsuo/py/vggt:/home/fangsuo/py/vggt/training \
python training/data/preprocess/openmaterial_depth_cache.py \
  --data_dir /home/fangsuo/py/OpenMaterial/datasets \
  --split both \
  --output_root /home/fangsuo/py/OpenMaterial/depth_cache \
  --backend nvdiffrast \
  --frame_batch_size 8
```

Important installation note for `nvdiffrast`:

- it is installed from GitHub, not from PyPI
- if the build fails with `.../nvvm/bin/cicc: not found`, your `CUDA_HOME` or `/usr/local/cuda` path likely points at an incomplete toolkit path instead of the real CUDA root
- on Ada/Blackwell-class GPUs, explicitly setting `TORCH_CUDA_ARCH_LIST` can avoid unnecessary arch compilation

Then launch training with:

```bash
cd training
PYTHONPATH=/home/fangsuo/py/vggt \
torchrun --standalone --nnodes=1 --nproc_per_node=1 launch.py --config lora_finetune \
  data.train.dataset.dataset_configs.0.data_dir=/home/fangsuo/py/OpenMaterial/datasets \
  data.val.dataset.dataset_configs.0.data_dir=/home/fangsuo/py/OpenMaterial/datasets \
  checkpoint.resume_checkpoint_path=/home/fangsuo/py/vggt/checkpoints/model.pt \
  data.train.common_config.depth_precompute_dir=/home/fangsuo/py/OpenMaterial/depth_cache \
  data.val.common_config.depth_precompute_dir=/home/fangsuo/py/OpenMaterial/depth_cache
```

On that server, use:

```bash
cd training
PYTHONPATH=/home/fangsuo/py/vggt \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 launch.py --config lora_finetune \
  data.train.dataset.dataset_configs.0.data_dir=/home/fangsuo/py/OpenMaterial/datasets \
  data.val.dataset.dataset_configs.0.data_dir=/home/fangsuo/py/OpenMaterial/datasets \
  checkpoint.resume_checkpoint_path=/home/fangsuo/py/vggt/checkpoints/model.pt \
  data.train.common_config.depth_precompute_dir=/home/fangsuo/py/OpenMaterial/depth_cache \
  data.val.common_config.depth_precompute_dir=/home/fangsuo/py/OpenMaterial/depth_cache
```

### Minimal cache-path verification

If you want to verify that training really reads the offline depth cache instead of silently
falling back to CPU rasterization, run a tiny smoke job with `require_precomputed_depth=True`.

Concrete example used locally:

```bash
cd training
PYTHONPATH=/home/fangsuo/py/vggt \
torchrun --standalone --nnodes=1 --nproc_per_node=1 launch.py --config lora_finetune \
  logging.log_dir=/tmp/vggt_lora_verify_precomputed_small \
  num_workers=0 \
  max_img_per_gpu=2 \
  data.train.max_img_per_gpu=2 \
  data.val.max_img_per_gpu=2 \
  data.train.common_config.img_nums=[2,2] \
  data.val.common_config.img_nums=[2,2] \
  limit_train_batches=1 \
  limit_val_batches=0 \
  max_epochs=1 \
  val_epoch_freq=1000 \
  data.train.dataset.dataset_configs.0.data_dir=/tmp/vggt_lora_smoke_subset \
  data.val.dataset.dataset_configs.0.data_dir=/tmp/vggt_lora_smoke_subset \
  checkpoint.resume_checkpoint_path=/home/fangsuo/py/vggt/checkpoints/model.pt \
  data.train.common_config.depth_precompute_dir=/tmp/vggt_lora_smoke_depth_cache \
  data.val.common_config.depth_precompute_dir=/tmp/vggt_lora_smoke_depth_cache \
  data.train.common_config.require_precomputed_depth=True \
  data.val.common_config.require_precomputed_depth=True
```

On that server, the same probe should be launched with:

```bash
cd training
PYTHONPATH=/home/fangsuo/py/vggt \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 launch.py --config lora_finetune \
  logging.log_dir=/tmp/vggt_lora_verify_precomputed_small \
  num_workers=0 \
  max_img_per_gpu=2 \
  data.train.max_img_per_gpu=2 \
  data.val.max_img_per_gpu=2 \
  data.train.common_config.img_nums=[2,2] \
  data.val.common_config.img_nums=[2,2] \
  limit_train_batches=1 \
  limit_val_batches=0 \
  max_epochs=1 \
  val_epoch_freq=1000 \
  data.train.dataset.dataset_configs.0.data_dir=/tmp/vggt_lora_smoke_subset \
  data.val.dataset.dataset_configs.0.data_dir=/tmp/vggt_lora_smoke_subset \
  checkpoint.resume_checkpoint_path=/home/fangsuo/py/vggt/checkpoints/model.pt \
  data.train.common_config.depth_precompute_dir=/tmp/vggt_lora_smoke_depth_cache \
  data.val.common_config.depth_precompute_dir=/tmp/vggt_lora_smoke_depth_cache \
  data.train.common_config.require_precomputed_depth=True \
  data.val.common_config.require_precomputed_depth=True
```

The equivalent fixed entrypoint is:

```bash
cd /opt/data/private/fyp/vggt
training/run_openmaterial_probe_server.sh
```

If you still need a one-off override, append it to the script call, for example:

```bash
cd /opt/data/private/fyp/vggt
training/run_openmaterial_probe_server.sh logging.log_dir=/opt/data/private/fyp/vggt_runs/probe_alt
```

For full training on that server, use:

```bash
cd /opt/data/private/fyp/vggt
training/run_openmaterial_train_server.sh
```

### Scene-disjoint split for benchmark

Using all 105 scenes for training and validation is acceptable for bring-up and ablations, but it is not a clean benchmark because validation still comes from scenes seen during training.

To create a held-out scene split on the server:

```bash
cd /opt/data/private/fyp/vggt
training/run_openmaterial_scene_split_server.sh
```

This writes:

- `/opt/data/private/fyp/vggt_runs/splits/openmaterial_scene_split_seed42/train.txt`
- `/opt/data/private/fyp/vggt_runs/splits/openmaterial_scene_split_seed42/test.txt`
- `/opt/data/private/fyp/vggt_runs/splits/openmaterial_scene_split_seed42/summary.json`

The manifest format is one scene identifier per line. Each entry can be either `scene_name` or `hash_id/scene_name`; the split script writes the safer `hash_id/scene_name` form.

After generating the manifests, use the disjoint probe/train entrypoints:

```bash
cd /opt/data/private/fyp/vggt
training/run_openmaterial_probe_server_disjoint.sh
training/run_openmaterial_train_server_disjoint.sh
```

What this proves:

- if any requested cache file is missing, dataset loading fails immediately
- if the run reaches `Train Epoch`, the training input path has already consumed valid cached depths
- on this machine, a small dynamic batch such as `img_nums=[2,2]` and `max_img_per_gpu=2` is a safer probe than the default larger batch

Note:

- keep `img_size=518` when using the released `model.pt`
- do not use this tiny verification batch as the final training configuration

### Current caveats

- Some OpenMaterial scenes may not have masks in the exact location expected by the loader. The loader handles `mask/` and `masks/`, but it still assumes one mask file per frame.
- Mesh rasterization still has a CPU fallback, but the preferred precompute route is now the GPU `nvdiffrast` backend when available.
- The first full OpenMaterial batch can still be slow without offline depth precompute, because mesh depth generation happens inside dataloader workers.
- The depth supervision is only as good as the GT mesh and camera alignment.

## 5. Training on Multiple Datasets

The dataloader supports multiple datasets naturally. For example, if you have downloaded VKitti using `preprocess/vkitti.sh`, you can train on Co3D+VKitti by configuring:

```yaml
data:
  train:
    dataset:
      _target_: data.composed_dataset.ComposedDataset
      dataset_configs:
        - _target_: data.datasets.co3d.Co3dDataset
          split: train
          CO3D_DIR: /YOUR/PATH/TO/CO3D
          CO3D_ANNOTATION_DIR: /YOUR/PATH/TO/CO3D_ANNOTATION
          len_train: 100000
        - _target_: data.datasets.vkitti.VKittiDataset
          split: train
          VKitti_DIR: /YOUR/PATH/TO/VKitti
          len_train: 100000
          expand_ratio: 8 
```

The ratio of different datasets can be controlled by setting `len_train`. For example, Co3D with `len_train: 10000` and VKitti with `len_train: 2000` will result in Co3D being sampled five times more frequently than VKitti.

## 6. Common Questions

### Memory Management

If you encounter out-of-memory (OOM) errors on your GPU, consider adjusting the following parameters in `training/config/default.yaml`:

- `max_img_per_gpu`: Reduce this value to decrease the batch size per GPU
- `accum_steps`: Sets the number of gradient accumulation steps (default is 2). This feature splits batches into smaller chunks to save memory, though it may slightly increase training time. Note that gradient accumulation was not used for the original VGGT model.

### Learning Rate Tuning

The main hyperparameter to be careful about is learning rate. Note that learning rate depends on the effective batch size, which is `batch_size_per_gpu × num_gpus`. Therefore, I highly recommend trying several learning rates based on your training setup. Generally, trying values like `5e-6`, `1e-5`, `5e-5`, `1e-4`, `5e-4` should be sufficient.

For the current OpenMaterial LoRA config, the most sensitive knobs are usually:

- learning rate
- `block_indices`
- `max_img_per_gpu`
- `mesh_near_plane`
- `depth_cache_max_mb`

### Tracking Head

The tracking head can slightly improve accuracy but is not necessary. For general cases, especially when GPU resources are limited, we suggest fine-tuning the pre-trained model only with camera and depth heads, which is the setting in `default.yaml`. This will provide good enough results.

### Dataloader Validation

To check if your dataloader is working correctly, the best approach is to visualize its output. You can save the 3D world points as follows and then visually inspect the PLY files:

```python
def save_ply(points, colors, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

# Usage example
save_ply(
    batch["world_points"][0].reshape(-1, 3), 
    batch["images"][0].permute(0, 2, 3, 1).reshape(-1, 3), 
    "debug.ply"
)
```

### Handling Unordered Sequences

For unordered sequences, you can check how we compute the ranking (similarity) between one frame and all other frames, as discussed in [Issue #82](https://github.com/facebookresearch/vggt/issues/82).

### Expected Coordinate System

Camera poses are expected to follow the OpenCV `camera-from-world` convention. Depth maps should be aligned with their corresponding camera poses.
