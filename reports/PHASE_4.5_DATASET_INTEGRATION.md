# VGGT Phong渲染系统 - 阶段4.5进展报告

---

## 报告基本信息

| 项目名称 | VGGT + Phong Rendering System |
|---------|-------------------------------|
| **报告版本** | Phase 4.5 - Dataset Integration |
| **报告阶段** | Phase 4.5 (数据集集成 + 深度监督) |
| **报告生成时间** | 2025-12-14 |
| **基于Commit** | `290f1c9` |
| **当前分支** | `phong` |

---

## 阶段目标与完成情况

### 本阶段目标

| # | 任务 | 状态 |
|---|------|------|
| 1 | 实现GT深度渲染预处理脚本 | ✅ 完成 |
| 2 | 实现OpenMaterialDataset数据加载器 | ✅ 完成 |
| 3 | 添加深度监督损失 (L_depth) | ✅ 完成 |
| 4 | 集成到训练流程 | ✅ 完成 |
| 5 | 更新配置文件 | ✅ 完成 |

### 完成率: 100%

---

## 代码变更汇总

### 新增文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `scripts/render_depth_from_mesh.py` | 471 | GT深度渲染预处理脚本 |
| `training/data/datasets/openmaterial.py` | ~480 | OpenMaterial数据集加载器 |
| `reports/PHASE_4_NORMAL_HEAD_AND_DATA_STRATEGY.md` | 438 | Phase 4 进展报告 |

### 修改文件

| 文件 | 变更 | 描述 |
|------|------|------|
| `training/rendering/phong_loss.py` | +50 lines | 添加深度监督损失 |
| `training/train_phong.py` | +80 lines | 数据集集成、命令行参数 |
| `training/configs/phong_training_config.json` | +15 lines | 数据配置、深度权重 |

---

## 架构设计

### 1. GT深度监督训练架构

```
Input: RGB Images
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                         VGGT                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ DepthHead   │  │ NormalHead  │  │MaterialHead │       │
│  │ (阶段2解冻) │  │  (训练)     │  │  (训练)     │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         │                │                │               │
│         ▼                ▼                ▼               │
│    Pred_Depth      Pred_Normal      Materials            │
└─────────┼────────────────┼────────────────┼──────────────┘
          │                │                │
          │                │                ▼
          │                │         ┌─────────────┐
          │                │         │ LightHead   │
          │                │         └──────┬──────┘
          │                │                │
          │                ▼                ▼
          │         ┌─────────────────────────────┐
          │         │     Phong Renderer          │
          │         └──────────────┬──────────────┘
          │                        │
          │                        ▼
          │                  Rendered_RGB
          │                        │
          ▼                        ▼
    ┌───────────┐           ┌───────────┐
    │ L_depth   │           │  L_rgb    │
    │ (GT监督)  │           │(光度损失) │
    └─────┬─────┘           └─────┬─────┘
          │                       │
          ▼                       │
    ┌───────────┐                 │
    │ L_cons    │◄────────────────┘
    │(法线一致) │
    └─────┬─────┘
          │
          ▼
    L_total = L_rgb + λ_depth × L_depth + λ_cons × L_cons + λ_smooth × L_smooth
```

### 2. 损失函数组成

```python
# PhongLossWithRegularization 输出
{
    'loss_phong_photometric': L_rgb,        # RGB渲染损失 (L1)
    'loss_phong_smoothness': L_smooth,      # 材质平滑度
    'loss_phong_energy': L_energy,          # 能量守恒约束
    'loss_phong_normal_consistency': L_cons, # 法线-深度一致性
    'loss_phong_depth': L_depth,            # [NEW] GT深度监督
    'loss_phong_total': L_total,            # 加权总和
}
```

### 3. 默认损失权重

| 损失项 | 权重 | 作用 |
|--------|------|------|
| L_rgb | 1.0 | 主损失，驱动材质学习 |
| L_depth | 0.5 | 固定几何，消除歧义 |
| L_cons | 0.1 | 约束法线与深度一致 |
| L_smooth | 0.01 | 材质平滑正则化 |
| L_energy | 0.01 | 能量守恒约束 |

---

## 核心实现

### 1. 深度监督损失

**文件**: `training/rendering/phong_loss.py`

```python
def compute_depth_supervision_loss(
    self,
    pred_depth: torch.Tensor,  # (B, S, H, W) 预测深度
    gt_depth: torch.Tensor,    # (B, S, H, W) GT深度
    mask: torch.Tensor = None  # (B, S, H, W) 有效mask
) -> torch.Tensor:
    """
    L1 深度监督损失

    只在有效区域计算 (排除 max_depth 位置)
    """
    loss = torch.abs(pred_depth - gt_depth)

    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-6)
    else:
        loss = loss.mean()

    return loss
```

**自动Mask生成**:
```python
# 在 forward() 中自动创建有效深度mask
depth_mask = (gt_depth < 90.0).float()  # 排除无效区域
if mask is not None:
    combined_mask = depth_mask * mask  # 结合物体mask
```

### 2. OpenMaterial数据集加载器

**文件**: `training/data/datasets/openmaterial.py`

**核心功能**:
- NeRF格式相机参数转换 (OpenGL → OpenCV)
- 非方形图像处理 (短边缩放 + 中心裁剪)
- GT深度图加载 (从预渲染的.npy文件)
- 物体Mask加载

**坐标系转换**:
```python
def _nerf_c2w_to_opencv_w2c(self, c2w: np.ndarray) -> np.ndarray:
    """
    NeRF camera-to-world → OpenCV world-to-camera

    NeRF/OpenGL: Y↑, Z← (相机看向-Z)
    OpenCV:      Y↓, Z→ (相机看向+Z)
    """
    # 1. 翻转 Y 和 Z
    flip = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    c2w_opencv = c2w @ flip

    # 2. camera-to-world → world-to-camera
    w2c_opencv = np.linalg.inv(c2w_opencv)

    return w2c_opencv[:3, :]  # 返回 3x4
```

**短边缩放 + 中心裁剪**:
```python
def _scale_intrinsic(self, K, orig_w, orig_h):
    """
    处理非方形图像 (如 1600x1200 → 518x518)

    策略:
    1. 按短边缩放 (保持宽高比)
    2. 中心裁剪到目标尺寸
    """
    short_side = min(orig_w, orig_h)
    scale = self.img_size / short_side

    # 缩放内参
    K_scaled = K.copy()
    K_scaled[0, :] *= scale  # fx, cx
    K_scaled[1, :] *= scale  # fy, cy

    # 中心裁剪偏移
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    crop_x = (new_w - self.img_size) // 2
    crop_y = (new_h - self.img_size) // 2

    K_scaled[0, 2] -= crop_x  # 调整 cx
    K_scaled[1, 2] -= crop_y  # 调整 cy

    return K_scaled
```

### 3. GT深度渲染脚本

**文件**: `scripts/render_depth_from_mesh.py`

**功能**: 从PLY mesh使用光线追踪渲染深度图

```python
def render_depth_raycast(mesh, c2w, width, height, fx, fy, cx, cy):
    """
    使用 trimesh 光线追踪渲染深度
    """
    # 1. 生成相机光线
    ray_origins, ray_directions = get_camera_rays(
        width, height, fx, fy, cx, cy, c2w
    )

    # 2. 光线-三角形求交
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False
    )

    # 3. 计算深度
    depth = np.full((height * width,), max_depth, dtype=np.float32)
    if len(locations) > 0:
        hit_distances = np.linalg.norm(locations - ray_origins[index_ray], axis=-1)
        depth[index_ray] = hit_distances

    return depth.reshape(height, width)
```

**使用方法**:
```bash
# 单个场景
python scripts/render_depth_from_mesh.py \
    --mesh_dir datasets/groundtruth \
    --data_dir datasets/openmaterial \
    --output_dir datasets/openmaterial_depth \
    --scene_id 5c4ae9c4a3cb47a4b6273eb2839a7b8c

# 批量处理
python scripts/render_depth_from_mesh.py \
    --mesh_dir datasets/groundtruth \
    --data_dir datasets/openmaterial \
    --output_dir datasets/openmaterial_depth \
    --all
```

---

## 训练配置

### 完整配置示例

```json
{
  "experiment_name": "phong_material_light_v1",

  "pretrained_checkpoint": "facebook/VGGT-1B",

  "freeze_patterns": [
    "patch_embed", "aggregator", "camera_head",
    "point_head", "depth_head"
  ],

  "two_stage_training": {
    "enabled": true,
    "unfreeze_depth_at_step": 5000,
    "depth_lr_ratio": 0.1
  },

  "loss": {
    "weight": 1.0,
    "photometric_type": "l1",
    "smoothness_weight": 0.01,
    "energy_weight": 0.01,
    "normal_consistency_weight": 0.1,
    "depth_supervision_weight": 0.5
  },

  "data": {
    "data_dir": "datasets/openmaterial",
    "depth_dir": "datasets/openmaterial_depth",
    "scene_ids": null,
    "split": "train",
    "batch_size": 2,
    "num_workers": 4,
    "img_size": 518,
    "num_views": 4
  }
}
```

### 命令行使用

```bash
# 使用配置文件 + 真实数据
python training/train_phong.py \
    --config training/configs/phong_training_config.json \
    --data_dir datasets/openmaterial \
    --depth_dir datasets/openmaterial_depth \
    --batch_size 2 \
    --epochs 100

# 使用虚拟数据测试
python training/train_phong.py \
    --use_dummy_data \
    --epochs 1

# 命令行参数覆盖配置
python training/train_phong.py \
    --config training/configs/phong_training_config.json \
    --lr 5e-5 \
    --batch_size 4
```

---

## 测试结果

### 深度监督损失测试

```
Loss dict keys: ['loss_phong_photometric', 'loss_phong_normal_consistency',
                 'loss_phong_depth', 'loss_phong_total']

  loss_phong_photometric: 1.002717
  loss_phong_normal_consistency: 0.987694
  loss_phong_depth: 3.370397
  loss_phong_total: 2.786685

Depth supervision loss integration test passed!
```

### 完整训练步骤测试

```
[PhongTrainer] Renderer initialized
[PhongTrainer] Loss function initialized
[PhongTrainer] Optimizer: AdamW, LR: 0.0001
[Monitor] Initialized

Running train_step with gt_depth...

Loss dict:
  loss_phong_photometric: 1.074963
  loss_phong_smoothness: 0.042966
  loss_phong_energy: 0.000007
  loss_phong_normal_consistency: 1.151923
  loss_phong_depth: 4.243756
  loss_phong_total: 3.312463

[SUCCESS] Depth supervision loss is included!
```

---

## 数据流程

### 完整训练数据流

```
1. 预处理阶段 (一次性)
   ┌─────────────────────────────────────────────┐
   │  PLY Mesh + transforms.json                 │
   │         │                                   │
   │         ▼                                   │
   │  render_depth_from_mesh.py                  │
   │         │                                   │
   │         ▼                                   │
   │  GT Depth Maps (.npy)                       │
   └─────────────────────────────────────────────┘

2. 训练阶段
   ┌─────────────────────────────────────────────┐
   │  OpenMaterialDataset                        │
   │    ├── RGB Images                           │
   │    ├── GT Depths (from step 1)              │
   │    ├── Object Masks                         │
   │    ├── Intrinsics (scaled)                  │
   │    └── Extrinsics (converted)               │
   │         │                                   │
   │         ▼                                   │
   │  train_step(batch)                          │
   │    ├── VGGT forward                         │
   │    ├── Phong rendering                      │
   │    └── Loss computation (with L_depth)      │
   └─────────────────────────────────────────────┘
```

### 数据集目录结构

```
datasets/
├── groundtruth/                    # GT Mesh
│   └── {scene_id}/
│       └── clean_{scene_id}.ply
│
├── openmaterial/                   # RGB + Camera
│   └── {scene_id}/
│       ├── train/images/*.png
│       ├── test/images/*.png
│       ├── mask/*.png
│       ├── transforms_train.json
│       └── transforms_test.json
│
└── openmaterial_depth/             # 预渲染深度 (新增)
    └── {scene_id}/
        └── train/depths/
            ├── 000.npy
            ├── 000_vis.png
            └── ...
```

---

## 已知问题与解决方案

### 1. 非方形图像处理

**问题**: OpenMaterial图像为1600x1200，直接resize会导致fx≠fy

**解决**: 使用短边缩放 + 中心裁剪
```
1600x1200 → (短边1200缩放到518) → 691x518 → (中心裁剪) → 518x518
```

### 2. 坐标系不一致

**问题**: NeRF使用OpenGL坐标系，VGGT使用OpenCV坐标系

**解决**: 在数据加载时自动转换
```python
# OpenGL → OpenCV: 翻转Y和Z
# camera-to-world → world-to-camera: 求逆
```

### 3. 无效深度区域

**问题**: 光线追踪可能miss，产生max_depth值

**解决**: 自动生成depth_mask，只在有效区域计算损失
```python
depth_mask = (gt_depth < 90.0).float()
```

---

## 文件清单

### 新增文件 (未提交)

| 文件 | 状态 | 描述 |
|------|------|------|
| `scripts/render_depth_from_mesh.py` | 未跟踪 | GT深度渲染脚本 |
| `training/data/datasets/openmaterial.py` | 未跟踪 | 数据集加载器 |
| `reports/PHASE_4_NORMAL_HEAD_AND_DATA_STRATEGY.md` | 未跟踪 | Phase 4报告 |
| `reports/PHASE_4.5_DATASET_INTEGRATION.md` | 未跟踪 | 本报告 |

### 修改文件 (未暂存)

| 文件 | 描述 |
|------|------|
| `training/rendering/phong_loss.py` | 添加深度监督损失 |
| `training/train_phong.py` | 数据集集成 |
| `training/configs/phong_training_config.json` | 配置更新 |

---

## 下一步计划

| 步骤 | 任务 | 优先级 |
|------|------|--------|
| 1 | 提交当前代码变更 | P0 |
| 2 | 在实际数据上测试训练流程 | P0 |
| 3 | 渲染几个场景的GT深度图 | P0 |
| 4 | 调优损失权重 | P1 |
| 5 | 添加验证集评估 | P1 |
| 6 | 可视化训练过程 | P2 |

---

## 总结

### 本阶段成果

| 类别 | 内容 |
|------|------|
| **新增代码** | ~1000 lines |
| **新增功能** | GT深度监督、OpenMaterial数据加载 |
| **架构改进** | 完整的训练数据流 |
| **工具脚本** | PLY→深度图渲染 |

### 训练系统完整度

```
[✓] MaterialHead - 材质预测
[✓] LightHead - 光照预测
[✓] NormalHead - 法线预测
[✓] PhongRenderer - 可微渲染
[✓] PhongLoss - 多项损失
[✓] Two-stage Training - 两阶段训练
[✓] OpenMaterialDataset - 数据加载
[✓] Depth Supervision - GT深度监督
[ ] Real Data Testing - 真实数据测试
[ ] Validation - 验证评估
```

### 当前状态

**训练系统核心功能100%完成**，可以开始在真实数据上训练。

---

**报告完成时间**: 2025-12-14
**撰写**: Claude Code (AI Assistant)
**状态**: Phase 4.5 完成，准备真实数据训练

---

**END OF PROGRESS REPORT**
