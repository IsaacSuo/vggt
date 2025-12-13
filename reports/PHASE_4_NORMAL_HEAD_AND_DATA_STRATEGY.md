# VGGT Phong渲染系统 - 阶段四进展报告

---

## 📋 报告基本信息

| 项目名称 | VGGT + Phong Rendering System |
|---------|-------------------------------|
| **报告版本** | Phase 4 Progress v1.0 |
| **报告阶段** | Phase 4 (NormalHead + 数据策略) |
| **报告生成时间** | 2025-12-14 |
| **当前HEAD** | `290f1c9` |
| **当前分支** | `phong` |

---

## 🎯 Phase 4 目标与进展

### 原始目标
1. ✅ 重命名 PBR → Phong（命名规范化）
2. ✅ 修复训练流程 bug
3. ✅ 支持 HuggingFace 预训练权重加载
4. ✅ 实现 NormalHead（直接预测法线）
5. ✅ 添加法线-深度一致性约束
6. ✅ 实现两阶段训练策略
7. 🔄 数据集集成（进行中）

### 达成情况
**核心功能 100% 完成** - NormalHead + 两阶段训练已实现并测试通过

---

## 📊 代码变更汇总

### Commit 历史

| Commit | 描述 | 变更文件数 |
|--------|------|-----------|
| `290f1c9` | Add NormalHead with depth consistency constraint | 4 |
| `af5ed26` | Support loading pretrained weights from HuggingFace | 1 |
| `fb63ba7` | Fix training pipeline bugs | 3 |
| `3ed3bf1` | [Phase 4] Add learnable lighting and Phong training infrastructure | 8 |

### 新增/修改文件

```
vggt/
├── heads/
│   └── normal_head.py          [NEW] 223 lines - 法线预测头
├── models/
│   └── vggt.py                 [MOD] +15 lines - 集成NormalHead

training/
├── train_phong.py              [MOD] +80 lines - 两阶段训练逻辑
├── configs/
│   └── phong_training_config.json [MOD] +5 lines - 两阶段配置
└── rendering/
    └── phong_loss.py           [MOD] +50 lines - 法线一致性损失
```

---

## 🏗️ 架构变更

### 1. NormalHead 实现

**设计决策**: 直接预测法线，而非从深度计算

**原因**:
- 深度图可能不准确，导致法线计算错误
- 直接预测可以捕获高频细节
- 通过一致性约束保持与深度的几何关系

**架构** (与其他DPT Head一致):
```python
class NormalHead(nn.Module):
    """
    预测表面法线的DPT头
    输入: aggregated_tokens (B, S, N, D)
    输出: normals (B, S, H, W, 3) 单位法线
    """
    def __init__(self, dim_in=768, patch_size=14, features=256, ...):
        # 4个Reassemble层 + 4个Fusion层 + 输出卷积
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, 1, 0),  # 输出3通道
        )

    def forward(self, ...):
        out = self._forward_impl(...)
        normals = F.normalize(out, p=2, dim=-1, eps=1e-6)  # 单位化
        return normals
```

**参数量**: 32.65M (与 depth_head, point_head, material_head 相同)

### 2. 法线-深度一致性约束

**Loss 设计**:
```python
def compute_normal_consistency_loss(self, predicted_normals, depth, depth_conf=None):
    """
    约束预测法线与深度导出法线的一致性

    L_cons = 1 - cos(normal_pred, normal_from_depth)
    """
    # 从深度计算法线 (Sobel梯度)
    depth_normals = self.depth_to_normals(depth)

    # 余弦距离
    cosine_sim = (predicted_normals * depth_normals).sum(dim=-1)
    cosine_dist = 1.0 - cosine_sim

    # 可选：用深度置信度加权
    if depth_conf is not None:
        conf_weight = depth_conf / (depth_conf.mean() + 1e-6)
        cosine_dist = cosine_dist * conf_weight

    return cosine_dist.mean()
```

**作用**:
- 单向约束（当前）: NormalHead 学习匹配 DepthHead 输出
- 双向约束（阶段二）: 解冻 DepthHead 后，渲染梯度可以修正深度

### 3. 两阶段训练策略

**问题**: 一致性约束是单向的，如果 depth_head 冻结：
- NormalHead 被迫学习 DepthHead 的噪声
- 渲染梯度无法修正几何

**解决方案**: 两阶段训练

| 阶段 | Steps | depth_head | 学习率 | 目的 |
|------|-------|-----------|--------|------|
| 1 (Warm-up) | 0-4999 | 冻结 | - | Material/Light/Normal 先学基础 |
| 2 (Fine-tune) | 5000+ | 解冻 | base_lr × 0.1 | 双向梯度流，精修几何 |

**实现**:
```python
def _maybe_unfreeze_depth(self):
    """检查是否应该解冻depth_head"""
    two_stage_config = self.config.get('two_stage_training', {})

    if not two_stage_config.get('enabled', False):
        return

    unfreeze_step = two_stage_config.get('unfreeze_depth_at_step', 5000)

    if self.global_step >= unfreeze_step:
        self._unfreeze_depth_head()

def _unfreeze_depth_head(self):
    """解冻depth_head，使用更小的学习率"""
    depth_lr = base_lr * depth_lr_ratio  # 0.1x

    for param in self.model.depth_head.parameters():
        param.requires_grad = True

    # 添加到optimizer，独立学习率
    self.optimizer.add_param_group({
        'params': list(self.model.depth_head.parameters()),
        'lr': depth_lr,
        'name': 'depth_head'
    })
```

**配置**:
```json
"two_stage_training": {
    "enabled": true,
    "unfreeze_depth_at_step": 5000,
    "depth_lr_ratio": 0.1
}
```

---

## 🧪 测试结果

### NormalHead 集成测试

```
[PhongTrainer] material_head: 32.65M total, 32.65M trainable
[PhongTrainer] light_head: 1.18M total, 1.18M trainable
[PhongTrainer] normal_head: 32.65M total, 32.65M trainable

loss/loss_phong_normal_consistency: 0.000131
loss/loss_phong_total: 0.250118
[Main] Training completed!
```

### 两阶段训练测试

```
Step 1-3: depth_unfrozen=False, requires_grad=False  (阶段1)
[Stage 2: Unfreeze depth_head] Unfrozen 32.65M params, LR=1e-05
Step 4-6: depth_unfrozen=True, requires_grad=True    (阶段2)

Optimizer param groups:
  Group 0 (default): lr=1.00e-04   <- material, light, normal
  Group 1 (depth_head): lr=1.00e-05  <- depth (10x smaller)
```

### 模型参数统计

| 组件 | 参数量 | 训练状态 |
|------|--------|---------|
| aggregator | 909.11M | 冻结 |
| camera_head | 216.17M | 冻结 |
| point_head | 32.65M | 冻结 |
| depth_head | 32.65M | 阶段1冻结→阶段2解冻 |
| track_head | 65.94M | 冻结 |
| material_head | 32.65M | 训练 |
| light_head | 1.18M | 训练 |
| normal_head | 32.65M | 训练 |
| **总计** | **1.32B** | |

---

## 📁 数据集策略

### 目标数据集: OpenMaterial

**结构**:
```
datasets/
├── groundtruth/
│   └── {scene_id}/
│       └── clean_{scene_id}.ply      # GT Mesh
├── openmaterial/
│   └── {scene_id}/
│       ├── train/images/*.png        # RGB图像
│       ├── test/images/*.png
│       ├── mask/                     # 物体掩码
│       ├── transforms_train.json     # 相机参数
│       └── transforms_test.json
```

**数据格式** (NeRF/Instant-NGP):
```json
{
    "fl_x": 2333.33, "fl_y": 2333.33,  // 焦距
    "cx": 800, "cy": 600,              // 主点
    "w": 1600, "h": 1200,              // 图像尺寸
    "frames": [
        {
            "file_path": "train/images/000.png",
            "transform_matrix": [[4x4]]  // camera-to-world (OpenGL)
        }
    ]
}
```

### 训练策略决策

**问题**: 可微分渲染是病态问题
```
I ≈ G(Geometry) × M(Material) × L(Lighting)
```
给定图像 I，有无数种 G, M, L 组合。

**方案对比**:

| 方案 | 深度来源 | 优点 | 缺点 |
|------|---------|------|------|
| A | VGGT预测 | 简单 | 深度可能不准，材质学偏 |
| B | GT Mesh渲染 | 几何准确 | 需要预处理 |
| **A+B混合** | GT监督VGGT | 最佳 | 需要预处理 |

**最终决策**: A+B 混合策略

### GT 引导训练架构

```
Input: RGB
   │
   ▼
VGGT (DepthHead 解冻)
   │
   ├──► Pred_Depth ──► L_depth ◄── GT_Depth (从PLY渲染)
   │         │
   │         ▼
   ├──► Pred_Normal ──► L_cons ◄── Derived(Pred_Depth)
   │         │
   │         ▼
   │    Material + Light
   │         │
   │         ▼
   └──► Phong Render ──► L_rgb ◄── Target RGB
```

**Loss 组合**:
```python
L_total = L_rgb + λ_depth * L_depth + λ_cons * L_cons + λ_smooth * L_smooth
```

**优势**:
1. **消除歧义**: L_depth 固定几何，迫使 L_rgb 优化材质/光照
2. **提升能力**: VGGT depth_head 学习准确深度
3. **推断独立**: 训练后可扔掉 GT Mesh
4. **细节保留**: L_cons 让 NormalHead 学习高频细节

---

## 📋 待实施任务

### 下一步实施计划

| 步骤 | 任务 | 预估时间 | 优先级 |
|------|------|---------|--------|
| 1 | 预处理脚本: PLY → 深度图 | 1h | P0 |
| 2 | OpenMaterialDataset 类 | 2h | P0 |
| 3 | 添加 L_depth 监督损失 | 30min | P0 |
| 4 | 修改训练配置 | 15min | P0 |
| 5 | 端到端测试 | 1h | P0 |

### 预处理脚本需求

```python
# scripts/render_depth_from_mesh.py

def render_depth(mesh_path, camera_params, output_dir):
    """
    从PLY mesh渲染深度图

    Args:
        mesh_path: GT mesh 路径
        camera_params: transforms.json 中的相机参数
        output_dir: 输出深度图目录

    Output:
        每帧对应的 depth_xxx.npy 或 depth_xxx.png
    """
    # 使用 PyTorch3D 或 Trimesh + PyRender
```

### OpenMaterialDataset 需求

```python
class OpenMaterialDataset(BaseDataset):
    """
    OpenMaterial 数据集加载器

    输出:
        images: (S, 3, H, W)
        depths: (S, H, W)        # 从PLY预渲染
        masks: (S, H, W)         # 物体掩码
        extrinsics: (S, 3, 4)    # world-to-camera (OpenCV)
        intrinsics: (S, 3, 3)
    """
```

---

## ⚠️ 注意事项

### 坐标系转换

**NeRF 格式**: camera-to-world, OpenGL 约定
```
Y ↑    Z (后)
  |   /
  |  /
  | /
  +------→ X
```

**VGGT 格式**: world-to-camera, OpenCV 约定
```
      Z (前)
     /
    /
   +------→ X
   |
   ↓ Y
```

**转换步骤**:
1. OpenGL → OpenCV: 翻转 Y 和 Z
2. camera-to-world → world-to-camera: 求逆

```python
def convert_nerf_to_opencv(c2w_opengl):
    """NeRF camera-to-world → OpenCV world-to-camera"""
    # 1. OpenGL to OpenCV
    c2w_opencv = c2w_opengl.copy()
    c2w_opencv[:, 1:3] *= -1  # 翻转 Y, Z

    # 2. camera-to-world to world-to-camera
    w2c_opencv = np.linalg.inv(c2w_opencv)

    return w2c_opencv[:3, :]  # 返回 3x4
```

---

## 📈 总结

### 阶段四成果

| 类别 | 内容 |
|------|------|
| **新增代码** | ~400 lines |
| **新增模块** | NormalHead |
| **新增功能** | 法线预测、一致性约束、两阶段训练 |
| **架构改进** | 解耦几何与材质学习 |
| **策略确定** | GT深度监督 + 两阶段训练 |

### 当前状态

**✅ 核心功能完成**:
- NormalHead 实现并集成
- 法线一致性约束
- 两阶段训练策略
- 训练流程测试通过

**🔄 进行中**:
- 数据集集成
- GT 深度渲染预处理

### 下一步

1. 实现深度渲染预处理脚本
2. 实现 OpenMaterialDataset
3. 添加 L_depth 监督损失
4. 端到端训练测试

---

**报告完成时间**: 2025-12-14
**审查人**: Claude Code (AI Assistant)
**状态**: Phase 4 核心功能完成，数据集集成进行中

---

**END OF PROGRESS REPORT**
