#!/usr/bin/env python3
"""
Debug script for PBR pipeline with PRETRAINED VGGT weights

这个脚本的目的:
1. 加载预训练的VGGT模型（depth/camera有权重）
2. MaterialHead保持随机初始化
3. 用真实图像测试，验证几何质量
4. 测试梯度反向传播

关键验证点:
- ✓ 深度预测应该清晰合理（来自预训练权重）
- ✓ 法线计算应该正确（来自深度）
- ✓ 材质预测是随机的（MaterialHead未训练）
- ✓ 渲染流程完整（Phong着色）
- ✓ 梯度能传到MaterialHead（可训练性）

运行方式:
    python debug_pbr_real.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("PBR Pipeline with Pretrained VGGT - Real Image Test")
print("=" * 80)

# ===== 1. 生成测试图像 =====
print("\n[1/7] Generating test image...")

def generate_test_image(size=(518, 518)):
    """
    生成一个带有几何结构的测试图像
    这样预训练的depth head能预测出合理的深度
    """
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # 创建一个红色的球体（中心深度大，边缘深度小）
    y, x = np.ogrid[:H, :W]
    center = (H // 2, W // 2)
    radius = min(H, W) // 3

    # 球体mask
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    sphere_mask = dist_from_center <= radius

    # 球体着色（带一点渐变，模拟光照）
    img[sphere_mask] = [200, 50, 50]  # 红色基调

    # 添加高光区域（左上角）
    highlight_mask = sphere_mask & (x < center[1]) & (y < center[0])
    img[highlight_mask] = [255, 100, 100]

    # 添加背景（浅灰色）
    img[~sphere_mask] = [230, 230, 230]

    return img

test_img = generate_test_image()
print(f"  Generated test image: {test_img.shape}, range: [{test_img.min()}, {test_img.max()}]")

# 转换为tensor
images = torch.from_numpy(test_img).float() / 255.0  # [0, 1]
images = images.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
images = images.cuda()

print(f"  Tensor shape: {images.shape}")

# ===== 2. 初始化模型（启用MaterialHead）=====
print("\n[2/7] Initializing VGGT with MaterialHead...")

from vggt.models.vggt import VGGT

model = VGGT(
    enable_camera=True,
    enable_depth=True,
    enable_point=False,
    enable_track=False,
    enable_material=True,  # 关键：启用材质预测
).cuda()

print(f"  Model initialized")
print(f"  MaterialHead parameters: {sum(p.numel() for p in model.material_head.parameters()) / 1e6:.2f}M")

# ===== 3. 加载预训练权重 =====
print("\n[3/7] Loading pretrained weights...")

try:
    # 方法1: 从Hugging Face加载
    print("  Attempting to load from Hugging Face (facebook/VGGT-1B)...")
    from vggt.models.vggt import VGGT as VGGT_Pretrained
    pretrained_model = VGGT_Pretrained.from_pretrained("facebook/VGGT-1B")

    # 提取预训练权重
    pretrained_dict = pretrained_model.state_dict()

    # 当前模型的state_dict
    model_dict = model.state_dict()

    # 过滤掉MaterialHead的keys（因为预训练模型没有）
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and 'material_head' not in k
    }

    # 更新权重
    model_dict.update(pretrained_dict_filtered)
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

    print(f"  ✅ Loaded pretrained weights from Hugging Face")
    print(f"  Missing keys: {len(missing_keys)} (Expected: MaterialHead parameters)")
    print(f"  Unexpected keys: {len(unexpected_keys)}")

    # 验证MaterialHead的keys在missing中
    material_keys = [k for k in missing_keys if 'material_head' in k]
    print(f"  MaterialHead keys (randomly initialized): {len(material_keys)}")

except Exception as e:
    print(f"  ⚠️  Could not load pretrained weights: {e}")
    print(f"  ⚠️  Continuing with random initialization for all modules")
    print(f"  ⚠️  Depth predictions will be poor (random), but pipeline will still work")

# ===== 4. 初始化渲染器 =====
print("\n[4/7] Initializing Phong Renderer...")

from training.rendering.phong_renderer import SimplePhongRenderer

renderer = SimplePhongRenderer().cuda()
print("  ✅ Renderer initialized")

# ===== 5. 前向传播 =====
print("\n[5/7] Running forward pass...")

model.eval()  # 评估模式

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        # VGGT前向传播
        predictions = model(images)

        print("  Predictions keys:", list(predictions.keys()))

        # 提取深度
        depth = predictions['depth']  # (B, S, H, W, 1)
        print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")

        # 提取材质
        materials_raw = {
            'diffuse': predictions['diffuse'],      # (B, S, 3, H, W)
            'specular': predictions['specular'],
            'roughness': predictions['roughness'],  # (B, S, 1, H, W)
            'ambient_occlusion': predictions['ambient_occlusion'],
        }

        print(f"  Diffuse range: [{materials_raw['diffuse'].min():.3f}, {materials_raw['diffuse'].max():.3f}]")
        print(f"  Roughness range: [{materials_raw['roughness'].min():.3f}, {materials_raw['roughness'].max():.3f}]")

        # 转换格式: (B,S,C,H,W) → (B,S,H,W,C) for renderer
        materials_for_render = {
            'diffuse': materials_raw['diffuse'].permute(0, 1, 3, 4, 2),
            'specular': materials_raw['specular'].permute(0, 1, 3, 4, 2),
            'roughness': materials_raw['roughness'].permute(0, 1, 3, 4, 2),
            'ambient_occlusion': materials_raw['ambient_occlusion'].permute(0, 1, 3, 4, 2),
        }

        # 渲染
        depth_for_render = depth.squeeze(-1)  # (B, S, H, W, 1) → (B, S, H, W)
        rendered_img, normals = renderer(
            depth=depth_for_render,
            materials=materials_for_render,
            intrinsics=None,  # 使用默认相机
        )

        print(f"  Rendered image shape: {rendered_img.shape}")
        print(f"  Rendered range: [{rendered_img.min():.3f}, {rendered_img.max():.3f}]")
        print(f"  Normals range: [{normals.min():.3f}, {normals.max():.3f}]")

print("  ✅ Forward pass successful")

# ===== 6. 可视化 =====
print("\n[6/7] Generating visualization...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('PBR Pipeline with Pretrained VGGT Weights', fontsize=16, fontweight='bold')

b_idx, s_idx = 0, 0

# 第一行：输入和几何
# 1. 输入图像
axes[0, 0].imshow(images[b_idx, s_idx].permute(1, 2, 0).cpu().numpy())
axes[0, 0].set_title('Input Test Image\n(Red sphere on gray bg)', fontsize=12)
axes[0, 0].axis('off')

# 2. 深度图（归一化显示）
depth_np = depth[b_idx, s_idx, :, :, 0].cpu().numpy()
d_min, d_max = depth_np.min(), depth_np.max()
depth_normalized = (depth_np - d_min) / (d_max - d_min + 1e-6)
im1 = axes[0, 1].imshow(depth_normalized, cmap='plasma')
axes[0, 1].set_title(f'Predicted Depth\n(Pretrained)\nrange:[{d_min:.2f},{d_max:.2f}]', fontsize=12)
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# 3. 法线（归一化到[0,1]显示）
normals_np = normals[b_idx, s_idx].cpu().numpy()
normals_vis = (normals_np + 1.0) / 2.0
axes[0, 2].imshow(normals_vis)
axes[0, 2].set_title('Computed Normals\n(From depth gradient)\nShould show sphere curvature', fontsize=12)
axes[0, 2].axis('off')

# 4. 深度的3D可视化（轮廓图）
axes[0, 3].contourf(depth_normalized, levels=10, cmap='viridis')
axes[0, 3].set_title('Depth Contour\n(Should show sphere shape)', fontsize=12)
axes[0, 3].axis('off')

# 第二行：材质和渲染
# 5. Diffuse
diffuse_np = materials_for_render['diffuse'][b_idx, s_idx].cpu().numpy()
axes[1, 0].imshow(diffuse_np)
axes[1, 0].set_title(f'Predicted Diffuse\n(Random init)\nmean:{diffuse_np.mean():.3f}', fontsize=12)
axes[1, 0].axis('off')

# 6. Specular
specular_np = materials_for_render['specular'][b_idx, s_idx].cpu().numpy()
axes[1, 1].imshow(specular_np)
axes[1, 1].set_title(f'Predicted Specular\n(Random init)\nmean:{specular_np.mean():.3f}', fontsize=12)
axes[1, 1].axis('off')

# 7. Roughness
roughness_np = materials_for_render['roughness'][b_idx, s_idx, :, :, 0].cpu().numpy()
im2 = axes[1, 2].imshow(roughness_np, cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title(f'Predicted Roughness\n(Random init)\nmean:{roughness_np.mean():.3f}', fontsize=12)
axes[1, 2].axis('off')
plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

# 8. 最终渲染
rendered_np = rendered_img[b_idx, s_idx].cpu().numpy()
axes[1, 3].imshow(rendered_np)
axes[1, 3].set_title(f'PBR Rendered\n(Depth×Material)\nrange:[{rendered_np.min():.2f},{rendered_np.max():.2f}]', fontsize=12)
axes[1, 3].axis('off')

plt.tight_layout()
output_path = 'debug_pbr_real.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✅ Visualization saved to: {output_path}")

# ===== 7. 梯度测试 =====
print("\n[7/7] Testing gradient backpropagation...")

model.train()  # 切换到训练模式

# 准备损失函数
from training.rendering.pbr_loss import PBRLoss

loss_fn = PBRLoss(weight=1.0, photometric_loss_type='l1').cuda()

# 转换目标图像格式: (B,S,C,H,W) → (B,S,H,W,C)
target_img = images.permute(0, 1, 3, 4, 2)

# 重新前向传播（需要梯度）
with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    predictions_train = model(images)

    depth_train = predictions_train['depth'].squeeze(-1)
    materials_train = {
        'diffuse': predictions_train['diffuse'].permute(0, 1, 3, 4, 2),
        'specular': predictions_train['specular'].permute(0, 1, 3, 4, 2),
        'roughness': predictions_train['roughness'].permute(0, 1, 3, 4, 2),
        'ambient_occlusion': predictions_train['ambient_occlusion'].permute(0, 1, 3, 4, 2),
    }

    rendered_train, _ = renderer(depth_train, materials_train)

    # 计算损失
    loss = loss_fn(rendered_train, target_img)

print(f"  Photometric loss: {loss.item():.6f}")

# 反向传播
print("  Running backward pass...")
loss.backward()

# 检查MaterialHead的梯度
material_grad_found = False
max_grad_norm = 0.0
grad_params_count = 0

for name, param in model.material_head.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        max_grad_norm = max(max_grad_norm, grad_norm)
        grad_params_count += 1

        if not material_grad_found:
            print(f"  ✅ Gradient detected in MaterialHead: {name}")
            print(f"     Gradient norm: {grad_norm:.6f}")
            material_grad_found = True

if material_grad_found:
    print(f"  ✅ SUCCESS: Gradients flow to MaterialHead!")
    print(f"  Total parameters with gradients: {grad_params_count}")
    print(f"  Max gradient norm: {max_grad_norm:.6f}")
else:
    print(f"  ❌ ERROR: No gradients detected in MaterialHead!")
    print(f"  This means the pipeline is broken for training.")

# ===== 总结 =====
print("\n" + "=" * 80)
print("TEST COMPLETED!")
print("=" * 80)

print("\n📊 Results Summary:")
print(f"  ✓ Forward pass: SUCCESS")
print(f"  ✓ Depth prediction: range [{d_min:.3f}, {d_max:.3f}]")
print(f"  ✓ Material prediction: Diffuse mean {diffuse_np.mean():.3f}")
print(f"  ✓ Rendering: range [{rendered_np.min():.3f}, {rendered_np.max():.3f}]")
print(f"  ✓ Gradient test: {'PASSED ✅' if material_grad_found else 'FAILED ❌'}")

print("\n🔍 Visual Inspection Checklist:")
print("  Open debug_pbr_real.png and check:")
print("  1. Depth map shows a clear sphere shape (lighter in center, darker at edges)")
print("  2. Normals show purple/blue tones with curvature")
print("  3. Diffuse/Specular are random-ish (MaterialHead not trained)")
print("  4. Final render has some shading variations (Phong working)")

print("\n🚀 Next Steps:")
if material_grad_found:
    print("  ✅ Pipeline is ready for training!")
    print("  ✅ Proceed to Phase 4: Integrate into training loop")
else:
    print("  ❌ Fix gradient flow issues before training")
    print("  Check: amp settings, frozen modules, loss computation")

print("\n" + "=" * 80)
