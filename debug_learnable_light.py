#!/usr/bin/env python3
"""
Debug script for Learnable Lighting in PBR pipeline

这个脚本验证可学习光照的完整pipeline:
1. VGGT预训练权重（depth/camera）
2. MaterialHead随机初始化
3. LightHead随机初始化，从图像特征预测光照方向
4. PhongRenderer使用预测的光照方向渲染
5. 验证梯度能反向传播到LightHead

关键验证点:
- ✓ LightHead预测光照方向（归一化向量）
- ✓ 光照方向影响渲染结果
- ✓ 梯度能传到LightHead和MaterialHead
- ✓ 光照方向初始化合理（非[0,0,1]头灯）

运行方式:
    python debug_learnable_light.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

print("=" * 80)
print("Learnable Lighting Test - Breaking the Co-located Light Assumption")
print("=" * 80)

# ===== 1. 生成测试图像 =====
print("\n[Stage 1/8] Generating test image...")

def generate_test_image(size=(518, 518)):
    """生成带有几何结构的测试图像"""
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # 红色球体
    y, x = np.ogrid[:H, :W]
    center = (H // 2, W // 2)
    radius = min(H, W) // 3

    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    sphere_mask = dist_from_center <= radius

    img[sphere_mask] = [200, 50, 50]
    highlight_mask = sphere_mask & (x < center[1]) & (y < center[0])
    img[highlight_mask] = [255, 100, 100]
    img[~sphere_mask] = [230, 230, 230]

    return img

test_img = generate_test_image()
print(f"  ✓ Generated test image: {test_img.shape}")

images = torch.from_numpy(test_img).float() / 255.0
images = images.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
images = images.cuda()

# ===== 2. 初始化VGGT（启用MaterialHead + LightHead）=====
print("\n[Stage 2/8] Initializing VGGT with MaterialHead + LightHead...")

from vggt.models.vggt import VGGT

model = VGGT(
    enable_camera=True,
    enable_depth=True,
    enable_point=False,
    enable_track=False,
    enable_material=True,  # 材质预测
    enable_light=True,     # 光照预测 ⭐ 核心
).cuda()

print(f"  ✓ Model initialized")
print(f"     MaterialHead parameters: {sum(p.numel() for p in model.material_head.parameters()) / 1e6:.2f}M")
print(f"     LightHead parameters: {sum(p.numel() for p in model.light_head.parameters()) / 1e6:.2f}M")

# ===== 3. 加载预训练权重 =====
print("\n[Stage 3/8] Loading pretrained weights (depth/camera only)...")

try:
    from vggt.models.vggt import VGGT as VGGT_Pretrained
    pretrained_model = VGGT_Pretrained.from_pretrained("facebook/VGGT-1B")
    pretrained_dict = pretrained_model.state_dict()

    model_dict = model.state_dict()

    # 过滤：只加载depth/camera，不加载material_head和light_head
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and 'material_head' not in k and 'light_head' not in k
    }

    model_dict.update(pretrained_dict_filtered)
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

    print(f"  ✓ Loaded pretrained weights from Hugging Face")
    print(f"     Loaded keys: {len(pretrained_dict_filtered)}")
    print(f"     Missing keys (MaterialHead + LightHead): {len([k for k in missing_keys if 'material_head' in k or 'light_head' in k])}")

except Exception as e:
    print(f"  ⚠️  Could not load pretrained weights: {e}")
    print(f"     Continuing with random initialization...")

model.eval()

# ===== 4. 初始化渲染器（支持可学习光照）=====
print("\n[Stage 4/8] Initializing PhongRenderer with learnable lighting...")

from training.rendering.phong_renderer import SimplePhongRenderer

renderer = SimplePhongRenderer(
    ambient_strength=0.3,
    use_learnable_light=True,  # ⭐ 关键：使用可学习光照
).cuda()

print(f"  ✓ Renderer initialized")
print(f"     use_learnable_light=True ✓")
print(f"     Default light direction: {renderer.default_light_dir.cpu().numpy()}")

# ===== 5. 前向传播：预测depth + materials + lighting =====
print("\n[Stage 5/8] Forward pass: predicting depth, materials, and lighting...")

with torch.no_grad():
    predictions = model(images)

    # 提取预测结果
    depth_pred = predictions['depth']  # (B, S, H, W, 1)
    materials_pred = {
        'diffuse': predictions['diffuse'],  # (B, S, 3, H, W)
        'specular': predictions['specular'],
        'roughness': predictions['roughness'],
        'ambient_occlusion': predictions['ambient_occlusion'],
    }
    light_params_pred = {
        'light_direction': predictions['light_direction'],  # (B, S, 3) ⭐
        'light_intensity': predictions['light_intensity'],  # (B, S, 1)
        'light_color': predictions['light_color'],          # (B, S, 3)
    }

print(f"  ✓ Forward pass successful")
print(f"     Depth shape: {depth_pred.shape}")
print(f"     Diffuse shape: {materials_pred['diffuse'].shape}")
print(f"     Light direction shape: {light_params_pred['light_direction'].shape}")

# 打印预测的光照参数
light_dir = light_params_pred['light_direction'][0, 0].cpu().numpy()
light_intensity = light_params_pred['light_intensity'][0, 0].cpu().numpy()
light_color = light_params_pred['light_color'][0, 0].cpu().numpy()

print(f"\n  📊 Predicted lighting parameters:")
print(f"     Direction: [{light_dir[0]:.3f}, {light_dir[1]:.3f}, {light_dir[2]:.3f}]")
print(f"     Norm: {np.linalg.norm(light_dir):.6f} (should be 1.0)")
print(f"     Intensity: {light_intensity[0]:.3f}")
print(f"     Color: [{light_color[0]:.3f}, {light_color[1]:.3f}, {light_color[2]:.3f}]")

# 验证是否不是头灯
is_headlight = np.allclose(light_dir, [0, 0, 1], atol=0.1)
if is_headlight:
    print(f"  ⚠️  WARNING: Light direction close to [0,0,1] (headlight)!")
else:
    print(f"  ✓ Light direction is NOT a headlight (good!)")

# ===== 6. 渲染 =====
print("\n[Stage 6/8] Rendering with predicted lighting...")

with torch.no_grad():
    # 转换材质格式: (B, S, C, H, W) -> (B, S, H, W, C)
    B, S, C, H, W = materials_pred['diffuse'].shape
    materials_for_render = {
        'diffuse': materials_pred['diffuse'].permute(0, 1, 3, 4, 2),
        'specular': materials_pred['specular'].permute(0, 1, 3, 4, 2),
        'roughness': materials_pred['roughness'].permute(0, 1, 3, 4, 2),
        'ambient_occlusion': materials_pred['ambient_occlusion'].permute(0, 1, 3, 4, 2),
    }

    # 渲染（传入预测的光照参数）⭐
    rendered, normals = renderer(
        depth=depth_pred.squeeze(-1),  # (B, S, H, W)
        materials=materials_for_render,
        light_params=light_params_pred,  # ⭐ 使用预测的光照
    )

print(f"  ✓ Rendering successful")
print(f"     Rendered shape: {rendered.shape}")
print(f"     Rendered range: [{rendered.min():.3f}, {rendered.max():.3f}]")

# ===== 7. 可视化 =====
print("\n[Stage 7/8] Visualizing results...")

fig = plt.figure(figsize=(20, 12))

# 第一行: 输入和深度
ax1 = fig.add_subplot(3, 4, 1)
ax1.imshow(test_img)
ax1.set_title("1. Input Image\n(Red sphere)")
ax1.axis('off')

ax2 = fig.add_subplot(3, 4, 2)
depth_vis = depth_pred[0, 0, :, :, 0].cpu().numpy()
ax2.imshow(depth_vis, cmap='viridis')
ax2.set_title(f"2. Predicted Depth\n[{depth_vis.min():.3f}, {depth_vis.max():.3f}]")
ax2.axis('off')

ax3 = fig.add_subplot(3, 4, 3)
normals_vis = (normals[0, 0].cpu().numpy() + 1) / 2  # [-1,1] -> [0,1]
ax3.imshow(normals_vis)
ax3.set_title("3. Computed Normals\n(from depth)")
ax3.axis('off')

ax4 = fig.add_subplot(3, 4, 4, projection='3d')
# 3D可视化光照方向
ax4.quiver(0, 0, 0, light_dir[0], light_dir[1], light_dir[2],
           color='yellow', arrow_length_ratio=0.2, linewidth=3, label='Predicted Light')
ax4.quiver(0, 0, 0, 0, 0, 1,
           color='red', arrow_length_ratio=0.2, linewidth=2, alpha=0.5, label='Headlight (0,0,1)')
ax4.set_xlim([-1, 1])
ax4.set_ylim([-1, 1])
ax4.set_zlim([0, 1])
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title("4. Predicted Light Direction\n(Yellow vs Red headlight)")
ax4.legend()

# 第二行: 材质
ax5 = fig.add_subplot(3, 4, 5)
diffuse_vis = materials_pred['diffuse'][0, 0].permute(1, 2, 0).cpu().numpy()
ax5.imshow(diffuse_vis)
ax5.set_title(f"5. Diffuse Albedo\n[{diffuse_vis.min():.3f}, {diffuse_vis.max():.3f}]")
ax5.axis('off')

ax6 = fig.add_subplot(3, 4, 6)
specular_vis = materials_pred['specular'][0, 0].permute(1, 2, 0).cpu().numpy()
ax6.imshow(specular_vis)
ax6.set_title(f"6. Specular Color\n[{specular_vis.min():.3f}, {specular_vis.max():.3f}]")
ax6.axis('off')

ax7 = fig.add_subplot(3, 4, 7)
roughness_vis = materials_pred['roughness'][0, 0, 0].cpu().numpy()
ax7.imshow(roughness_vis, cmap='gray')
ax7.set_title(f"7. Roughness\n[{roughness_vis.min():.3f}, {roughness_vis.max():.3f}]")
ax7.axis('off')

ax8 = fig.add_subplot(3, 4, 8)
ao_vis = materials_pred['ambient_occlusion'][0, 0, 0].cpu().numpy()
ax8.imshow(ao_vis, cmap='gray')
ax8.set_title(f"8. Ambient Occlusion\n[{ao_vis.min():.3f}, {ao_vis.max():.3f}]")
ax8.axis('off')

# 第三行: 渲染结果和对比
ax9 = fig.add_subplot(3, 4, 9)
rendered_vis = rendered[0, 0].cpu().numpy()
ax9.imshow(rendered_vis)
ax9.set_title(f"9. PBR Rendered (Learned Light)\n[{rendered_vis.min():.3f}, {rendered_vis.max():.3f}]")
ax9.axis('off')

# 对比：使用默认光照渲染
with torch.no_grad():
    rendered_default, _ = renderer(
        depth=depth_pred.squeeze(-1),
        materials=materials_for_render,
        light_params=None,  # 使用默认光照
    )

ax10 = fig.add_subplot(3, 4, 10)
rendered_default_vis = rendered_default[0, 0].cpu().numpy()
ax10.imshow(rendered_default_vis)
ax10.set_title(f"10. PBR Rendered (Default Light)\n[{rendered_default_vis.min():.3f}, {rendered_default_vis.max():.3f}]")
ax10.axis('off')

# 差异图
ax11 = fig.add_subplot(3, 4, 11)
diff = np.abs(rendered_vis - rendered_default_vis)
ax11.imshow(diff, cmap='hot')
ax11.set_title(f"11. Difference Map\nMax diff: {diff.max():.3f}")
ax11.axis('off')

# 光照参数文本
ax12 = fig.add_subplot(3, 4, 12)
ax12.axis('off')
info_text = f"""
Predicted Lighting Parameters:

Direction: [{light_dir[0]:.3f}, {light_dir[1]:.3f}, {light_dir[2]:.3f}]
Norm: {np.linalg.norm(light_dir):.6f}

Intensity: {light_intensity[0]:.3f}
Color: RGB({light_color[0]:.3f}, {light_color[1]:.3f}, {light_color[2]:.3f})

Default Light Direction:
[{renderer.default_light_dir[0]:.3f}, {renderer.default_light_dir[1]:.3f}, {renderer.default_light_dir[2]:.3f}]

Status:
{'✓ NOT a headlight' if not is_headlight else '⚠️  Close to headlight'}
"""
ax12.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
          verticalalignment='center')
ax12.set_title("12. Lighting Info")

plt.tight_layout()
output_path = "debug_learnable_light_output.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Visualization saved to: {output_path}")

# ===== 8. 梯度测试 =====
print("\n[Stage 8/8] Gradient backpropagation test...")

model.train()

# 前向传播（需要梯度）
predictions_train = model(images)

materials_train = {
    'diffuse': predictions_train['diffuse'].permute(0, 1, 3, 4, 2),
    'specular': predictions_train['specular'].permute(0, 1, 3, 4, 2),
    'roughness': predictions_train['roughness'].permute(0, 1, 3, 4, 2),
    'ambient_occlusion': predictions_train['ambient_occlusion'].permute(0, 1, 3, 4, 2),
}
light_params_train = {
    'light_direction': predictions_train['light_direction'],
    'light_intensity': predictions_train['light_intensity'],
    'light_color': predictions_train['light_color'],
}

rendered_train, _ = renderer(
    depth=predictions_train['depth'].squeeze(-1),
    materials=materials_train,
    light_params=light_params_train,
)

# 计算损失（使用输入图像作为target）
from training.rendering.pbr_loss import PBRLoss
loss_fn = PBRLoss(photometric_loss_type='l1').cuda()

# 使用输入图像作为target（让网络学习重建输入）
target_img = images[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 3)
loss = loss_fn(rendered_train, target_img)

print(f"  Loss: {loss.item():.6f}")

# 反向传播
loss.backward()

# 检查LightHead梯度
print(f"\n  Checking LightHead gradients:")
light_grad_count = 0
max_light_grad = 0.0
for name, param in model.light_head.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        light_grad_count += 1
        grad_norm = param.grad.norm().item()
        max_light_grad = max(max_light_grad, grad_norm)
        if light_grad_count <= 3:  # 打印前3个
            print(f"     ✓ {name}: grad_norm={grad_norm:.6f}")

print(f"  ✓ LightHead: {light_grad_count} parameters with gradients")
print(f"     Max gradient norm: {max_light_grad:.6f}")

# 检查MaterialHead梯度
print(f"\n  Checking MaterialHead gradients:")
material_grad_count = 0
max_material_grad = 0.0
for name, param in model.material_head.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        material_grad_count += 1
        grad_norm = param.grad.norm().item()
        max_material_grad = max(max_material_grad, grad_norm)
        if material_grad_count <= 3:
            print(f"     ✓ {name}: grad_norm={grad_norm:.6f}")

print(f"  ✓ MaterialHead: {material_grad_count} parameters with gradients")
print(f"     Max gradient norm: {max_material_grad:.6f}")

# ===== 总结 =====
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

checks = [
    ("Depth prediction quality", depth_vis.max() - depth_vis.min() > 0.1),
    ("Light direction normalized", abs(np.linalg.norm(light_dir) - 1.0) < 1e-3),
    ("Light direction NOT headlight", not is_headlight),
    ("LightHead gradients flow", light_grad_count > 0),
    ("MaterialHead gradients flow", material_grad_count > 0),
    ("Rendering difference visible", diff.max() > 0.01),
]

all_passed = True
for check_name, check_result in checks:
    status = "✓" if check_result else "✗"
    print(f"  {status} {check_name}")
    if not check_result:
        all_passed = False

if all_passed:
    print("\n🎉 All checks passed! Learnable lighting is working correctly.")
    print(f"   Key achievement: Light direction [{light_dir[0]:.3f}, {light_dir[1]:.3f}, {light_dir[2]:.3f}]")
    print(f"                    is NOT the co-located headlight [0, 0, 1]")
else:
    print("\n⚠️  Some checks failed. Please review the output above.")

print("=" * 80)
