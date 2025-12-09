#!/usr/bin/env python3
"""
Debug script for PBR rendering pipeline - DRY RUN TEST

这个脚本的目的是验证:
1. MaterialHead能否正确输出材质
2. Phong渲染器能否正常工作
3. 所有tensor维度是否匹配
4. 输出的图像是否合理

运行前确保:
- CUDA可用（或修改为CPU测试）
- 安装了matplotlib用于可视化

运行方式:
    python debug_pbr.py

成功标志:
- 无报错
- 生成 debug_pbr_output.png
- Diffuse/Specular应该是彩色噪点（模型未训练）
- Normals应该是紫蓝色调
- Final Render不应该全黑或全白
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("PBR Rendering Pipeline - Dry Run Test")
print("=" * 80)

# ===== 阶段 1: 导入模块 =====
print("\n[1/6] Importing modules...")
try:
    from vggt.heads.material_head import MaterialHead
    from training.rendering.phong_renderer import SimplePhongRenderer
    from training.rendering.pbr_loss import PBRLoss
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# ===== 阶段 2: 准备伪造数据 =====
print("\n[2/6] Preparing fake data...")

# 检查CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Using device: {device}")

# 超参数
B, S, H, W = 1, 2, 518, 518  # Batch=1, Sequence=2, VGGT标准分辨率
embed_dim = 2048  # VGGT aggregator输出维度 (2*1024)
patch_size = 14
patch_h, patch_w = H // patch_size, W // patch_size
num_patches = patch_h * patch_w

print(f"  Config: B={B}, S={S}, H={H}, W={W}")
print(f"  Patch grid: {patch_h} x {patch_w} = {num_patches} patches")

# 伪造aggregated_tokens_list (模拟aggregator输出)
# 需要4个层级的tokens，对应intermediate_layer_idx=[4,11,17,23]
print("  Creating fake aggregated_tokens_list...")
aggregated_tokens_list = []
num_layers = 24  # VGGT有24层
patch_start_idx = 2  # 前2个token是camera tokens

for layer_idx in range(num_layers):
    # 每层的token: [B, S, num_tokens, embed_dim]
    # num_tokens = camera_tokens + patch_tokens
    num_tokens = patch_start_idx + num_patches
    tokens = torch.randn(B, S, num_tokens, embed_dim, device=device)
    aggregated_tokens_list.append(tokens)

# 伪造图像
fake_images = torch.rand(B, S, 3, H, W, device=device)  # 范围[0,1]

# 伪造深度图
fake_depth = torch.rand(B, S, H, W, device=device) * 10.0 + 1.0  # 范围[1, 11]

# 伪造相机内参 (简化版)
fake_intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
fake_intrinsics = fake_intrinsics.expand(B, S, -1, -1)

print("✓ Fake data prepared")

# ===== 阶段 3: 测试 MaterialHead =====
print("\n[3/6] Testing MaterialHead...")

try:
    material_head = MaterialHead(
        dim_in=embed_dim,
        patch_size=patch_size,
    ).to(device)

    print(f"  MaterialHead parameters: {sum(p.numel() for p in material_head.parameters()) / 1e6:.2f}M")

    # 前向传播
    with torch.no_grad():
        materials = material_head(
            aggregated_tokens_list=aggregated_tokens_list,
            images=fake_images,
            patch_start_idx=patch_start_idx,
        )

    # 检查输出
    print(f"  Output shapes:")
    for key, value in materials.items():
        print(f"    {key}: {list(value.shape)} | range: [{value.min():.3f}, {value.max():.3f}]")

    # 验证输出范围（应该在[0,1]之间，因为有Sigmoid）
    for key, value in materials.items():
        assert value.min() >= 0 and value.max() <= 1, f"{key} out of range [0,1]"

    print("✓ MaterialHead test passed")

except Exception as e:
    print(f"✗ MaterialHead test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== 阶段 4: 测试 Phong Renderer =====
print("\n[4/6] Testing PhongRenderer...")

try:
    renderer = SimplePhongRenderer().to(device)

    # 转换材质格式: (B, S, 3, H, W) -> (B, S, H, W, 3)
    materials_for_render = {
        'diffuse': materials['diffuse'].permute(0, 1, 3, 4, 2),
        'specular': materials['specular'].permute(0, 1, 3, 4, 2),
        'roughness': materials['roughness'].permute(0, 1, 3, 4, 2),
        'ambient_occlusion': materials['ambient_occlusion'].permute(0, 1, 3, 4, 2),
    }

    # 渲染
    with torch.no_grad():
        rendered_img, normals = renderer(
            depth=fake_depth,
            materials=materials_for_render,
            intrinsics=fake_intrinsics,
        )

    print(f"  Rendered image shape: {list(rendered_img.shape)}")
    print(f"  Rendered image range: [{rendered_img.min():.3f}, {rendered_img.max():.3f}]")
    print(f"  Normals shape: {list(normals.shape)}")
    print(f"  Normals range: [{normals.min():.3f}, {normals.max():.3f}]")

    # 验证输出
    assert rendered_img.shape == (B, S, H, W, 3), "Rendered image shape mismatch"
    assert normals.shape == (B, S, H, W, 3), "Normals shape mismatch"

    print("✓ PhongRenderer test passed")

except Exception as e:
    print(f"✗ PhongRenderer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== 阶段 5: 测试 PBR Loss =====
print("\n[5/6] Testing PBRLoss...")

try:
    pbr_loss = PBRLoss(weight=0.1).to(device)

    # 伪造目标图像
    target_img = torch.rand(B, S, H, W, 3, device=device)

    # 计算损失
    with torch.no_grad():
        loss = pbr_loss(rendered_img, target_img)

    print(f"  PBR Loss value: {loss.item():.6f}")
    print(f"  Loss requires_grad: {loss.requires_grad}")

    # 验证损失是标量
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print("✓ PBRLoss test passed")

except Exception as e:
    print(f"✗ PBRLoss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== 阶段 6: 可视化输出 =====
print("\n[6/6] Visualizing outputs...")

try:
    # 取第一个batch的第一个sequence
    b_idx, s_idx = 0, 0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('PBR Rendering Pipeline - Dry Run Output', fontsize=16)

    # 第一行：输入和材质
    # 1. 输入图像（伪造的）
    axes[0, 0].imshow(fake_images[b_idx, s_idx].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Fake Input Image')
    axes[0, 0].axis('off')

    # 2. Diffuse (Albedo)
    diffuse_img = materials['diffuse'][b_idx, s_idx].permute(1, 2, 0).cpu().numpy()
    axes[0, 1].imshow(diffuse_img)
    axes[0, 1].set_title(f'Predicted Diffuse\nrange:[{diffuse_img.min():.2f},{diffuse_img.max():.2f}]')
    axes[0, 1].axis('off')

    # 3. Specular
    specular_img = materials['specular'][b_idx, s_idx].permute(1, 2, 0).cpu().numpy()
    axes[0, 2].imshow(specular_img)
    axes[0, 2].set_title(f'Predicted Specular\nrange:[{specular_img.min():.2f},{specular_img.max():.2f}]')
    axes[0, 2].axis('off')

    # 4. Roughness
    roughness_img = materials['roughness'][b_idx, s_idx, 0].cpu().numpy()
    im = axes[0, 3].imshow(roughness_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title(f'Predicted Roughness\nmean:{roughness_img.mean():.2f}')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # 第二行：渲染结果
    # 5. Depth (伪造的)
    depth_img = fake_depth[b_idx, s_idx].cpu().numpy()
    im = axes[1, 0].imshow(depth_img, cmap='viridis')
    axes[1, 0].set_title(f'Fake Depth\nrange:[{depth_img.min():.1f},{depth_img.max():.1f}]')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 6. Normals (归一化到[0,1]用于显示)
    normals_img = normals[b_idx, s_idx].cpu().numpy()
    normals_vis = (normals_img + 1.0) / 2.0  # [-1,1] -> [0,1]
    axes[1, 1].imshow(normals_vis)
    axes[1, 1].set_title('Computed Normals\n(should be purplish)')
    axes[1, 1].axis('off')

    # 7. Ambient Occlusion
    ao_img = materials['ambient_occlusion'][b_idx, s_idx, 0].cpu().numpy()
    im = axes[1, 2].imshow(ao_img, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Predicted AO\nmean:{ao_img.mean():.2f}')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    # 8. Final Rendered Image
    rendered_vis = rendered_img[b_idx, s_idx].cpu().numpy()
    axes[1, 3].imshow(rendered_vis)
    axes[1, 3].set_title(f'Final Rendered\nrange:[{rendered_vis.min():.2f},{rendered_vis.max():.2f}]')
    axes[1, 3].axis('off')

    plt.tight_layout()

    output_path = 'debug_pbr_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")

    # 打印检查提示
    print("\n" + "=" * 80)
    print("DRY RUN COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 80)
    print("\n请检查生成的图像 debug_pbr_output.png:")
    print("  ✓ Diffuse: 应该是彩色的噪点（模型未训练）")
    print("  ✓ Specular: 同样是彩色噪点")
    print("  ✓ Roughness: 应该是灰度图，分布在[0,1]")
    print("  ✓ Normals: 应该是紫蓝色调（法向量[0,0,1]映射为紫色）")
    print("  ✓ Rendered: 不应该全黑或全白，应该有一定的亮度变化")
    print("\n如果以上检查都通过，说明基础设施搭建成功！")
    print("下一步: 集成到VGGT模型中")

except Exception as e:
    print(f"✗ Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
