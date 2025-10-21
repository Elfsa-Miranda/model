#!/usr/bin/env python3
"""
快速测试模型创建是否正常
"""

import torch
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_creation():
    """测试模型创建"""
    try:
        print("正在测试模型创建...")
        
        # 测试TeacherModel
        from models import TeacherModel
        teacher_model = TeacherModel(
            num_joints=17,
            coord_dim=2,
            embed_dim=384,  # 使用较小的模型进行测试
            depth=6,
            num_heads=6
        )
        print("✅ TeacherModel 创建成功")
        
        # 测试StudentModel
        from models import StudentModel
        student_model = StudentModel(
            num_patches=10,
            patch_dim=64,
            embed_dim=384,  # 使用较小的模型进行测试
            depth=6,
            num_heads=6,
            num_joints=17,
            coord_dim=2
        )
        print("✅ StudentModel 创建成功")
        
        # 测试前向传播
        batch_size = 2
        
        # TeacherModel前向传播
        rgb_skeleton = torch.randn(batch_size, 17, 2)
        teacher_loss, teacher_pred, teacher_mask = teacher_model(rgb_skeleton)
        print(f"✅ TeacherModel 前向传播成功: loss={teacher_loss.item():.4f}")
        
        # StudentModel前向传播
        csi_patches = torch.randn(batch_size, 10, 64)
        student_outputs = student_model(csi_patches)
        print(f"✅ StudentModel 前向传播成功: 输出形状={student_outputs['reconstructed_patches'].shape}")
        
        print("\n🎉 所有模型测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_creation()
    if success:
        print("\n✅ 可以继续运行 test_mmfi_integration.py")
    else:
        print("\n❌ 请先解决模型问题")

