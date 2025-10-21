"""
Enhanced Multi-Modal DMAE 组件测试脚本
测试所有模块是否正常工作
"""

import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_processing():
    """测试数据预处理模块"""
    print("=" * 50)
    print("测试数据预处理模块")
    print("=" * 50)
    
    from data_processing import CSIPreprocessor, SkeletonPreprocessor
    
    # 测试CSI预处理器
    print("测试CSI预处理器...")
    csi_preprocessor = CSIPreprocessor(
        num_antennas=3,
        num_subcarriers=30,
        time_length=297,
        patch_size=8
    )
    
    # 创建模拟CSI数据
    batch_size = 2
    csi_data = torch.randn(batch_size, 3, 30, 297)
    
    patches, spectrogram = csi_preprocessor(csi_data)
    print(f"✅ CSI预处理成功: {csi_data.shape} -> {patches.shape}")
    
    # 测试骨骼点预处理器
    print("测试骨骼点预处理器...")
    skeleton_preprocessor = SkeletonPreprocessor(num_joints=17, coord_dim=2)
    
    skeleton_data = torch.randn(batch_size, 17, 2) * 100
    processed_skeleton = skeleton_preprocessor(skeleton_data)
    print(f"✅ 骨骼点预处理成功: {skeleton_data.shape} -> {processed_skeleton.shape}")
    
    return csi_preprocessor, skeleton_preprocessor


def test_models(csi_preprocessor):
    """测试模型"""
    print("\n" + "=" * 50)
    print("测试模型")
    print("=" * 50)
    
    from models import TeacherModel, StudentModel
    
    # 测试老师模型
    print("测试老师模型...")
    teacher_model = TeacherModel(
        num_joints=17,
        coord_dim=2,
        embed_dim=384,
        depth=6,
        num_heads=6
    )
    
    batch_size = 2
    skeleton_input = torch.randn(batch_size, 17, 2)
    
    # 测试预训练模式
    loss, pred, mask = teacher_model(skeleton_input)
    print(f"✅ 老师模型预训练: 损失={loss.item():.4f}")
    
    # 测试特征提取模式
    features = teacher_model.forward_features(skeleton_input, mask_ratio=0.0)
    print(f"✅ 老师模型特征提取: {len(features)}层特征")
    
    # 测试学生模型
    print("测试学生模型...")
    student_model = StudentModel(
        num_patches=csi_preprocessor.num_patches,
        patch_dim=csi_preprocessor.patch_dim,
        embed_dim=384,
        depth=6,
        num_heads=6,
        num_joints=17,
        coord_dim=2
    )
    
    csi_patches = torch.randn(batch_size, csi_preprocessor.num_patches, csi_preprocessor.patch_dim)
    outputs = student_model(csi_patches)
    
    print(f"✅ 学生模型输出:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"   {key}: {len(value)}个特征")
    
    return teacher_model, student_model, features, outputs


def test_losses(teacher_features, student_outputs):
    """测试损失函数"""
    print("\n" + "=" * 50)
    print("测试损失函数")
    print("=" * 50)
    
    from losses import MAELoss, DistillationLoss, ContrastiveLoss, CombinedLoss
    
    batch_size = 2
    
    # 测试MAE损失
    print("测试MAE损失...")
    mae_loss_fn = MAELoss()
    
    pred_patches = torch.randn(batch_size, 64, 256)
    target_patches = torch.randn(batch_size, 64, 256)
    mask = torch.randint(0, 2, (batch_size, 64)).bool()
    
    mae_loss = mae_loss_fn(pred_patches, target_patches, mask)
    print(f"✅ MAE损失: {mae_loss.item():.4f}")
    
    # 测试蒸馏损失
    print("测试蒸馏损失...")
    distill_loss_fn = DistillationLoss()
    
    student_features = [torch.randn(batch_size, 65, 384) for _ in range(3)]
    
    distill_loss = distill_loss_fn(student_features, teacher_features)
    print(f"✅ 蒸馏损失: {distill_loss.item():.4f}")
    
    # 测试对比学习损失
    print("测试对比学习损失...")
    contrast_loss_fn = ContrastiveLoss()
    
    anchor_features = torch.randn(batch_size, 256)
    positive_features = torch.randn(batch_size, 256)
    labels = torch.randint(0, 2, (batch_size,))
    
    contrast_loss = contrast_loss_fn(anchor_features, positive_features, labels)
    print(f"✅ 对比学习损失: {contrast_loss.item():.4f}")
    
    # 测试组合损失
    print("测试组合损失...")
    combined_loss_fn = CombinedLoss()
    
    skeleton_pred = torch.randn(batch_size, 17, 2)
    skeleton_target = torch.randn(batch_size, 17, 2)
    
    total_loss, loss_dict = combined_loss_fn(
        pred_patches, target_patches, mask,
        skeleton_pred, skeleton_target,
        student_features, teacher_features,
        anchor_features, positive_features, labels
    )
    
    print(f"✅ 组合损失: {total_loss.item():.4f}")
    print("   各项损失:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value.item():.4f}")


def test_utils():
    """测试工具函数"""
    print("\n" + "=" * 50)
    print("测试工具函数")
    print("=" * 50)
    
    from utils import (
        LossTracker, calculate_skeleton_metrics,
        get_parameter_count, print_model_info
    )
    
    # 测试损失跟踪器
    print("测试损失跟踪器...")
    tracker = LossTracker()
    
    for i in range(3):
        loss_dict = {
            'total_loss': torch.tensor(1.0 - i * 0.1),
            'mae_loss': torch.tensor(0.5 - i * 0.05)
        }
        tracker.update(loss_dict)
    
    current_losses = tracker.get_current_losses()
    print(f"✅ 损失跟踪: {current_losses}")
    
    # 测试骨骼点指标
    print("测试骨骼点指标...")
    pred_skeleton = torch.randn(4, 17, 2)
    target_skeleton = torch.randn(4, 17, 2)
    
    metrics = calculate_skeleton_metrics(pred_skeleton, target_skeleton)
    print(f"✅ 骨骼点指标: MPJPE={metrics['MPJPE']:.4f}")
    
    # 测试参数计数
    print("测试参数计数...")
    import torch.nn as nn
    test_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    param_count = get_parameter_count(test_model)
    print(f"✅ 参数计数: {param_count}")


def test_integration():
    """集成测试"""
    print("\n" + "=" * 50)
    print("集成测试")
    print("=" * 50)
    
    # 模拟完整的训练步骤
    from data_processing import CSIPreprocessor, SkeletonPreprocessor
    from models import TeacherModel, StudentModel
    from losses import CombinedLoss
    
    print("创建组件...")
    
    # 预处理器
    csi_preprocessor = CSIPreprocessor(patch_size=8)
    skeleton_preprocessor = SkeletonPreprocessor()
    
    # 模型
    teacher_model = TeacherModel(embed_dim=256, depth=4, num_heads=4)
    student_model = StudentModel(
        num_patches=csi_preprocessor.num_patches,
        patch_dim=csi_preprocessor.patch_dim,
        embed_dim=256,
        depth=4,
        num_heads=4
    )
    
    # 损失函数
    loss_fn = CombinedLoss()
    
    print("模拟训练步骤...")
    
    # 模拟数据
    batch_size = 2
    csi_data = torch.randn(batch_size, 3, 30, 297)
    rgb_skeleton = torch.randn(batch_size, 17, 2)
    
    # 数据预处理
    csi_patches, _ = csi_preprocessor(csi_data)
    processed_skeleton = skeleton_preprocessor(rgb_skeleton)
    
    # 老师模型前向传播
    teacher_model.eval()
    with torch.no_grad():
        teacher_features = teacher_model.forward_features(processed_skeleton, mask_ratio=0.0)
    
    # 学生模型前向传播
    student_model.train()
    student_outputs = student_model(csi_patches)
    
    # 计算损失
    total_loss, loss_dict = loss_fn(
        student_outputs['reconstructed_patches'], csi_patches, student_outputs['mask'],
        student_outputs['skeleton_pred'], processed_skeleton,
        student_outputs['distill_features'], teacher_features,
        student_outputs['contrast_features'][:1],
        student_outputs['contrast_features'][1:],
        torch.randint(0, 2, (1,))
    )
    
    print(f"✅ 集成测试成功!")
    print(f"   总损失: {total_loss.item():.4f}")
    print(f"   预测骨骼点形状: {student_outputs['skeleton_pred'].shape}")
    
    # 模拟反向传播
    total_loss.backward()
    print("✅ 反向传播成功!")


def main():
    """主测试函数"""
    print("🚀 开始Enhanced Multi-Modal DMAE组件测试")
    print("=" * 60)
    
    try:
        # 测试各个组件
        csi_preprocessor, skeleton_preprocessor = test_data_processing()
        teacher_model, student_model, teacher_features, student_outputs = test_models(csi_preprocessor)
        test_losses(teacher_features, student_outputs)
        test_utils()
        test_integration()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！Enhanced Multi-Modal DMAE组件工作正常。")
        print("=" * 60)
        
        # 打印模型信息摘要
        print("\n📊 模型信息摘要:")
        from utils import get_parameter_count
        
        teacher_params = get_parameter_count(teacher_model)
        student_params = get_parameter_count(student_model)
        
        print(f"老师模型参数: {teacher_params['total_params']:,}")
        print(f"学生模型参数: {student_params['total_params']:,}")
        print(f"总参数量: {teacher_params['total_params'] + student_params['total_params']:,}")
        
        print("\n🎯 下一步:")
        print("1. 准备MMFi数据集")
        print("2. 运行: python example.py demo")
        print("3. 开始训练: python train.py <dataset_root> <dataset_config>")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

