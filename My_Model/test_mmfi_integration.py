"""
测试My_Model与MMFi数据集的整合
验证数据加载和模型训练是否正常工作
"""

import os
import sys
import torch
import yaml

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mmfi_dataloader import create_enhanced_mmfi_dataloaders, test_enhanced_mmfi_dataloader
from models import TeacherModel, StudentModel
from data_processing import CSIPreprocessor, SkeletonPreprocessor
from losses import CombinedLoss
from utils import calculate_skeleton_metrics, print_model_info


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试MMFi数据加载")
    print("=" * 60)
    
    # 测试数据加载器
    success = test_enhanced_mmfi_dataloader()
    
    if not success:
        print("⚠️  数据加载测试失败，请检查数据集路径")
        print("请确保:")
        print("1. MMFi数据集已正确下载和解压")
        print("2. 数据集路径正确设置")
        print("3. 数据集结构符合要求")
        return False
    
    return True


def test_model_with_real_data():
    """使用真实MMFi数据测试模型"""
    print("\n" + "=" * 60)
    print("使用真实MMFi数据测试模型")
    print("=" * 60)
    
    # 数据路径配置
    data_root = "C:\\tangyx\\MMFi_Dataset\\filtered_mmwave\\filtered_mmwave"  # 请修改为你的数据集路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MMFi_dataset', 'config.yaml')
    
    if not os.path.exists(data_root):
        print(f"⚠️  数据集路径不存在: {data_root}")
        print("请修改data_root变量为你的MMFi数据集路径")
        return False
    
    if not os.path.exists(config_path):
        print(f"⚠️  配置文件不存在: {config_path}")
        return False
    
    try:
        # 创建数据加载器
        print("创建数据加载器...")
        train_loader, val_loader = create_enhanced_mmfi_dataloaders(
            data_root, config_path, batch_size=4, num_workers=0
        )
        
        # 获取一个批次的数据
        print("获取数据批次...")
        batch = next(iter(train_loader))
        
        print(f"批次数据形状:")
        print(f"  RGB骨骼点: {batch['rgb_skeleton'].shape}")
        print(f"  CSI数据: {batch['csi_data'].shape}")
        print(f"  GT骨骼点: {batch['gt_skeleton'].shape}")
        
        # 创建预处理器
        print("创建预处理器...")
        
        # 根据实际数据调整CSI预处理器参数
        csi_shape = batch['csi_data'].shape  # [batch, freq, time, antennas]
        freq_bins, time_bins, num_antennas = csi_shape[1], csi_shape[2], csi_shape[3]
        
        csi_preprocessor = CSIPreprocessor(
            num_antennas=num_antennas,
            num_subcarriers=freq_bins,
            time_length=time_bins,
            patch_size=8,  # 使用较小的patch_size
            normalize=True
        )
        
        skeleton_preprocessor = SkeletonPreprocessor(
            num_joints=17,
            coord_dim=2,
            normalize=True
        )
        
        # 预处理数据
        print("预处理数据...")
        rgb_skeleton = batch['rgb_skeleton']
        csi_data = batch['csi_data']
        
        # 预处理RGB骨骼点
        processed_rgb = skeleton_preprocessor(rgb_skeleton)
        print(f"处理后RGB骨骼点形状: {processed_rgb.shape}")
        
        # 预处理CSI数据
        # 需要转换维度: [batch, freq, time, antennas] -> [batch, antennas, freq, time]
        csi_data_reshaped = csi_data.permute(0, 3, 1, 2)
        csi_patches, csi_spectrogram = csi_preprocessor(csi_data_reshaped)
        print(f"CSI补丁形状: {csi_patches.shape}")
        print(f"CSI时频谱形状: {csi_spectrogram.shape}")
        
        # 创建模型
        print("创建模型...")
        teacher_model = TeacherModel(
            num_joints=17,
            coord_dim=2,
            embed_dim=256,  # 使用较小的维度进行测试
            depth=4,
            num_heads=4,
            mask_ratio=0.75
        )
        
        student_model = StudentModel(
            num_patches=csi_preprocessor.num_patches,
            patch_dim=csi_preprocessor.patch_dim,
            embed_dim=256,
            depth=4,
            num_heads=4,
            num_joints=17,
            coord_dim=2,
            contrast_dim=128,
            mask_ratio=0.75
        )
        
        print_model_info(teacher_model, "Teacher Model")
        print_model_info(student_model, "Student Model")
        
        # 测试模型前向传播
        print("测试模型前向传播...")
        
        # 老师模型
        teacher_loss, teacher_pred, teacher_mask = teacher_model(processed_rgb)
        print(f"老师模型损失: {teacher_loss.item():.4f}")
        
        # 老师模型特征提取
        teacher_features = teacher_model.forward_features(processed_rgb, mask_ratio=0.0)
        print(f"老师模型特征层数: {len(teacher_features)}")
        
        # 学生模型
        student_outputs = student_model(csi_patches)
        print(f"学生模型输出键: {list(student_outputs.keys())}")
        
        # 测试组合损失
        print("测试组合损失...")
        loss_fn = CombinedLoss(
            mae_weight=1.0,
            distill_weight=1.0,
            contrast_weight=0.5
        )
        
        batch_size = processed_rgb.shape[0]
        contrast_labels = torch.ones(max(1, batch_size//2), dtype=torch.long)
        
        total_loss, loss_dict = loss_fn(
            student_outputs['reconstructed_patches'], csi_patches, student_outputs['mask'],
            student_outputs['skeleton_pred'], processed_rgb,
            student_outputs['distill_features'], teacher_features,
            student_outputs['contrast_features'][:max(1, batch_size//2)],
            student_outputs['contrast_features'][:max(1, batch_size//2)],  # 简化处理
            contrast_labels
        )
        
        print(f"总损失: {total_loss.item():.4f}")
        print("各项损失:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")
        
        # 测试骨骼点指标
        print("测试骨骼点指标...")
        metrics = calculate_skeleton_metrics(
            student_outputs['skeleton_pred'], processed_rgb
        )
        print(f"MPJPE: {metrics['MPJPE']:.4f}")
        
        print("\n✅ 使用真实MMFi数据的模型测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试一个完整的训练步骤"""
    print("\n" + "=" * 60)
    print("测试完整训练步骤")
    print("=" * 60)
    
    try:
        # 这里可以添加一个简化的训练步骤测试
        print("模拟训练步骤...")
        
        # 创建模拟优化器
        import torch.optim as optim
        
        # 使用之前测试中创建的模型
        teacher_model = TeacherModel(embed_dim=128, depth=2, num_heads=4)
        student_model = StudentModel(
            num_patches=64,
            patch_dim=64,
            embed_dim=128,
            depth=2,
            num_heads=4
        )
        
        teacher_optimizer = optim.AdamW(teacher_model.parameters(), lr=1e-4)
        student_optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)
        
        # 模拟数据
        batch_size = 2
        rgb_skeleton = torch.randn(batch_size, 17, 2)
        csi_patches = torch.randn(batch_size, 64, 64)
        
        # 老师模型训练步骤
        teacher_optimizer.zero_grad()
        teacher_loss, _, _ = teacher_model(rgb_skeleton)
        teacher_loss.backward()
        teacher_optimizer.step()
        
        print(f"老师模型训练步骤完成，损失: {teacher_loss.item():.4f}")
        
        # 学生模型训练步骤
        teacher_model.eval()
        with torch.no_grad():
            teacher_features = teacher_model.forward_features(rgb_skeleton, mask_ratio=0.0)
        
        student_optimizer.zero_grad()
        student_outputs = student_model(csi_patches)
        
        # 简化损失计算
        mae_loss = torch.nn.MSELoss()(student_outputs['skeleton_pred'], rgb_skeleton)
        mae_loss.backward()
        student_optimizer.step()
        
        print(f"学生模型训练步骤完成，损失: {mae_loss.item():.4f}")
        
        print("✅ 训练步骤测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始My_Model与MMFi数据集整合测试")
    print("=" * 60)
    
    test_results = []
    
    # 测试1: 数据加载
    print("\n📊 测试1: 数据加载")
    result1 = test_data_loading()
    test_results.append(("数据加载", result1))
    
    # 测试2: 模型与真实数据
    if result1:  # 只有数据加载成功才进行模型测试
        print("\n🤖 测试2: 模型与真实数据")
        result2 = test_model_with_real_data()
        test_results.append(("模型与真实数据", result2))
    else:
        print("\n⚠️  跳过模型测试（数据加载失败）")
        test_results.append(("模型与真实数据", False))
    
    # 测试3: 训练步骤
    print("\n🏃 测试3: 训练步骤")
    result3 = test_training_step()
    test_results.append(("训练步骤", result3))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！My_Model与MMFi数据集整合成功。")
        print("\n📝 下一步:")
        print("1. 确保MMFi数据集路径正确")
        print("2. 运行完整训练: python train.py <dataset_root> <config_file>")
        print("3. 监控训练过程和损失变化")
    else:
        print("⚠️  部分测试失败，请检查:")
        print("1. MMFi数据集是否正确安装")
        print("2. 数据集路径是否正确")
        print("3. 依赖库是否完整安装")
    
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)









