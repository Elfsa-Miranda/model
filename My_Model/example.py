"""
Enhanced Multi-Modal DMAE 使用示例
展示如何使用增强型多模态DMAE进行训练和推理

使用方法:
1. 训练模型: python example.py train --dataset_root C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave --dataset_config ../MMFi_dataset/config.yaml
2. 测试模型: python example.py test --model_path /path/to/model.pth --test_data C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave
3. 推理单个样本: python example.py infer --model_path /path/to/model.pth --csi_file /path/to/csi.mat
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TeacherModel, StudentModel
from data_processing import CSIPreprocessor, SkeletonPreprocessor
from train import EnhancedDMAETrainer, load_config
from utils import (
    load_checkpoint, calculate_skeleton_metrics,
    visualize_skeleton_prediction, print_model_info, get_device
)


class EnhancedDMAEInference:
    """增强型多模态DMAE推理器"""
    
    def __init__(self, model_path, config_path=None):
        """
        Args:
            model_path: 学生模型路径
            config_path: 配置文件路径
        """
        self.device = get_device()
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            self.config = self.get_default_inference_config()
        
        # 初始化预处理器
        self.csi_preprocessor = CSIPreprocessor(**self.config['csi_preprocessor'])
        self.skeleton_preprocessor = SkeletonPreprocessor(**self.config['skeleton_preprocessor'])
        
        # 初始化学生模型
        self.student_model = StudentModel(
            num_patches=self.csi_preprocessor.num_patches,
            patch_dim=self.csi_preprocessor.patch_dim,
            **self.config['student_model']
        ).to(self.device)
        
        # 加载模型权重
        self.load_model(model_path)
        
        print_model_info(self.student_model, "Student Model (Inference)")
    
    def get_default_inference_config(self):
        """获取默认推理配置"""
        return {
            'csi_preprocessor': {
                'num_antennas': 3,
                'num_subcarriers': 30,
                'time_length': 297,
                'stft_window': 32,
                'stft_hop': 16,
                'patch_size': 16,
                'normalize': True
            },
            'skeleton_preprocessor': {
                'num_joints': 17,
                'coord_dim': 2,
                'normalize': True
            },
            'student_model': {
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'decoder_embed_dim': 512,
                'decoder_depth': 8,
                'decoder_num_heads': 16,
                'num_joints': 17,
                'coord_dim': 2,
                'contrast_dim': 256,
                'mask_ratio': 0.75
            }
        }
    
    def load_model(self, model_path):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.student_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.student_model.load_state_dict(checkpoint)
        
        self.student_model.eval()
        print(f"✅ 模型已加载: {model_path}")
    
    def predict_skeleton(self, csi_data):
        """
        从CSI数据预测骨骼点
        
        Args:
            csi_data: CSI数据 [batch, freq, time, antennas] (MMFi格式) 或 
                     [freq, time, antennas] (单样本) 或
                     [batch, antennas, subcarriers, time] (标准格式)
        
        Returns:
            skeleton: 预测的骨骼点 [batch, num_joints, coord_dim] 或 [num_joints, coord_dim]
        """
        # 确保输入是4维的
        if len(csi_data.shape) == 3:
            csi_data = csi_data.unsqueeze(0)  # 添加batch维度
            single_sample = True
        else:
            single_sample = False
        
        # 转换为tensor并移动到设备
        if not isinstance(csi_data, torch.Tensor):
            csi_data = torch.tensor(csi_data, dtype=torch.float32)
        csi_data = csi_data.to(self.device)
        
        with torch.no_grad():
            # 检查数据格式并转换
            if csi_data.shape[-1] <= 10:  # MMFi格式: [batch, freq, time, antennas]
                # 转换为标准格式: [batch, antennas, freq, time]
                csi_data = csi_data.permute(0, 3, 1, 2)
            
            # 预处理CSI数据
            csi_patches, _ = self.csi_preprocessor(csi_data)
            
            # 学生模型推理（不使用掩码）
            outputs = self.student_model(csi_patches, mask_ratio=0.0)
            skeleton_pred = outputs['skeleton_pred']
            
            # 如果输入是单个样本，移除batch维度
            if single_sample:
                skeleton_pred = skeleton_pred.squeeze(0)
        
        return skeleton_pred.cpu()
    
    def predict_with_confidence(self, csi_data, num_samples=10):
        """
        使用蒙特卡洛dropout估计预测置信度
        
        Args:
            csi_data: CSI数据
            num_samples: 采样次数
        
        Returns:
            mean_skeleton: 平均预测骨骼点
            std_skeleton: 标准差（置信度指标）
        """
        # 启用dropout进行蒙特卡洛采样
        self.student_model.train()
        
        predictions = []
        for _ in range(num_samples):
            pred = self.predict_skeleton(csi_data)
            predictions.append(pred)
        
        # 恢复评估模式
        self.student_model.eval()
        
        # 计算统计量
        predictions = torch.stack(predictions)
        mean_skeleton = predictions.mean(dim=0)
        std_skeleton = predictions.std(dim=0)
        
        return mean_skeleton, std_skeleton


def train_model(args):
    """训练模型"""
    print("=" * 60)
    print("开始训练Enhanced Multi-Modal DMAE")
    print("=" * 60)
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"⚠️  配置文件不存在: {args.config}，使用默认配置")
        from train import get_default_config
        config = get_default_config()
    
    # 更新输出目录
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # 创建训练器
    trainer = EnhancedDMAETrainer(config)
    
    # 开始训练
    trainer.train(args.dataset_root, args.dataset_config)


def test_model(args):
    """测试模型"""
    print("=" * 60)
    print("测试Enhanced Multi-Modal DMAE")
    print("=" * 60)
    
    # 创建推理器
    inference = EnhancedDMAEInference(args.model_path, args.config)
    
    # 加载测试数据
    from mmfi_dataloader import create_enhanced_mmfi_dataloaders
    
    # 这里需要一个测试配置文件
    test_config_path = args.test_config or args.dataset_config
    _, test_loader = create_enhanced_mmfi_dataloaders(
        args.dataset_root, test_config_path, batch_size=1
    )
    
    # 测试循环
    all_pred_skeletons = []
    all_target_skeletons = []
    
    print("正在进行模型测试...")
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 100:  # 限制测试样本数量
            break
        
        csi_data = batch['csi_data']
        target_skeleton = batch['rgb_skeleton']
        
        # 预测
        pred_skeleton = inference.predict_skeleton(csi_data)
        
        all_pred_skeletons.append(pred_skeleton)
        all_target_skeletons.append(target_skeleton)
    
    # 计算指标
    pred_skeletons = torch.cat(all_pred_skeletons, dim=0)
    target_skeletons = torch.cat(all_target_skeletons, dim=0)
    
    metrics = calculate_skeleton_metrics(pred_skeletons, target_skeletons)
    
    print("\n=== 测试结果 ===")
    print(f"MPJPE: {metrics['MPJPE']:.4f}")
    for key, value in metrics.items():
        if key.startswith('PCK'):
            print(f"{key}: {value:.4f}")
    
    # 可视化几个样本
    output_dir = args.output_dir or "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(5, len(pred_skeletons))):
        vis_path = os.path.join(output_dir, f"test_sample_{i}.png")
        visualize_skeleton_prediction(
            pred_skeletons[i], target_skeletons[i], vis_path
        )
    
    print(f"✅ 测试完成，结果保存到: {output_dir}")


def infer_single(args):
    """推理单个样本"""
    print("=" * 60)
    print("单样本推理")
    print("=" * 60)
    
    # 创建推理器
    inference = EnhancedDMAEInference(args.model_path, args.config)
    
    # 加载CSI数据
    if args.csi_file.endswith('.mat'):
        import scipy.io as scio
        csi_data = scio.loadmat(args.csi_file)['CSIamp']
    elif args.csi_file.endswith('.npy'):
        csi_data = np.load(args.csi_file)
    else:
        raise ValueError(f"不支持的文件格式: {args.csi_file}")
    
    print(f"CSI数据形状: {csi_data.shape}")
    
    # 预测骨骼点
    pred_skeleton = inference.predict_skeleton(csi_data)
    print(f"预测骨骼点形状: {pred_skeleton.shape}")
    
    # 可视化结果
    output_dir = args.output_dir or "./inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    vis_path = os.path.join(output_dir, "inference_result.png")
    
    # 创建简单的可视化
    plt.figure(figsize=(8, 6))
    skeleton = pred_skeleton.numpy()
    plt.scatter(skeleton[:, 0], skeleton[:, 1], c='red', s=50, alpha=0.7)
    
    # 添加关节连接
    joint_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上身
        (5, 11), (6, 12), (11, 12),  # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16)  # 下身
    ]
    
    for connection in joint_connections:
        if connection[0] < len(skeleton) and connection[1] < len(skeleton):
            x_coords = [skeleton[connection[0], 0], skeleton[connection[1], 0]]
            y_coords = [skeleton[connection[0], 1], skeleton[connection[1], 1]]
            plt.plot(x_coords, y_coords, 'r-', alpha=0.5)
    
    plt.title('Predicted Skeleton from CSI')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 推理完成，结果保存到: {vis_path}")
    
    # 保存预测结果
    result_path = os.path.join(output_dir, "predicted_skeleton.npy")
    np.save(result_path, pred_skeleton.numpy())
    print(f"✅ 预测骨骼点保存到: {result_path}")


def demo():
    """演示模式"""
    print("=" * 60)
    print("Enhanced Multi-Modal DMAE 演示")
    print("=" * 60)
    
    # 创建模拟数据
    print("创建模拟数据...")
    
    # 模拟CSI数据
    batch_size = 4
    num_antennas = 3
    num_subcarriers = 30
    time_length = 297
    
    csi_data = torch.randn(batch_size, num_antennas, num_subcarriers, time_length)
    print(f"模拟CSI数据形状: {csi_data.shape}")
    
    # 模拟RGB骨骼点数据
    rgb_skeleton = torch.randn(batch_size, 17, 2) * 100  # 模拟像素坐标
    print(f"模拟RGB骨骼点形状: {rgb_skeleton.shape}")
    
    # 测试数据预处理
    print("\n测试数据预处理...")
    from data_processing import CSIPreprocessor, SkeletonPreprocessor
    
    csi_preprocessor = CSIPreprocessor()
    skeleton_preprocessor = SkeletonPreprocessor()
    
    csi_patches, csi_spectrogram = csi_preprocessor(csi_data)
    processed_skeleton = skeleton_preprocessor(rgb_skeleton)
    
    print(f"CSI补丁形状: {csi_patches.shape}")
    print(f"CSI时频谱形状: {csi_spectrogram.shape}")
    print(f"处理后骨骼点形状: {processed_skeleton.shape}")
    
    # 测试模型
    print("\n测试模型...")
    from models import TeacherModel, StudentModel
    
    teacher_model = TeacherModel(embed_dim=384, depth=6)
    student_model = StudentModel(
        num_patches=csi_preprocessor.num_patches,
        patch_dim=csi_preprocessor.patch_dim,
        embed_dim=384,
        depth=6
    )
    
    print_model_info(teacher_model, "Teacher Model")
    print_model_info(student_model, "Student Model")
    
    # 老师模型前向传播
    teacher_loss, teacher_pred, teacher_mask = teacher_model(processed_skeleton)
    print(f"老师模型损失: {teacher_loss.item():.4f}")
    
    # 学生模型前向传播
    student_outputs = student_model(csi_patches)
    print(f"学生模型输出键: {list(student_outputs.keys())}")
    print(f"预测骨骼点形状: {student_outputs['skeleton_pred'].shape}")
    
    # 测试损失函数
    print("\n测试损失函数...")
    from losses import CombinedLoss
    
    loss_fn = CombinedLoss(
        mae_weight=1.0,
        distill_weight=1.0,
        contrast_weight=0.5
    )
    
    # 模拟老师特征
    teacher_features = teacher_model.forward_features(processed_skeleton, mask_ratio=0.0)
    
    # 计算组合损失
    total_loss, loss_dict = loss_fn(
        student_outputs['reconstructed_patches'], csi_patches, student_outputs['mask'],
        student_outputs['skeleton_pred'], processed_skeleton,
        student_outputs['distill_features'], teacher_features,
        student_outputs['contrast_features'][:2],  # anchor
        student_outputs['contrast_features'][2:],  # positive
        torch.randint(0, 2, (2,))  # 随机标签
    )
    
    print(f"总损失: {total_loss.item():.4f}")
    print("各项损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ 演示完成！所有组件工作正常。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Modal DMAE Example")
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('dataset_root', type=str, help='MMFi数据集根目录')
    train_parser.add_argument('dataset_config', type=str, help='数据集配置文件')
    train_parser.add_argument('--config', type=str, default='config.yaml', help='训练配置文件')
    train_parser.add_argument('--output_dir', type=str, help='输出目录')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('model_path', type=str, help='模型路径')
    test_parser.add_argument('dataset_root', type=str, help='测试数据集根目录')
    test_parser.add_argument('dataset_config', type=str, help='数据集配置文件')
    test_parser.add_argument('--config', type=str, help='模型配置文件')
    test_parser.add_argument('--test_config', type=str, help='测试配置文件')
    test_parser.add_argument('--output_dir', type=str, help='输出目录')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='推理单个样本')
    infer_parser.add_argument('model_path', type=str, help='模型路径')
    infer_parser.add_argument('csi_file', type=str, help='CSI数据文件')
    infer_parser.add_argument('--config', type=str, help='模型配置文件')
    infer_parser.add_argument('--output_dir', type=str, help='输出目录')
    
    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='演示模式')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'test':
        test_model(args)
    elif args.command == 'infer':
        infer_single(args)
    elif args.command == 'demo':
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
