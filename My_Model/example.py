"""
Enhanced Multi-Modal DMAE 使用示例 - 修复版 + 自动模型识别
展示如何使用增强型多模态DMAE进行训练和推理

修复内容:
1. ✅ 延迟初始化 StudentModel (等待 num_patches 确定)
2. ✅ 自动从数据中计算 num_patches
3. ✅ 兼容训练框架的初始化流程
4. ✅ 增强错误提示和调试信息
5. ✅ 自动识别并加载 TeacherModel 或 StudentModel
6. ✅ 支持部分权重加载(strict=False)

使用方法:
1. 训练模型: python example.py train --dataset_root <path> --dataset_config <config.yaml>
2. 测试模型: python example.py test --model_path <model.pth> --dataset_root <path> --dataset_config <config.yaml>
3. 推理单个样本: python example.py infer --model_path <model.pth> --csi_file <csi.mat>
4. 演示模式: python example.py demo
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
    """
    增强型多模态DMAE推理器 - 自动模型识别版

    新增功能:
    - 自动识别 TeacherModel 或 StudentModel
    - 支持部分权重加载(strict=False)
    - 完整的加载报告输出
    """

    def __init__(self, model_path, config_path=None, num_patches=None, patch_dim=None):
        """
        Args:
            model_path: 模型路径
            config_path: 配置文件路径
            num_patches: 可选,提前传入的num_patches,避免重复计算
            patch_dim: 可选,提前传入的patch_dim,避免重复计算
        """
        self.device = get_device()
        self.model_path = model_path
        self.model_type = None  # 'teacher' 或 'student'
        self.model = None

        # 加载配置
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            self.config = self.get_default_inference_config()

        # 初始化预处理器
        self.csi_preprocessor = CSIPreprocessor(**self.config['csi_preprocessor'])
        self.skeleton_preprocessor = SkeletonPreprocessor(**self.config['skeleton_preprocessor'])

        # 模型初始化状态
        self._model_initialized = False
        self._num_patches_determined = False

        # 如果提前给定 num_patches 和 patch_dim,直接使用
        if num_patches is not None and patch_dim is not None:
            self.num_patches = num_patches
            self.patch_dim = patch_dim
            self._num_patches_determined = True
        else:
            self.num_patches = None
            self.patch_dim = None

        # 自动检测模型类型
        self._detect_model_type()

        print("✅ 推理器初始化完成 (模型将在首次使用时加载)")

    def _detect_model_type(self):
        """自动检测模型类型"""
        model_path_lower = self.model_path.lower()

        if 'teacher' in model_path_lower:
            self.model_type = 'teacher'
            print("✅ 检测到 Teacher 模型")
        else:
            self.model_type = 'student'
            print("✅ 检测到 Student 模型")

    def get_default_inference_config(self):
        """获取默认推理配置"""
        print("⚠️  未提供 --config 参数, 正在使用 example.py 中的默认配置...")
        return {
            'csi_preprocessor': {
                'num_antennas': 3,
                'num_subcarriers': 114,
                'time_length': 10,
                'stft_window': 64,
                'stft_hop': 16,
                'patch_size': 8,
                'normalize': True
            },
            'skeleton_preprocessor': {
                'num_joints': 17,
                'coord_dim': 2,
                'normalize': True
            },
            'teacher_model': {
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'decoder_embed_dim': 512,
                'decoder_depth': 8,
                'decoder_num_heads': 16,
                'num_joints': 17,
                'coord_dim': 2,
                'mask_ratio': 0.75
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
                'contrast_dim': 128,
                'mask_ratio': 0.75,
                'num_antennas': 3,
                'use_multi_attn': True
            }
        }

    def _load_model_auto(self):
        """自动加载模型权重"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        print("\n" + "=" * 60)
        print("🔧 正在加载模型权重...")
        print("=" * 60)

        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 提取state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 尝试严格加载
        try:
            self.model.load_state_dict(state_dict, strict=True)
            load_info = {'missing_keys': [], 'unexpected_keys': []}
            loaded_percent = 100.0
            skipped_layers = 0
        except RuntimeError as e:
            # 如果严格加载失败,使用非严格模式
            print(f"⚠️  严格加载失败,切换到 strict=False 模式")
            load_info = self.model.load_state_dict(state_dict, strict=False)

            # 计算加载比例
            total_params = len(self.model.state_dict())
            missing_params = len(load_info.get('missing_keys', []))
            unexpected_params = len(load_info.get('unexpected_keys', []))
            skipped_layers = missing_params + unexpected_params
            loaded_percent = ((total_params - missing_params) / total_params) * 100 if total_params > 0 else 0.0

            # 打印警告
            if missing_params > 0:
                print(f"⚠️  权重部分不匹配,已跳过 {skipped_layers} 层:")
                print(f"   缺失的键 ({missing_params}): {load_info['missing_keys'][:5]}...")
            if unexpected_params > 0:
                print(f"   意外的键 ({unexpected_params}): {load_info['unexpected_keys'][:5]}...")

        self.model.eval()

        # 打印加载报告
        print("\n" + "=" * 60)
        print("🧩 模型加载报告")
        print("=" * 60)
        print(f"类型: {self.model.__class__.__name__}")
        print(f"路径: {self.model_path}")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"总参数: {total_params:,}")
        print(f"已加载: {loaded_percent:.1f}%")
        print(f"跳过层数: {skipped_layers}")
        print("=" * 60 + "\n")

        if loaded_percent < 50:
            print("⚠️  警告: 加载的参数比例过低,模型可能无法正常工作")

    def _ensure_model_initialized(self, sample_csi_data=None):
        """确保模型已初始化"""
        if self._model_initialized:
            return

        print("\n" + "=" * 60)
        print("🔧 正在初始化模型...")
        print("=" * 60)

        if self.model_type == 'teacher':
            # 教师模型直接初始化
            self._initialize_teacher_model()
        else:
            # 学生模型需要确定 num_patches
            self._initialize_student_model(sample_csi_data)

        # 加载权重
        self._load_model_auto()

        # 打印模型信息
        model_name = "Teacher Model" if self.model_type == 'teacher' else "Student Model"
        print_model_info(self.model, f"{model_name} (Inference)")

        self._model_initialized = True
        print("=" * 60)
        print("✅ 模型初始化完成\n")

    def _initialize_teacher_model(self):
        """初始化教师模型"""
        try:
            self.model = TeacherModel(
                **self.config['teacher_model']
            ).to(self.device)
            print("✅ 教师模型创建成功")
        except Exception as e:
            raise RuntimeError(f"创建教师模型失败: {e}") from e

    def _initialize_student_model(self, sample_csi_data):
        """初始化学生模型"""
        # 步骤1: 确定 num_patches
        if not self._num_patches_determined:
            if sample_csi_data is None:
                raise RuntimeError(
                    "学生模型需要样本数据来确定 num_patches,但未提供 sample_csi_data"
                )

            try:
                # 确保是4维张量
                if len(sample_csi_data.shape) == 3:
                    sample_csi_data = sample_csi_data.unsqueeze(0)

                # 检查数据格式并转换
                if sample_csi_data.shape[-1] == self.csi_preprocessor.num_antennas:
                    print(f"   检测到MMFi格式 [B, F, T, A]: {sample_csi_data.shape}")
                    print("   ...正在转换为 [B, A, F, T]")
                    sample_csi_data = sample_csi_data.permute(0, 3, 1, 2)
                else:
                    print(f"   假设输入已是 [B, A, F, T] 格式: {sample_csi_data.shape}")

                print(f"   用于初始化的样本CSI形状: {sample_csi_data.shape}")

                # 运行预处理器
                with torch.no_grad():
                    patches, _ = self.csi_preprocessor(sample_csi_data)

                self.num_patches = self.csi_preprocessor.num_patches
                self.patch_dim = self.csi_preprocessor.patch_dim

                if self.num_patches is None:
                    raise RuntimeError(
                        "CSI预处理器未能确定 num_patches!\n"
                        f"Patches shape: {patches.shape}\n"
                        f"请检查数据格式是否正确, 以及 csi_preprocessor 配置是否与训练时一致"
                    )

                print(f"   ✅ num_patches 已确定: {self.num_patches}")
                print(f"   ✅ patch_dim 已确定: {self.patch_dim}")

            except Exception as e:
                raise RuntimeError(
                    f"确定 num_patches 失败: {e}\n"
                    "可能的原因:\n"
                    "1. CSI数据格式不正确\n"
                    "2. 预处理器配置参数错误 (get_default_inference_config)\n"
                    "3. 数据维度与配置不匹配"
                ) from e

            self._num_patches_determined = True

        # 步骤2: 创建学生模型
        try:
            self.model = StudentModel(
                num_patches=self.num_patches,
                patch_dim=self.patch_dim,
                **self.config['student_model']
            ).to(self.device)
            print("✅ 学生模型创建成功")
        except Exception as e:
            raise RuntimeError(f"创建学生模型失败: {e}") from e

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
        # 如果是教师模型,给出提示
        if self.model_type == 'teacher':
            print("⚠️  当前为教师模型,仅支持输入骨骼数据。")
            print("   将跳过CSI预处理,假设输入为骨骼数据...")

            # 确保模型已初始化
            if not self._model_initialized:
                self._ensure_model_initialized()

            # 教师模型直接处理骨骼数据
            if not isinstance(csi_data, torch.Tensor):
                csi_data = torch.tensor(csi_data, dtype=torch.float32)
            csi_data = csi_data.to(self.device)

            with torch.no_grad():
                _, skeleton_pred, _ = self.model(csi_data)

            return skeleton_pred.cpu()

        # 学生模型处理CSI数据
        # 确保输入是4维的
        if len(csi_data.shape) == 3:
            csi_data = csi_data.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # 转换为tensor并移动到设备
        if not isinstance(csi_data, torch.Tensor):
            csi_data = torch.tensor(csi_data, dtype=torch.float32)
        csi_data = csi_data.to(self.device)

        # 确保模型已初始化
        self._ensure_model_initialized(csi_data[:1])

        with torch.no_grad():
            # 检查数据格式并转换
            if (csi_data.shape[-1] == self.csi_preprocessor.num_antennas and
                    csi_data.shape[1] != self.csi_preprocessor.num_antennas):
                csi_data = csi_data.permute(0, 3, 1, 2)

            # 预处理CSI数据
            csi_patches, _ = self.csi_preprocessor(csi_data)

            # 学生模型推理(不使用掩码)
            outputs = self.model(csi_patches, mask_ratio=0.0)
            skeleton_pred = outputs['skeleton_pred']

            # 如果输入是单个样本,移除batch维度
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
            std_skeleton: 标准差(置信度指标)
        """
        # 启用dropout进行蒙特卡洛采样
        self.model.train()

        predictions = []
        for _ in range(num_samples):
            pred = self.predict_skeleton(csi_data)
            predictions.append(pred)

        # 恢复评估模式
        self.model.eval()

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
        print(f"⚠️ 配置文件不存在: {args.config},使用默认配置")
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
    """
    测试模型 - 修复版

    修复内容:
    - ✅ 修复: 对 target_skeleton (真实数据) 应用与训练时相同的归一化
    - 自动处理 num_patches 初始化
    - 增强错误提示
    - 兼容不同数据格式
    """
    print("=" * 60)
    print("测试Enhanced Multi-Modal DMAE")
    print("=" * 60)

    # 创建推理器 (模型将在首次使用时初始化)
    try:
        inference = EnhancedDMAEInference(args.model_path, args.config)
    except Exception as e:
        print(f"❌ 创建推理器失败: {e}")
        return

    # 加载测试数据
    try:
        from mmfi_dataloader import create_enhanced_mmfi_dataloaders

        test_config_path = args.test_config or args.dataset_config
        _, test_loader = create_enhanced_mmfi_dataloaders(
            args.dataset_root, test_config_path, batch_size=1
        )

        print(f"✅ 测试数据已加载: {len(test_loader)} 个批次")

    except Exception as e:
        print(f"❌ 加载测试数据失败: {e}")
        return

    # 测试循环
    all_pred_skeletons = []
    all_target_skeletons = []

    print("\n正在进行模型测试...")

    try:
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 100:  # 限制测试样本数量
                break

            if batch_idx == 0:
                print(f"   第一个批次 - CSI形状: {batch['csi_data'].shape}")

            csi_data = batch['csi_data']
            target_skeleton_raw = batch['rgb_skeleton']
            target_skeleton_normalized = inference.skeleton_preprocessor(target_skeleton_raw)

            # 预测 (首次调用会自动初始化模型)
            try:
                pred_skeleton = inference.predict_skeleton(csi_data)
            except Exception as e:
                print(f"❌ 预测失败 (batch {batch_idx}): {e}")
                continue

            all_pred_skeletons.append(pred_skeleton)
            all_target_skeletons.append(target_skeleton_normalized)

            if (batch_idx + 1) % 20 == 0:
                print(f"   已处理: {batch_idx + 1}/{min(100, len(test_loader))} 个批次")

    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return

    if not all_pred_skeletons:
        print("❌ 没有成功预测任何样本")
        return

    # 计算指标
    try:
        pred_skeletons = torch.cat(all_pred_skeletons, dim=0)
        target_skeletons = torch.cat(all_target_skeletons, dim=0)

        metrics = calculate_skeleton_metrics(pred_skeletons, target_skeletons)

        print("\n=== 测试结果 ===")
        print(f"MPJPE: {metrics['MPJPE']:.4f}")
        for key, value in metrics.items():
            if key.startswith('PCK'):
                print(f"{key}: {value:.4f}")

    except Exception as e:
        print(f"❌ 计算指标失败: {e}")
        return

    # 可视化几个样本
    output_dir = args.output_dir or "./test_results"
    os.makedirs(output_dir, exist_ok=True)

    try:
        for i in range(min(5, len(pred_skeletons))):
            vis_path = os.path.join(output_dir, f"test_sample_{i}.png")
            visualize_skeleton_prediction(
                pred_skeletons[i], target_skeletons[i], vis_path
            )

        print(f"✅ 测试完成,结果保存到: {output_dir}")

    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")


def infer_single(args):
    """
    推理单个样本 - 修复版
    """
    print("=" * 60)
    print("单样本推理")
    print("=" * 60)

    # 创建推理器
    try:
        inference = EnhancedDMAEInference(args.model_path, args.config)
    except Exception as e:
        print(f"❌ 创建推理器失败: {e}")
        return

    # 加载CSI数据
    try:
        if args.csi_file.endswith('.mat'):
            import scipy.io as scio
            csi_data = scio.loadmat(args.csi_file)['CSIamp']
        elif args.csi_file.endswith('.npy'):
            csi_data = np.load(args.csi_file)
        else:
            raise ValueError(f"不支持的文件格式: {args.csi_file}")

        print(f"CSI数据形状: {csi_data.shape}")

    except Exception as e:
        print(f"❌ 加载CSI数据失败: {e}")
        return

    # 预测骨骼点
    try:
        pred_skeleton = inference.predict_skeleton(csi_data)
        print(f"✅ 预测骨骼点形状: {pred_skeleton.shape}")

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 可视化结果
    output_dir = args.output_dir or "./inference_results"
    os.makedirs(output_dir, exist_ok=True)

    vis_path = os.path.join(output_dir, "inference_result.png")

    try:
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

        print(f"✅ 推理完成,结果保存到: {vis_path}")

    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")

    # 保存预测结果
    try:
        result_path = os.path.join(output_dir, "predicted_skeleton.npy")
        np.save(result_path, pred_skeleton.numpy())
        print(f"✅ 预测骨骼点保存到: {result_path}")

    except Exception as e:
        print(f"⚠️ 保存失败: {e}")


def demo():
    """
    演示模式 - 修复版

    展示延迟初始化的工作流程
    """
    print("=" * 60)
    print("Enhanced Multi-Modal DMAE 演示")
    print("=" * 60)

    # 创建模拟数据
    print("\n创建模拟数据...")

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
    print(f"✅ num_patches 已确定: {csi_preprocessor.num_patches}")

    # 测试模型 (使用确定的 num_patches)
    print("\n测试模型...")
    from models import TeacherModel, StudentModel

    # ✅ 关键: 使用已确定的 num_patches 创建模型
    teacher_model = TeacherModel(embed_dim=384, depth=6)
    student_model = StudentModel(
        num_patches=csi_preprocessor.num_patches,
        patch_dim=csi_preprocessor.patch_dim,
        embed_dim=384,
        depth=6,
        num_antennas=num_antennas,
        use_multi_attn=True
    )

    print_model_info(teacher_model, "Teacher Model")
    print_model_info(student_model, "Student Model")

    # 教师模型前向传播
    teacher_loss, teacher_pred, teacher_mask = teacher_model(processed_skeleton)
    print(f"\n教师模型损失: {teacher_loss.item():.4f}")

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

    # 模拟教师特征
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

    print("\n✅ 演示完成:所有组件工作正常。")
    print("\n关键点总结:")
    print("1. ✅ CSI预处理器自动计算 num_patches")
    print("2. ✅ 使用确定的 num_patches 创建 StudentModel")
    print("3. ✅ 模型前向传播正常")
    print("4. ✅ 损失计算正常")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Modal DMAE Example")
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--dataset_root', type=str, required=True, help='MMFi数据集根目录')
    train_parser.add_argument('--dataset_config', type=str, required=True, help='数据集配置文件')
    train_parser.add_argument('--config', type=str, default='config.yaml', help='训练配置文件')
    train_parser.add_argument('--output_dir', type=str, help='输出目录')

    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    test_parser.add_argument('--dataset_root', type=str, required=True, help='测试数据集根目录')
    test_parser.add_argument('--dataset_config', type=str, required=True, help='数据集配置文件')
    test_parser.add_argument('--config', type=str, help='模型配置文件')
    test_parser.add_argument('--test_config', type=str, help='测试配置文件')
    test_parser.add_argument('--output_dir', type=str, help='输出目录')

    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='推理单个样本')
    infer_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    infer_parser.add_argument('--csi_file', type=str, required=True, help='CSI数据文件')
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