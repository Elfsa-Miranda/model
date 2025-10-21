"""
Enhanced Multi-Modal DMAE Training Pipeline - 修复版
修复内容: 解决学生模型初始化和optimizer为None的问题
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import re
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TeacherModel, StudentModel
from losses import CombinedLoss, MAELoss
from data_processing import CSIPreprocessor, SkeletonPreprocessor
from utils import (
    LossTracker, save_checkpoint, load_checkpoint,
    calculate_skeleton_metrics, visualize_skeleton_prediction,
    visualize_training_curves, save_training_log, load_training_log,
    print_model_info, set_seed, get_device, warmup_lr_scheduler
)

# 导入MMFi数据集
from mmfi_dataloader import create_enhanced_mmfi_dataloaders


def get_latest_checkpoint(checkpoint_dir, prefix="teacher"):
    """从检查点目录中找到最新的检查点文件"""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None, 0

    checkpoint_files = list(checkpoint_dir.glob(f"{prefix}_epoch_*.pth"))

    if not checkpoint_files:
        return None, 0

    epoch_pattern = re.compile(rf'{prefix}_epoch_(\d+)\.pth')
    checkpoints = []

    for ckpt_file in checkpoint_files:
        match = epoch_pattern.search(ckpt_file.name)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, ckpt_file))

    if not checkpoints:
        return None, 0

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_checkpoint = checkpoints[0]

    print(f"找到最新检查点: {latest_checkpoint.name} (已完成 Epoch {latest_epoch})")

    return str(latest_checkpoint), latest_epoch


class EnhancedDMAETrainer:
    """增强型多模态DMAE训练器"""

    def __init__(self, config, resume_teacher=None, resume_student=None):
        self.config = config
        self.device = get_device()
        set_seed(config.get('seed', 42))

        # 初始化数据预处理器
        self.csi_preprocessor = CSIPreprocessor(**config['csi_preprocessor'])
        self.skeleton_preprocessor = SkeletonPreprocessor(**config['skeleton_preprocessor'])

        # 初始化教师模型
        self.teacher_model = TeacherModel(**config['teacher_model']).to(self.device)

        # 学生模型延迟初始化
        self.student_model = None
        self.student_model_config = config['student_model']
        print("⚠️  学生模型将在第一次使用时初始化（等待 num_patches 确定）")

        # 打印模型信息
        print_model_info(self.teacher_model, "Teacher Model")

        # 初始化损失函数
        self.teacher_loss_fn = MAELoss(**config.get('teacher_loss', {}))
        self.combined_loss_fn = CombinedLoss(**config.get('combined_loss', {}))

        # 初始化优化器
        self.teacher_optimizer = self._create_optimizer(
            self.teacher_model, config.get('teacher_optimizer', {})
        )
        self.student_optimizer = None
        self.student_optimizer_config = config.get('student_optimizer', {})

        # 初始化学习率调度器
        self.teacher_scheduler = self._create_scheduler(
            self.teacher_optimizer, config.get('teacher_scheduler', {})
        )
        self.student_scheduler = None
        self.student_scheduler_config = config.get('student_scheduler', {})

        # 训练状态
        self.teacher_start_epoch = 0
        self.student_start_epoch = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.loss_tracker = LossTracker()

        # 输出目录
        self.output_dir = config.get('output_dir', './outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        # 日志文件
        self.log_file = os.path.join(self.output_dir, 'training_log.json')
        self.training_log = load_training_log(self.log_file)

        # TensorBoard设置
        self.use_tensorboard = config.get('logging', {}).get('tensorboard', False)
        if self.use_tensorboard:
            self.tb_log_dir = os.path.join(self.output_dir, 'tensorboard_logs')
            os.makedirs(self.tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_log_dir)
            print(f"✅ TensorBoard日志目录: {self.tb_log_dir}")
        else:
            self.writer = None

        # 恢复训练
        if resume_teacher:
            self._resume_teacher_training(resume_teacher)

        if resume_student:
            self._resume_student_training(resume_student)

    def _initialize_student_model(self):
        """初始化学生模型（在第一次使用时调用）"""
        if self.student_model is not None:
            return

        if self.csi_preprocessor.num_patches is None:
            raise ValueError("必须先运行一次CSI预处理器来确定 num_patches")

        print("\n" + "=" * 60)
        print("🔧 正在初始化学生模型...")
        print("=" * 60)

        self.student_model = StudentModel(
            num_patches=self.csi_preprocessor.num_patches,
            patch_dim=self.csi_preprocessor.patch_dim,
            **self.student_model_config
        ).to(self.device)

        print_model_info(self.student_model, "Student Model")

        # 创建优化器
        self.student_optimizer = self._create_optimizer(
            self.student_model, self.student_optimizer_config
        )

        # 创建学习率调度器
        self.student_scheduler = self._create_scheduler(
            self.student_optimizer, self.student_scheduler_config
        )

        print("✅ 学生模型初始化完成")
        print("=" * 60 + "\n")

    def _resume_teacher_training(self, checkpoint_path):
        """恢复教师模型训练"""
        if os.path.exists(checkpoint_path):
            print(f"\n正在恢复教师模型训练...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
            self.teacher_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint and self.teacher_scheduler:
                self.teacher_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.teacher_start_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('loss', float('inf'))

            print(f"✅ 教师模型已恢复: 已完成 Epoch {self.teacher_start_epoch}")
            print(f"   将从 Epoch {self.teacher_start_epoch + 1} 继续训练")
        else:
            print(f"❌ 检查点文件不存在: {checkpoint_path}")

    def _resume_student_training(self, checkpoint_path):
        """恢复学生模型训练"""
        if os.path.exists(checkpoint_path):
            print(f"\n正在恢复学生模型训练...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if self.student_model is None:
                print("⚠️  学生模型尚未初始化，正在从检查点恢复...")
                self.student_start_epoch = checkpoint.get('epoch', 0)
                self.best_loss = checkpoint.get('loss', float('inf'))
                print(f"✅ 学生模型检查点信息已记录: 已完成 Epoch {self.student_start_epoch}")
                print(f"   模型将在训练开始时加载")
                return

            self.student_model.load_state_dict(checkpoint['model_state_dict'])
            self.student_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint and self.student_scheduler:
                self.student_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.student_start_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('loss', float('inf'))

            print(f"✅ 学生模型已恢复: 已完成 Epoch {self.student_start_epoch}")
            print(f"   将从 Epoch {self.student_start_epoch + 1} 继续训练")
        else:
            print(f"❌ 检查点文件不存在: {checkpoint_path}")

    def _create_optimizer(self, model, optimizer_config):
        """创建优化器"""
        optimizer_type = optimizer_config.get('type', 'adamw')
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-2)

        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        return optimizer

    def _create_scheduler(self, optimizer, scheduler_config):
        """创建学习率调度器"""
        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type.lower() == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_type.lower() == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'warmup':
            warmup_epochs = scheduler_config.get('warmup_epochs', 10)
            total_epochs = scheduler_config.get('total_epochs', 100)
            base_lr = scheduler_config.get('base_lr', 1e-4)
            scheduler = warmup_lr_scheduler(optimizer, warmup_epochs, total_epochs, base_lr)
        else:
            scheduler = None

        return scheduler

    def load_data(self, dataset_root, config_file):
        """加载数据"""
        print("正在加载数据...")

        self.train_loader, self.val_loader = create_enhanced_mmfi_dataloaders(
            dataset_root,
            config_file,
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 4)
        )

        print(f"✅ 数据加载完成")
        print(f"   训练批次数: {len(self.train_loader)}")
        print(f"   验证批次数: {len(self.val_loader)}")

        print("\n验证数据格式...")
        try:
            sample_batch = next(iter(self.train_loader))
            print(f"   批次键: {sample_batch.keys()}")
            print(f"   CSI形状: {sample_batch['csi_data'].shape}")
            print(f"   骨骼点形状: {sample_batch['rgb_skeleton'].shape}")
            print("✅ 数据格式验证通过")
        except Exception as e:
            print(f"❌ 数据格式验证失败: {e}")
            raise

    def train_student_distillation(self, num_epochs):
        """阶段2: 学生模型蒸馏训练"""
        print("\n" + "=" * 60)
        print("阶段2: 学生模型蒸馏训练")
        print("=" * 60)

        # ✅ 关键修复：确保学生模型已初始化
        if self.student_model is None:
            print("🔍 学生模型尚未初始化，正在初始化...")

            if not hasattr(self, 'train_loader') or self.train_loader is None:
                raise RuntimeError("数据加载器未初始化！请先调用 load_data() 方法。")

            try:
                print("   正在获取样本批次...")
                sample_batch = next(iter(self.train_loader))
                csi_sample = sample_batch['csi_data'].to(self.device)
                print(f"   CSI样本形状: {csi_sample.shape}")

                # 检测数据格式并调整
                if csi_sample.shape[1] == 3:
                    csi_sample_reshaped = csi_sample[:1]
                    print(f"   检测到标准格式: {csi_sample_reshaped.shape}")
                elif csi_sample.shape[-1] == 3:
                    csi_sample_reshaped = csi_sample.permute(0, 3, 1, 2)[:1]
                    print(f"   检测到MMFi格式，已转换: {csi_sample_reshaped.shape}")
                else:
                    raise ValueError(f"无法识别的CSI数据格式: {csi_sample.shape}")

                print("   运行CSI预处理器...")
                patches, _ = self.csi_preprocessor(csi_sample_reshaped)

                if self.csi_preprocessor.num_patches is None:
                    raise RuntimeError(f"CSI预处理器未能确定 num_patches！patches shape: {patches.shape}")

                print(f"   ✅ num_patches 已确定: {self.csi_preprocessor.num_patches}")

                # 初始化学生模型
                self._initialize_student_model()
                print("   ✅ 学生模型初始化完成")

                # === GPT5 FIX START: 记录预期patch数，确保一致 ===
                self.student_model.init_grid = self.csi_preprocessor.patch_grid
                self.student_model.expected_grid = self.csi_preprocessor.patch_grid
                self.expected_num_patches = self.csi_preprocessor.num_patches

                print(f"✅ Student expected patch grid: {self.expected_num_patches} "
                      f"({self.csi_preprocessor.patch_grid[0]}×{self.csi_preprocessor.patch_grid[1]}×{self.csi_preprocessor.num_antennas})")
                # === GPT5 FIX END ===



            except StopIteration:
                raise RuntimeError("数据加载器为空！请检查数据集是否正确加载。")
            except Exception as e:
                print(f"   ❌ 初始化失败: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"学生模型初始化失败: {e}\n"
                    "可能的原因:\n"
                    "1. CSI数据格式不匹配\n"
                    "2. config.yaml 中的 csi_preprocessor 参数配置错误\n"
                    "3. 数据加载器返回的数据格式异常"
                ) from e

        # 冻结教师模型
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.student_model.train()

        # 从恢复的epoch开始
        start_epoch = self.student_start_epoch
        total_epochs = start_epoch + num_epochs

        print(f"训练范围: Epoch {start_epoch + 1} 到 Epoch {total_epochs}")

        for epoch in range(start_epoch, total_epochs):
            actual_epoch = epoch + 1

            epoch_start_time = time.time()
            self.loss_tracker.reset_current()

            # 训练循环
            train_bar = tqdm(
                self.train_loader,
                desc=f'Epoch {actual_epoch}/{total_epochs}',
                ncols=120,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )

            for batch_idx, batch in enumerate(train_bar):
                # 获取数据
                csi_data = batch['csi_data'].to(self.device)
                rgb_skeleton = batch['rgb_skeleton'].to(self.device)

                # 预处理CSI数据
                batch_size = csi_data.shape[0]
                csi_data_reshaped = csi_data.permute(0, 3, 1, 2)
                csi_patches, _ = self.csi_preprocessor(csi_data_reshaped)

                # 预处理RGB骨骼点
                rgb_skeleton = self.skeleton_preprocessor(rgb_skeleton)

                # 对比学习标签
                contrast_labels = torch.ones(batch_size, device=self.device)

                # 学生模型前向传播
                self.student_optimizer.zero_grad()
                student_outputs = self.student_model(csi_patches)

                # 教师模型前向传播
                with torch.no_grad():
                    teacher_features = self.teacher_model.forward_features(rgb_skeleton, mask_ratio=0.0)

                # 计算组合损失
                total_loss, loss_dict = self.combined_loss_fn(
                    student_outputs['reconstructed_patches'], csi_patches, student_outputs['mask'],
                    student_outputs['skeleton_pred'], rgb_skeleton,
                    student_outputs['distill_features'], teacher_features,
                    student_outputs['contrast_features'][:batch_size//2] if batch_size > 1 else student_outputs['contrast_features'][:1],
                    student_outputs['contrast_features'][batch_size//2:] if batch_size > 1 else student_outputs['contrast_features'][:1],
                    contrast_labels[:batch_size//2] if batch_size > 1 else contrast_labels[:1]
                )

                # 反向传播
                total_loss.backward()
                self.student_optimizer.step()

                # 更新损失
                self.loss_tracker.update(loss_dict, csi_patches.shape[0])

                # 更新进度条
                current_losses = self.loss_tracker.get_current_losses()
                train_bar.set_postfix({
                    'Loss': f"{current_losses.get('total_loss', 0):.4f}",
                    'MAE': f"{current_losses.get('mae_total_loss', 0):.4f}",
                    'Distill': f"{current_losses.get('distill_loss', 0):.4f}",
                    'Contrast': f"{current_losses.get('contrast_loss', 0):.4f}"
                })

            # 学习率调度
            if self.student_scheduler:
                self.student_scheduler.step()

            # 验证
            val_loss, val_metrics = self.validate_student()

            # 保存epoch损失
            self.loss_tracker.current_losses['student_val_loss'].update(val_loss)
            # 保存epoch损失
            self.loss_tracker.current_losses['student_val_loss'].update(val_loss)

            # 将 val_metrics 中的每项转换为可供 AverageMeter.update() 接受的单一标量（取均值）
            for key, value in val_metrics.items():
                # 默认跳过不能处理的类型
                scalar_value = None

                # list -> 取均值（适用于 joint_errors 列表）
                if isinstance(value, (list, tuple)):
                    try:
                        scalar_value = float(np.mean(value))
                    except Exception:
                        # 如果列表内含非数值，跳过记录
                        continue

                # torch.Tensor -> 标量或均值
                elif isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        scalar_value = float(value.item())
                    else:
                        scalar_value = float(value.mean().item())

                # 直接是数值
                elif isinstance(value, (float, int, np.floating, np.integer)):
                    scalar_value = float(value)

                # 其他类型跳过
                else:
                    continue

                # 更新到 loss_tracker（AverageMeter 期望一个数值）
                self.loss_tracker.current_losses[f'val_{key}'].update(scalar_value)

            # （可选）把 PCK 单独写入 TensorBoard（原来逻辑）
            for key, value in val_metrics.items():
                if self.writer and key.startswith('PCK@'):
                    # 确保传入的是标量
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            v = float(value.item())
                        else:
                            v = float(value.mean().item())
                    elif isinstance(value, (list, tuple)):
                        v = float(np.mean(value))
                    else:
                        v = float(value)
                    self.writer.add_scalar(f'Student/Val_{key}', v, actual_epoch)

            # TensorBoard记录
            if self.writer:
                self.writer.add_scalar('Student/Train_Total_Loss', current_losses.get('total_loss', 0), actual_epoch)
                self.writer.add_scalar('Student/Train_MAE_Loss', current_losses.get('mae_total_loss', 0), actual_epoch)
                self.writer.add_scalar('Student/Train_Distill_Loss', current_losses.get('distill_loss', 0), actual_epoch)
                self.writer.add_scalar('Student/Train_Contrast_Loss', current_losses.get('contrast_loss', 0), actual_epoch)
                self.writer.add_scalar('Student/Val_Loss', val_loss, actual_epoch)
                self.writer.add_scalar('Student/Val_MPJPE', val_metrics.get('MPJPE', 0), actual_epoch)
                self.writer.add_scalar('Student/Learning_Rate', self.student_optimizer.param_groups[0]['lr'], actual_epoch)

                for key, value in val_metrics.items():
                    if key.startswith('PCK@'):
                        self.writer.add_scalar(f'Student/Val_{key}', value, actual_epoch)

            # 保存检查点
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            if actual_epoch % self.config.get('save_freq', 10) == 0 or is_best:
                save_checkpoint(
                    self.student_model, self.student_optimizer, self.student_scheduler,
                    actual_epoch,
                    val_loss,
                    os.path.join(self.output_dir, 'student_checkpoints'),
                    f'student_epoch_{actual_epoch}.pth',
                    is_best
                )

            # 打印epoch信息
            best_marker = " 🎯 Best!" if is_best else ""
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {actual_epoch}/{total_epochs} - "
                  f"Train Loss: {current_losses.get('total_loss', 0):.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val MPJPE: {val_metrics.get('MPJPE', 0):.4f}, "
                  f"Time: {epoch_time:.2f}s{best_marker}")

        # 更新起始epoch
        self.student_start_epoch = total_epochs

        print("✅ 学生模型蒸馏训练完成")

    def validate_student(self):
        """验证学生模型"""
        self.student_model.eval()
        total_loss = 0.0
        all_pred_skeletons = []
        all_target_skeletons = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                csi_data = batch['csi_data'].to(self.device)
                rgb_skeleton = batch['rgb_skeleton'].to(self.device)

                # 预处理
                csi_data_reshaped = csi_data.permute(0, 3, 1, 2)
                csi_patches, _ = self.csi_preprocessor(csi_data_reshaped)
                rgb_skeleton = self.skeleton_preprocessor(rgb_skeleton)

                # 学生模型推理
                student_outputs = self.student_model(csi_patches, mask_ratio=0.0)

                # 计算损失
                skeleton_loss = nn.MSELoss()(student_outputs['skeleton_pred'], rgb_skeleton)
                total_loss += skeleton_loss.item()

                # 收集预测结果
                all_pred_skeletons.append(student_outputs['skeleton_pred'].cpu())
                all_target_skeletons.append(rgb_skeleton.cpu())

                num_batches += 1

        self.student_model.train()

        # 计算骨骼点指标
        if all_pred_skeletons:
            pred_skeletons = torch.cat(all_pred_skeletons, dim=0)
            target_skeletons = torch.cat(all_target_skeletons, dim=0)
            metrics = calculate_skeleton_metrics(pred_skeletons, target_skeletons)

            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = value.item()
        else:
            metrics = {'MPJPE': 0.0}

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, metrics

    def save_training_results(self):
        """保存训练结果"""
        loss_history = self.loss_tracker.get_history()

        curves_path = os.path.join(self.output_dir, 'training_curves.png')
        visualize_training_curves(loss_history, curves_path)

        self.training_log.update({
            'loss_history': loss_history,
            'config': self.config,
            'best_loss': self.best_loss,
            'total_epochs': self.current_epoch
        })
        save_training_log(self.training_log, self.log_file)

        print(f"✅ 训练结果已保存到: {self.output_dir}")

    def train(self, dataset_root, config_file):
        """完整训练流程"""
        print("=" * 60)
        print("Enhanced Multi-Modal DMAE 训练开始")
        print("=" * 60)

        # 加载数据
        self.load_data(dataset_root, config_file)

        # 阶段1: 教师模型预训练
        teacher_epochs = self.config.get('teacher_pretrain_epochs', 50)
        if teacher_epochs > 0:
            print("⚠️  跳过教师模型预训练 (epochs=0)")

        # 阶段2: 学生模型蒸馏训练
        student_epochs = self.config.get('student_distill_epochs', 100)
        if student_epochs > 0:
            self.train_student_distillation(student_epochs)

        # 保存训练结果
        self.save_training_results()

        # 关闭TensorBoard writer
        if self.writer:
            self.writer.close()

        print("\n" + "=" * 60)
        print("🎉 Enhanced Multi-Modal DMAE 训练完成!")
        print(f"最佳验证损失: {self.best_loss:.4f}")
        print(f"输出目录: {self.output_dir}")
        if self.use_tensorboard:
            print(f"TensorBoard日志: {self.tb_log_dir}")
            print("启动TensorBoard: tensorboard --logdir=" + self.tb_log_dir)
        print("=" * 60)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Enhanced Multi-Modal DMAE Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从头开始训练
  python train.py <dataset_root> <config_file> --config config.yaml --output_dir ./outputs

  # 恢复教师模型训练
  python train.py <dataset_root> <config_file> --config config.yaml --resume_teacher ./outputs/teacher_checkpoints/best_model.pth

  # 恢复学生模型训练
  python train.py <dataset_root> <config_file> --config config.yaml --resume_student ./outputs/student_checkpoints/best_model.pth
        """
    )

    parser.add_argument('dataset_root', type=str, help='数据集根目录路径')
    parser.add_argument('config_file', type=str, help='数据集配置文件路径 (config.yaml)')
    parser.add_argument('--config', type=str, default='config.yaml', help='训练配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--resume_teacher', type=str, default=None, help='教师模型检查点路径')
    parser.add_argument('--resume_student', type=str, default=None, help='学生模型检查点路径')
    parser.add_argument('--auto_resume', action='store_true', help='自动恢复最新检查点')

    args = parser.parse_args()

    print("=" * 80)
    print("Enhanced Multi-Modal DMAE Training Pipeline")
    print("=" * 80)
    print(f"数据集根目录: {args.dataset_root}")
    print(f"数据集配置: {args.config_file}")
    print(f"训练配置: {args.config}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    # 验证路径
    if not os.path.exists(args.dataset_root):
        print(f"❌ 错误: 数据集根目录不存在: {args.dataset_root}")
        return

    if not os.path.exists(args.config_file):
        print(f"❌ 错误: 数据集配置文件不存在: {args.config_file}")
        return

    if not os.path.exists(args.config):
        print(f"❌ 错误: 训练配置文件不存在: {args.config}")
        return

    # 加载训练配置
    try:
        config = load_config(args.config)
        config['output_dir'] = args.output_dir
        print("✅ 训练配置加载成功")
    except Exception as e:
        print(f"❌ 错误: 无法加载训练配置: {e}")
        return

    # 自动恢复检查点
    resume_teacher = args.resume_teacher
    resume_student = args.resume_student

    if args.auto_resume:
        print("\n检查是否存在检查点...")
        teacher_checkpoint_dir = os.path.join(args.output_dir, 'teacher_checkpoints')
        student_checkpoint_dir = os.path.join(args.output_dir, 'student_checkpoints')

        if resume_teacher is None:
            latest_teacher, _ = get_latest_checkpoint(teacher_checkpoint_dir, "teacher")
            if latest_teacher:
                resume_teacher = latest_teacher
                print(f"✅ 找到教师模型检查点: {resume_teacher}")

        if resume_student is None:
            latest_student, _ = get_latest_checkpoint(student_checkpoint_dir, "student")
            if latest_student:
                resume_student = latest_student
                print(f"✅ 找到学生模型检查点: {resume_student}")

    # 创建训练器
    try:
        print("\n初始化训练器...")
        trainer = EnhancedDMAETrainer(
            config,
            resume_teacher=resume_teacher,
            resume_student=resume_student
        )
        print("✅ 训练器初始化成功")
    except Exception as e:
        print(f"❌ 错误: 训练器初始化失败")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        return

    # 开始训练
    try:
        print("\n开始训练流程...")
        trainer.train(args.dataset_root, args.config_file)
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()