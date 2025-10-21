"""
Enhanced Multi-Modal DMAE Utilities
增强型多模态DMAE工具函数

包含:
1. 训练工具函数
2. 评估工具函数
3. 可视化工具函数
4. 模型保存/加载工具
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import time


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossTracker:
    """损失跟踪器"""
    
    def __init__(self):
        self.losses = defaultdict(list)
        self.current_losses = defaultdict(AverageMeter)
    
    def update(self, loss_dict, batch_size=1):
        """更新损失"""
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.current_losses[key].update(value, batch_size)
    
    def get_current_losses(self):
        """获取当前epoch的平均损失"""
        return {key: meter.avg for key, meter in self.current_losses.items()}
    
    def save_epoch(self):
        """保存当前epoch的损失并重置"""
        for key, meter in self.current_losses.items():
            self.losses[key].append(meter.avg)
        self.reset_current()
    
    def reset_current(self):
        """重置当前epoch的损失"""
        for meter in self.current_losses.values():
            meter.reset()
    
    def get_history(self):
        """获取历史损失"""
        return dict(self.losses)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, 
                   filename=None, is_best=False):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        loss: 当前损失
        checkpoint_dir: 检查点目录
        filename: 文件名，如果为None则自动生成
        is_best: 是否为最佳模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    # 计算模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time(),
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'model_class': model.__class__.__name__
        }
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✅ 最佳模型已保存: {best_path}")
    
    print(f"✅ 检查点已保存: {checkpoint_path}")
    print(f"   模型参数: {total_params:,} 个 (可训练: {trainable_params:,})")
    print(f"   模型大小: {model_size_mb:.2f} MB")
    print(f"   损失值: {loss:.4f}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
    
    Returns:
        epoch: 加载的epoch
        loss: 加载的损失
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"✅ 检查点已加载: {checkpoint_path}")
    print(f"   Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch, loss


def export_model_to_onnx(model, sample_input, output_path, model_name="model"):
    """
    导出模型为ONNX格式
    
    Args:
        model: 要导出的模型
        sample_input: 样本输入数据
        output_path: 输出路径
        model_name: 模型名称
    """
    try:
        import torch.onnx
        
        # 设置模型为评估模式
        model.eval()
        
        # 导出ONNX
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ 模型已导出为ONNX: {output_path}")
        return True
        
    except ImportError:
        print("❌ 错误: 需要安装onnx库: pip install onnx")
        return False
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return False


def save_model_weights_only(model, output_path, model_name="model"):
    """
    仅保存模型权重（不包含优化器等状态）
    
    Args:
        model: 模型
        output_path: 输出路径
        model_name: 模型名称
    """
    weights = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': time.time()
    }
    
    torch.save(weights, output_path)
    
    # 计算文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"✅ 模型权重已保存: {output_path}")
    print(f"   文件大小: {file_size_mb:.2f} MB")


def verify_checkpoint(checkpoint_path):
    """
    验证检查点文件的完整性
    
    Args:
        checkpoint_path: 检查点路径
    
    Returns:
        dict: 检查点信息
    """
    if not os.path.exists(checkpoint_path):
        return {'valid': False, 'error': '文件不存在'}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查必要字段
        required_fields = ['model_state_dict', 'epoch', 'loss']
        missing_fields = [field for field in required_fields if field not in checkpoint]
        
        if missing_fields:
            return {'valid': False, 'error': f'缺少字段: {missing_fields}'}
        
        # 检查模型状态字典
        model_state = checkpoint['model_state_dict']
        if not isinstance(model_state, dict):
            return {'valid': False, 'error': 'model_state_dict格式错误'}
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
        
        info = {
            'valid': True,
            'epoch': checkpoint.get('epoch', 'unknown'),
            'loss': checkpoint.get('loss', 'unknown'),
            'total_params': total_params,
            'model_class': checkpoint.get('model_info', {}).get('model_class', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
        }
        
        return info
        
    except Exception as e:
        return {'valid': False, 'error': f'加载失败: {str(e)}'}


def list_checkpoints(checkpoint_dir):
    """
    列出检查点目录中的所有模型文件
    
    Args:
        checkpoint_dir: 检查点目录
    
    Returns:
        list: 检查点文件列表
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, filename)
            info = verify_checkpoint(filepath)
            checkpoints.append({
                'filename': filename,
                'filepath': filepath,
                'info': info
            })
    
    # 按epoch排序
    checkpoints.sort(key=lambda x: x['info'].get('epoch', 0) if x['info']['valid'] else -1)
    
    return checkpoints


def calculate_skeleton_metrics(pred_skeleton, target_skeleton, valid_mask=None):
    """
    计算骨骼点预测指标
    
    Args:
        pred_skeleton: 预测骨骼点 [batch, num_joints, coord_dim]
        target_skeleton: 目标骨骼点 [batch, num_joints, coord_dim]
        valid_mask: 有效关节掩码 [batch, num_joints]
    
    Returns:
        metrics: 指标字典
    """
    # 计算欧几里得距离误差
    distances = torch.norm(pred_skeleton - target_skeleton, dim=-1)  # [batch, num_joints]
    
    if valid_mask is not None:
        distances = distances * valid_mask
        num_valid = valid_mask.sum()
    else:
        num_valid = distances.numel()
    
    # MPJPE (Mean Per Joint Position Error)
    mpjpe = distances.sum() / num_valid if num_valid > 0 else torch.tensor(0.0)
    
    # 每个关节的平均误差
    if valid_mask is not None:
        joint_errors = distances.sum(dim=0) / valid_mask.sum(dim=0).clamp(min=1)
    else:
        joint_errors = distances.mean(dim=0)
    
    # PCK (Percentage of Correct Keypoints) @ different thresholds
    pck_thresholds = [0.05, 0.1, 0.2, 0.5]  # 相对于图像尺寸的阈值
    pck_scores = {}
    
    for threshold in pck_thresholds:
        # 假设图像尺寸为1.0（归一化坐标）
        correct = (distances < threshold).float()
        if valid_mask is not None:
            correct = correct * valid_mask
            pck = correct.sum() / num_valid if num_valid > 0 else torch.tensor(0.0)
        else:
            pck = correct.mean()
        pck_scores[f'PCK@{threshold}'] = pck
    
    metrics = {
        'MPJPE': mpjpe,
        'joint_errors': joint_errors,
        **pck_scores
    }
    
    return metrics


def visualize_skeleton_prediction(pred_skeleton, target_skeleton, save_path=None, 
                                joint_connections=None):
    """
    可视化骨骼点预测结果
    
    Args:
        pred_skeleton: 预测骨骼点 [num_joints, 2]
        target_skeleton: 目标骨骼点 [num_joints, 2]
        save_path: 保存路径
        joint_connections: 关节连接关系列表
    """
    # COCO-17关节连接关系
    if joint_connections is None:
        joint_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上身
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 下身
        ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 转换为numpy
    if isinstance(pred_skeleton, torch.Tensor):
        pred_skeleton = pred_skeleton.detach().cpu().numpy()
    if isinstance(target_skeleton, torch.Tensor):
        target_skeleton = target_skeleton.detach().cpu().numpy()
    
    # 绘制目标骨骼点
    ax1.scatter(target_skeleton[:, 0], target_skeleton[:, 1], c='blue', s=50, alpha=0.7, label='Target')
    for connection in joint_connections:
        if connection[0] < len(target_skeleton) and connection[1] < len(target_skeleton):
            x_coords = [target_skeleton[connection[0], 0], target_skeleton[connection[1], 0]]
            y_coords = [target_skeleton[connection[0], 1], target_skeleton[connection[1], 1]]
            ax1.plot(x_coords, y_coords, 'b-', alpha=0.5)
    
    ax1.set_title('Target Skeleton')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # 图像坐标系
    ax1.legend()
    
    # 绘制预测骨骼点
    ax2.scatter(pred_skeleton[:, 0], pred_skeleton[:, 1], c='red', s=50, alpha=0.7, label='Prediction')
    for connection in joint_connections:
        if connection[0] < len(pred_skeleton) and connection[1] < len(pred_skeleton):
            x_coords = [pred_skeleton[connection[0], 0], pred_skeleton[connection[1], 0]]
            y_coords = [pred_skeleton[connection[0], 1], pred_skeleton[connection[1], 1]]
            ax2.plot(x_coords, y_coords, 'r-', alpha=0.5)
    
    ax2.set_title('Predicted Skeleton')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # 图像坐标系
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_training_curves(loss_history, save_path=None):
    """
    可视化训练曲线
    
    Args:
        loss_history: 损失历史字典
        save_path: 保存路径
    """
    num_losses = len(loss_history)
    if num_losses == 0:
        return
    
    # 计算子图布局
    cols = min(3, num_losses)
    rows = (num_losses + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # 绘制每个损失曲线
    for i, (loss_name, loss_values) in enumerate(loss_history.items()):
        if i < len(axes):
            axes[i].plot(loss_values, label=loss_name)
            axes[i].set_title(f'{loss_name}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
    
    # 隐藏多余的子图
    for i in range(num_losses, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_training_log(log_data, log_path):
    """
    保存训练日志
    
    Args:
        log_data: 日志数据字典
        log_path: 日志文件路径
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 转换tensor为float
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    log_data = convert_tensors(log_data)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练日志已保存: {log_path}")


def load_training_log(log_path):
    """
    加载训练日志
    
    Args:
        log_path: 日志文件路径
    
    Returns:
        log_data: 日志数据字典
    """
    if not os.path.exists(log_path):
        return {}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    return log_data


def get_parameter_count(model):
    """
    获取模型参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        param_count: 参数数量字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_count = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }
    
    return param_count


def print_model_info(model, model_name="Model"):
    """
    打印模型信息
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    param_count = get_parameter_count(model)
    
    print(f"\n=== {model_name} 信息 ===")
    print(f"总参数数量: {param_count['total_params']:,}")
    print(f"可训练参数数量: {param_count['trainable_params']:,}")
    print(f"不可训练参数数量: {param_count['non_trainable_params']:,}")
    
    # 计算模型大小（MB）
    param_size = param_count['total_params'] * 4 / (1024 * 1024)  # 假设float32
    print(f"模型大小: {param_size:.2f} MB")


def set_seed(seed=42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("⚠️  使用CPU")
    
    return device


def warmup_lr_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, warmup_lr=1e-6):
    """
    创建带预热的学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: 预热epoch数
        total_epochs: 总epoch数
        base_lr: 基础学习率
        warmup_lr: 预热起始学习率
    
    Returns:
        scheduler: 学习率调度器
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 预热阶段：线性增长
            return (base_lr - warmup_lr) * epoch / warmup_epochs + warmup_lr
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def test_utils():
    """测试工具函数"""
    print("=" * 50)
    print("测试工具函数")
    print("=" * 50)
    
    # 测试损失跟踪器
    tracker = LossTracker()
    
    # 模拟几个epoch的损失
    for epoch in range(3):
        for batch in range(5):
            loss_dict = {
                'total_loss': torch.tensor(1.0 - epoch * 0.1 + np.random.normal(0, 0.1)),
                'mae_loss': torch.tensor(0.5 - epoch * 0.05 + np.random.normal(0, 0.05)),
                'distill_loss': torch.tensor(0.3 - epoch * 0.03 + np.random.normal(0, 0.03)),
            }
            tracker.update(loss_dict)
        
        current_losses = tracker.get_current_losses()
        print(f"Epoch {epoch}: {current_losses}")
        tracker.save_epoch()
    
    # 测试骨骼点指标计算
    pred_skeleton = torch.randn(4, 17, 2)
    target_skeleton = torch.randn(4, 17, 2)
    metrics = calculate_skeleton_metrics(pred_skeleton, target_skeleton)
    print(f"\n骨骼点指标: MPJPE = {metrics['MPJPE']:.4f}")
    
    # 测试参数计数
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    print_model_info(model, "测试模型")
    
    print("✅ 工具函数测试通过")


if __name__ == "__main__":
    test_utils()

