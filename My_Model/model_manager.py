#!/usr/bin/env python3
"""
模型管理脚本
用于管理、导出和验证训练好的模型

使用方法:
python model_manager.py list [checkpoint_dir]           # 列出所有检查点
python model_manager.py info <checkpoint_path>          # 查看检查点信息
python model_manager.py export <checkpoint_path> <output_path>  # 导出模型
python model_manager.py weights <checkpoint_path> <output_path> # 仅保存权重
python model_manager.py onnx <checkpoint_path> <output_path>    # 导出ONNX
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    verify_checkpoint, list_checkpoints, save_model_weights_only, 
    export_model_to_onnx, load_checkpoint
)
from models import TeacherModel, StudentModel
from data_processing import CSIPreprocessor, SkeletonPreprocessor


def list_all_checkpoints(checkpoint_dir):
    """列出所有检查点"""
    print("=" * 60)
    print("检查点列表")
    print("=" * 60)
    
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"❌ 在 {checkpoint_dir} 中没有找到检查点文件")
        return
    
    print(f"找到 {len(checkpoints)} 个检查点文件:\n")
    
    for i, checkpoint in enumerate(checkpoints, 1):
        info = checkpoint['info']
        filename = checkpoint['filename']
        
        if info['valid']:
            print(f"{i:2d}. {filename}")
            print(f"    Epoch: {info['epoch']}")
            print(f"    Loss: {info['loss']:.4f}")
            print(f"    参数数量: {info['total_params']:,}")
            print(f"    文件大小: {info['file_size_mb']:.2f} MB")
            print(f"    模型类型: {info['model_class']}")
            if info['timestamp'] != 'unknown':
                timestamp = datetime.fromtimestamp(info['timestamp'])
                print(f"    保存时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        else:
            print(f"{i:2d}. {filename} ❌ 无效: {info['error']}")
            print()


def show_checkpoint_info(checkpoint_path):
    """显示检查点详细信息"""
    print("=" * 60)
    print("检查点详细信息")
    print("=" * 60)
    
    info = verify_checkpoint(checkpoint_path)
    
    if not info['valid']:
        print(f"❌ 检查点无效: {info['error']}")
        return
    
    print(f"文件路径: {checkpoint_path}")
    print(f"文件名: {os.path.basename(checkpoint_path)}")
    print(f"Epoch: {info['epoch']}")
    print(f"Loss: {info['loss']:.4f}")
    print(f"参数数量: {info['total_params']:,}")
    print(f"文件大小: {info['file_size_mb']:.2f} MB")
    print(f"模型类型: {info['model_class']}")
    
    if info['timestamp'] != 'unknown':
        timestamp = datetime.fromtimestamp(info['timestamp'])
        print(f"保存时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载检查点获取更多信息
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n检查点内容:")
        for key, value in checkpoint.items():
            if key == 'model_state_dict':
                print(f"  {key}: 模型状态字典 ({len(value)} 个参数)")
            elif key == 'optimizer_state_dict':
                print(f"  {key}: 优化器状态字典")
            elif key == 'scheduler_state_dict':
                print(f"  {key}: 调度器状态字典")
            elif key == 'model_info':
                print(f"  {key}: 模型信息")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")


def export_model(checkpoint_path, output_path):
    """导出完整模型"""
    print("=" * 60)
    print("导出模型")
    print("=" * 60)
    
    # 验证检查点
    info = verify_checkpoint(checkpoint_path)
    if not info['valid']:
        print(f"❌ 检查点无效: {info['error']}")
        return False
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存完整检查点
        torch.save(checkpoint, output_path)
        
        # 计算文件大小
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✅ 模型已导出: {output_path}")
        print(f"   文件大小: {file_size_mb:.2f} MB")
        print(f"   包含内容: 模型参数 + 优化器状态 + 调度器状态")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False


def export_weights_only(checkpoint_path, output_path):
    """仅导出模型权重"""
    print("=" * 60)
    print("导出模型权重")
    print("=" * 60)
    
    # 验证检查点
    info = verify_checkpoint(checkpoint_path)
    if not info['valid']:
        print(f"❌ 检查点无效: {info['error']}")
        return False
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 仅保存模型权重
        weights = {
            'model_state_dict': checkpoint['model_state_dict'],
            'model_class': checkpoint.get('model_info', {}).get('model_class', 'unknown'),
            'timestamp': time.time()
        }
        
        torch.save(weights, output_path)
        
        # 计算文件大小
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✅ 模型权重已导出: {output_path}")
        print(f"   文件大小: {file_size_mb:.2f} MB")
        print(f"   包含内容: 仅模型参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False


def export_to_onnx(checkpoint_path, output_path):
    """导出为ONNX格式"""
    print("=" * 60)
    print("导出为ONNX格式")
    print("=" * 60)
    
    # 验证检查点
    info = verify_checkpoint(checkpoint_path)
    if not info['valid']:
        print(f"❌ 检查点无效: {info['error']}")
        return False
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 根据模型类型创建模型实例
        model_class = checkpoint.get('model_info', {}).get('model_class', 'StudentModel')
        
        if model_class == 'TeacherModel':
            model = TeacherModel(
                num_joints=17,
                coord_dim=2,
                embed_dim=768,
                depth=12,
                num_heads=12,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=16,
                mask_ratio=0.75
            )
        else:  # StudentModel
            model = StudentModel(
                num_patches=100,  # 需要根据实际数据调整
                patch_dim=256,    # 需要根据实际数据调整
                embed_dim=768,
                depth=12,
                num_heads=12,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=16,
                num_joints=17,
                coord_dim=2,
                contrast_dim=256,
                mask_ratio=0.75
            )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 创建样本输入
        if model_class == 'TeacherModel':
            sample_input = torch.randn(1, 17, 2)  # [batch, joints, coords]
        else:  # StudentModel
            sample_input = torch.randn(1, 100, 256)  # [batch, patches, patch_dim]
        
        # 导出ONNX
        success = export_model_to_onnx(model, sample_input, output_path, model_class)
        
        if success:
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   文件大小: {file_size_mb:.2f} MB")
        
        return success
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # list命令
    list_parser = subparsers.add_parser('list', help='列出所有检查点')
    list_parser.add_argument('checkpoint_dir', nargs='?', default='./outputs', 
                           help='检查点目录 (默认: ./outputs)')
    
    # info命令
    info_parser = subparsers.add_parser('info', help='查看检查点信息')
    info_parser.add_argument('checkpoint_path', help='检查点文件路径')
    
    # export命令
    export_parser = subparsers.add_parser('export', help='导出完整模型')
    export_parser.add_argument('checkpoint_path', help='源检查点路径')
    export_parser.add_argument('output_path', help='输出路径')
    
    # weights命令
    weights_parser = subparsers.add_parser('weights', help='仅导出模型权重')
    weights_parser.add_argument('checkpoint_path', help='源检查点路径')
    weights_parser.add_argument('output_path', help='输出路径')
    
    # onnx命令
    onnx_parser = subparsers.add_parser('onnx', help='导出为ONNX格式')
    onnx_parser.add_argument('checkpoint_path', help='源检查点路径')
    onnx_parser.add_argument('output_path', help='输出路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_all_checkpoints(args.checkpoint_dir)
    elif args.command == 'info':
        show_checkpoint_info(args.checkpoint_path)
    elif args.command == 'export':
        export_model(args.checkpoint_path, args.output_path)
    elif args.command == 'weights':
        export_weights_only(args.checkpoint_path, args.output_path)
    elif args.command == 'onnx':
        export_to_onnx(args.checkpoint_path, args.output_path)


if __name__ == "__main__":
    main()

