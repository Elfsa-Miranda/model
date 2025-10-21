#!/usr/bin/env python3
"""
TensorBoard启动脚本
用于启动TensorBoard可视化训练进度

使用方法:
python start_tensorboard.py [log_dir] [port]

示例:
python start_tensorboard.py ./outputs/tensorboard_logs
python start_tensorboard.py ./outputs/tensorboard_logs 6006
"""

import os
import sys
import subprocess
import argparse


def start_tensorboard(log_dir, port=6006):
    """
    启动TensorBoard
    
    Args:
        log_dir: TensorBoard日志目录
        port: TensorBoard端口号
    """
    # 检查日志目录是否存在
    if not os.path.exists(log_dir):
        print(f"❌ 错误: TensorBoard日志目录不存在: {log_dir}")
        print("请先运行训练脚本生成日志文件")
        return False
    
    # 检查是否有日志文件
    log_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.tfevents'):
                log_files.append(os.path.join(root, file))
    
    if not log_files:
        print(f"❌ 错误: 在 {log_dir} 中没有找到TensorBoard日志文件")
        print("请确保训练已经开始并生成了日志")
        return False
    
    print(f"✅ 找到 {len(log_files)} 个TensorBoard日志文件")
    print(f"🚀 启动TensorBoard...")
    print(f"   日志目录: {log_dir}")
    print(f"   端口: {port}")
    print(f"   访问地址: http://localhost:{port}")
    print("\n按 Ctrl+C 停止TensorBoard")
    
    try:
        # 启动TensorBoard
        cmd = ['tensorboard', '--logdir', log_dir, '--port', str(port)]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ TensorBoard启动失败: {e}")
        print("请确保已安装tensorboard: pip install tensorboard")
        return False
    except KeyboardInterrupt:
        print("\n✅ TensorBoard已停止")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动TensorBoard可视化")
    parser.add_argument("log_dir", nargs='?', default="./outputs/tensorboard_logs", 
                       help="TensorBoard日志目录 (默认: ./outputs/tensorboard_logs)")
    parser.add_argument("--port", type=int, default=6006, 
                       help="TensorBoard端口号 (默认: 6006)")
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    log_dir = os.path.abspath(args.log_dir)
    
    print("=" * 60)
    print("TensorBoard 启动器")
    print("=" * 60)
    
    success = start_tensorboard(log_dir, args.port)
    
    if success:
        print("✅ TensorBoard启动成功")
    else:
        print("❌ TensorBoard启动失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

