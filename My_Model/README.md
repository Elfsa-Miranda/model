# Enhanced Multi-Modal DMAE

增强型多模态DMAE，用于CSI-RGB跨模态对比学习和黑暗环境下的人体姿态估计。

## 🎯 项目概述

本项目实现了一个创新的跨模态学习框架，通过WiFi CSI信号预测人体骨骼点，特别适用于黑暗或隐私敏感环境。

### 核心特性

- **🏫 老师-学生架构**: RGB骨骼点老师模型指导CSI学生模型
- **🎭 掩码自编码**: 基于MAE的预训练和重建任务
- **🔄 知识蒸馏**: 跨模态特征对齐和知识转移
- **⚖️ 对比学习**: 正负样本对学习，增强跨模态表示
- **🌙 黑暗推理**: 仅使用CSI信号进行人体姿态估计

### 技术架构

```
训练阶段:
RGB骨骼点 → 老师模型(MAE) → 特征表示
CSI时频谱 → 学生模型 → 重建CSI + 预测骨骼点 + 特征表示
                    ↓
            三重损失: L_MAE + L_Distill + L_Contrast

推理阶段:
CSI时频谱 → 学生模型 → 预测骨骼点
```

## 📁 项目结构

```
My_Model/
├── __init__.py              # 包初始化
├── models.py                # 模型定义 (TeacherModel, StudentModel)
├── losses.py                # 损失函数 (MAE, 蒸馏, 对比学习)
├── data_processing.py       # 数据预处理 (CSI, 骨骼点)
├── utils.py                 # 工具函数 (训练, 评估, 可视化)
├── train.py                 # 训练流程
├── example.py               # 使用示例
├── config.yaml              # 配置文件
└── README.md               # 说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib pyyaml tqdm
pip install timm  # Transformer模型库

# 可选: 安装其他依赖
pip install tensorboard wandb  # 日志记录
```

### 2. 数据准备

确保MMFi数据集按照以下结构组织：

```
${DATASET_ROOT}/
├── E01/
│   ├── S01/
│   │   ├── A01/
│   │   │   ├── rgb/           # RGB关键点数据 (.npy)
│   │   │   ├── wifi-csi/      # WiFi-CSI数据 (.mat)
│   │   │   └── ground_truth.npy
│   │   └── ...
│   └── ...
└── ...
```

### 3. 配置设置

编辑 `config.yaml` 文件，调整模型和训练参数：

```yaml
# 基础设置
seed: 42
output_dir: "./outputs"

# 模型配置
teacher_model:
  embed_dim: 768
  depth: 12
  num_heads: 12

student_model:
  embed_dim: 768
  depth: 12
  num_heads: 12

# 训练配置
teacher_pretrain_epochs: 50
student_distill_epochs: 100
```

### 4. 模型训练

#### 方法1: 使用训练脚本

```bash
# 完整训练流程
python train.py C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave ../MMFi_dataset/config.yaml \
    --config config.yaml \
    --output_dir ./outputs
```

#### 方法2: 使用示例脚本

```bash
# 训练模型
python example.py train C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave ../MMFi_dataset/config.yaml \
    --config config.yaml \
    --output_dir ./outputs
```

### 5. 模型测试

```bash
# 测试模型性能
python example.py test ./outputs/student_checkpoints/best_model.pth \
    C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave ../MMFi_dataset/config.yaml \
    --output_dir ./test_results
```

### 6. 单样本推理

```bash
# 推理单个CSI文件
python example.py infer ./outputs/student_checkpoints/best_model.pth \
    /path/to/csi_file.mat \
    --output_dir ./inference_results
```

### 7. 演示模式

```bash
# 运行演示，测试所有组件
python example.py demo
```

## 📊 训练流程

### 阶段1: 老师模型预训练

- **输入**: RGB骨骼点数据
- **任务**: 掩码自编码重建
- **目标**: 学习骨骼点的语义表示

```python
# 老师模型预训练
loss, pred, mask = teacher_model(rgb_skeleton)
```

### 阶段2: 学生模型蒸馏训练

- **输入**: CSI时频谱数据
- **任务**: CSI重建 + 骨骼点预测 + 知识蒸馏 + 对比学习
- **目标**: 从CSI学习预测骨骼点

```python
# 学生模型训练
student_outputs = student_model(csi_patches)
total_loss = mae_loss + distill_loss + contrast_loss
```

### 损失函数组合

```
L_total = L_MAE + α * L_Distill + β * L_Contrast

其中:
- L_MAE: CSI重建损失 + 骨骼点预测损失
- L_Distill: 学生-老师特征对齐损失
- L_Contrast: 正负样本对对比损失
- α, β: 损失权重 (默认: α=1.0, β=0.5)
```

## 🔧 配置说明

### 模型配置

```yaml
# 老师模型 (RGB骨骼点MAE)
teacher_model:
  num_joints: 17           # 关节点数量
  coord_dim: 2             # 坐标维度
  embed_dim: 768           # 嵌入维度
  depth: 12                # Transformer层数
  mask_ratio: 0.75         # 掩码比例

# 学生模型 (CSI多分支)
student_model:
  embed_dim: 768           # 嵌入维度
  depth: 12                # Transformer层数
  contrast_dim: 256        # 对比学习特征维度
  mask_ratio: 0.75         # 掩码比例
```

### 损失配置

```yaml
combined_loss:
  mae_weight: 1.0          # MAE损失权重
  distill_weight: 1.0      # 蒸馏损失权重
  contrast_weight: 0.5     # 对比学习损失权重
```

### 训练配置

```yaml
# 训练epoch数
teacher_pretrain_epochs: 50    # 老师预训练
student_distill_epochs: 100    # 学生蒸馏训练

# 优化器
teacher_optimizer:
  type: "adamw"
  lr: 1.0e-4
  weight_decay: 1.0e-2

# 学习率调度器
teacher_scheduler:
  type: "cosine"
  T_max: 50
```

## 📈 评估指标

### 骨骼点预测指标

- **MPJPE**: Mean Per Joint Position Error (平均关节位置误差)
- **PCK@α**: Percentage of Correct Keypoints (正确关键点百分比)
  - PCK@0.05, PCK@0.1, PCK@0.2, PCK@0.5

### 训练监控

- **MAE损失**: CSI重建 + 骨骼点预测
- **蒸馏损失**: 特征对齐
- **对比损失**: 正负样本对区分

## 🎨 可视化功能

### 训练曲线

```python
# 自动生成训练曲线
visualize_training_curves(loss_history, save_path)
```

### 骨骼点预测

```python
# 可视化预测结果
visualize_skeleton_prediction(pred_skeleton, target_skeleton, save_path)
```

### CSI时频谱

```python
# 可视化CSI数据
visualize_csi_data(csi_data, save_path)
```

## 🔍 使用示例

### Python API使用

```python
from My_Model import EnhancedDMAEInference

# 创建推理器
inference = EnhancedDMAEInference(model_path, config_path)

# 预测骨骼点
skeleton = inference.predict_skeleton(csi_data)

# 带置信度预测
mean_skeleton, std_skeleton = inference.predict_with_confidence(csi_data)
```

### 批量处理

```python
# 批量预测
batch_csi = torch.stack([csi1, csi2, csi3])  # [batch, ...]
batch_skeletons = inference.predict_skeleton(batch_csi)
```

## 📋 模型性能

### 计算复杂度

- **老师模型**: ~86M 参数 (ViT-Base规模)
- **学生模型**: ~86M 参数 + 多分支头
- **推理速度**: ~10ms/样本 (GPU)

### 内存使用

- **训练**: ~8GB GPU内存 (batch_size=32)
- **推理**: ~2GB GPU内存

## 🛠️ 高级功能

### 自定义数据集

```python
# 继承并实现自定义数据集
class CustomCSIDataset(Dataset):
    def __init__(self, csi_files, skeleton_files):
        # 实现数据加载逻辑
        pass
    
    def __getitem__(self, idx):
        # 返回 {'csi': csi_data, 'skeleton': skeleton_data}
        pass
```

### 模型微调

```python
# 加载预训练模型
model = StudentModel(...)
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 冻结部分层
for param in model.blocks[:6].parameters():
    param.requires_grad = False

# 微调训练
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)
```

### 分布式训练

```python
# 使用DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```yaml
   # 减小batch_size或模型尺寸
   train_loader:
     batch_size: 16  # 从32减小到16
   ```

2. **收敛缓慢**
   ```yaml
   # 调整学习率和权重
   combined_loss:
     distill_weight: 2.0  # 增加蒸馏损失权重
   ```

3. **预测精度低**
   ```yaml
   # 增加训练epoch或调整掩码比例
   student_distill_epochs: 200
   student_model:
     mask_ratio: 0.5  # 降低掩码比例
   ```

### 调试模式

```bash
# 启用详细日志
python train.py --debug --log_level DEBUG
```

## 📚 参考文献

1. **MAE**: Masked Autoencoders Are Scalable Vision Learners
2. **DMAE**: Distilled Masked Autoencoders
3. **SimCLR**: A Simple Framework for Contrastive Learning
4. **MMFi**: Multi-Modal Human Activity Recognition

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [https://github.com/your-username/enhanced-dmae](https://github.com/your-username/enhanced-dmae)
- 问题反馈: [Issues](https://github.com/your-username/enhanced-dmae/issues)

---

**注意**: 本项目仅用于研究目的，请遵守相关数据使用协议和隐私保护规定。
