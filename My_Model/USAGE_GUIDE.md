# Enhanced Multi-Modal DMAE 使用指南

## 📋 概述

Enhanced Multi-Modal DMAE 是一个创新的跨模态学习框架，专门用于从WiFi CSI信号预测人体骨骼点，特别适用于黑暗或隐私敏感环境。本指南将详细说明如何使用MMFi数据集训练和部署模型。

## 🗂️ 项目结构

```
My_Model/
├── __init__.py                    # 包初始化
├── models.py                      # 模型定义 (TeacherModel, StudentModel)
├── losses.py                      # 损失函数 (MAE, 蒸馏, 对比学习)
├── data_processing.py             # 数据预处理 (CSI, 骨骼点)
├── utils.py                       # 工具函数 (训练, 评估, 可视化)
├── train.py                       # 训练流程
├── example.py                     # 使用示例
├── mmfi_dataloader.py            # MMFi数据加载器 (新增)
├── test_mmfi_integration.py      # 整合测试脚本 (新增)
├── test_components.py             # 组件测试脚本
├── config.yaml                    # 配置文件
├── README.md                      # 项目文档
└── USAGE_GUIDE.md                # 本使用指南
```

## 🎯 核心改进

### 1. MMFi数据集整合

- **新增**: `mmfi_dataloader.py` - 专门处理MMFi数据集的数据加载器
- **特性**: 
  - 直接读取RGB骨骼点 (.npy文件)
  - 处理CSI数据 (.mat文件)
  - 支持frame和sequence两种模式
  - 自动数据预处理和归一化

### 2. 数据格式适配

- **RGB骨骼点**: 从MMFi数据集的`rgb/frame*.npy`文件读取2D关键点
- **CSI数据**: 从MMFi数据集的`wifi-csi/frame*.mat`文件读取CSIamp
- **Ground Truth**: 从`ground_truth.npy`读取3D骨骼点标注

### 3. 训练流程优化

- **阶段1**: 老师模型使用RGB骨骼点进行MAE预训练
- **阶段2**: 学生模型使用CSI数据，结合知识蒸馏和对比学习
- **自动**: 数据预处理、模型保存、指标计算

## 🚀 快速开始

### 步骤1: 准备MMFi数据集

确保你的MMFi数据集按照以下结构组织：

```
${DATASET_ROOT}/
├── E01/
│   ├── S01/
│   │   ├── A01/
│   │   │   ├── rgb/           # RGB关键点数据 (.npy文件)
│   │   │   │   ├── frame001.npy
│   │   │   │   ├── frame002.npy
│   │   │   │   └── ...
│   │   │   ├── wifi-csi/      # WiFi-CSI数据 (.mat文件)
│   │   │   │   ├── frame001.mat
│   │   │   │   ├── frame002.mat
│   │   │   │   └── ...
│   │   │   └── ground_truth.npy
│   │   └── ...
│   └── ...
└── ...
```

### 步骤2: 配置文件设置

编辑 `config.yaml` 文件：

```yaml
# 基础设置
seed: 42
output_dir: "./outputs"
batch_size: 32
num_workers: 4

# CSI预处理器配置 (根据你的数据调整)
csi_preprocessor:
  num_antennas: 3          # MMFi数据集的天线数量
  num_subcarriers: 30      # 子载波数量
  time_length: 297         # 时间序列长度
  patch_size: 16           # 补丁大小
  normalize: true

# 模型配置
teacher_model:
  embed_dim: 768
  depth: 12
  num_heads: 12

student_model:
  embed_dim: 768
  depth: 12
  num_heads: 12
```

### 步骤3: 数据集配置

使用MMFi_dataset中的配置文件，或创建新的配置：

```yaml
# MMFi_dataset/config.yaml
modality: rgb|wifi-csi 
protocol: protocol3
data_unit: frame  # 使用单帧模式进行姿态估计
split_to_use: random_split

random_split:
  ratio: 0.8
  random_seed: 0

train_loader:
  batch_size: 32
validation_loader:
  batch_size: 1
```

### 步骤4: 测试整合

在开始训练前，先测试数据加载和模型整合：

```bash
# 测试MMFi数据集整合
cd pythonProject/My_Model
python test_mmfi_integration.py
```

这个脚本会：
- 测试MMFi数据加载
- 验证模型与真实数据的兼容性
- 测试训练步骤

### 步骤5: 开始训练

```bash
# 完整训练流程
python train.py C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave ../MMFi_dataset/config.yaml \
    --config config.yaml \
    --output_dir ./outputs

# 示例 (Windows)
cd C:\TSM\MMFi-MoCov2\pythonProject\My_Model

python train.py "C:\\tangyx\\MMFi_Dataset\\filtered_mmwave\\filtered_mmwave" "C:\TSM\MMFi-MoCov2\pythonProject\MMFi_dataset\config.yaml" --config config.yaml --output_dir ".\outputs"                                               

#指定从特定阶段继续（如果你想要从学生模型阶段开始（假设老师模型已经训练完成））：
bash
cd C:\TSM\MMFi-MoCov2\pythonProject\My_Model

1. 先修改 config.yaml：
yaml# config.yaml
teacher_pretrain_epochs: 0      # 🔑 设为 0 跳过教师预训练
student_distill_epochs: 100     # 开始学生模型训练
save_freq: 10

2. 运行命令：
python train.py "C:\tangyx\MMFi_Dataset\filtered_mmwave\filtered_mmwave" "C:\TSM\MMFi-MoCov2\pythonProject\MMFi_dataset\config.yaml" --config config.yaml --output_dir ".\outputs" --resume ".\outputs\teacher_checkpoints\best_model.pth" 

```
## 📊 数据流程详解

### 1. 数据加载流程

```python
# MMFi数据加载器的工作流程
from mmfi_dataloader import create_enhanced_mmfi_dataloaders

# 创建数据加载器
train_loader, val_loader = create_enhanced_mmfi_dataloaders(
    dataset_root="C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave",
    config_path="../MMFi_dataset/config.yaml",
    batch_size=32
)

# 数据批次格式
for batch in train_loader:
    rgb_skeleton = batch['rgb_skeleton']    # [batch, 17, 2] RGB 2D关键点
    csi_data = batch['csi_data']           # [batch, freq, time, antennas] CSI数据
    gt_skeleton = batch['gt_skeleton']      # [batch, 17, 3] 3D关键点标注
    # ... 其他元信息
```

### 2. 数据预处理流程

```python
# CSI数据预处理
csi_data_reshaped = csi_data.permute(0, 3, 1, 2)  # 转换维度
csi_patches, _ = csi_preprocessor(csi_data_reshaped)

# RGB骨骼点预处理  
processed_rgb = skeleton_preprocessor(rgb_skeleton)
```

### 3. 模型训练流程

```python
# 老师模型预训练
teacher_loss, pred, mask = teacher_model(processed_rgb)

# 学生模型训练
student_outputs = student_model(csi_patches)
teacher_features = teacher_model.forward_features(processed_rgb)

# 组合损失计算
total_loss = mae_loss + distill_loss + contrast_loss
```

## 🔧 高级配置

### 1. 调整CSI预处理参数

根据你的MMFi数据集实际参数调整：

```python
# 检查你的CSI数据维度
import scipy.io as scio
csi_sample = scio.loadmat('path/to/frame001.mat')['CSIamp']
print(f"CSI数据形状: {csi_sample.shape}")  # 例如: (30, 100, 3)

# 相应调整配置
csi_preprocessor:
  num_antennas: 3      # 对应最后一个维度
  num_subcarriers: 30  # 对应第一个维度
  time_length: 100     # 对应第二个维度
```

### 2. 模型规模调整

根据你的硬件资源调整模型大小：

```yaml
# 小型模型 (适合GPU内存<8GB)
teacher_model:
  embed_dim: 384
  depth: 6
  num_heads: 6

student_model:
  embed_dim: 384
  depth: 6
  num_heads: 6

# 大型模型 (适合GPU内存>=16GB)
teacher_model:
  embed_dim: 1024
  depth: 24
  num_heads: 16

student_model:
  embed_dim: 1024
  depth: 24
  num_heads: 16
```

### 3. 训练策略调整

```yaml
# 快速训练 (测试用)
teacher_pretrain_epochs: 10
student_distill_epochs: 20

# 完整训练 (最佳性能)
teacher_pretrain_epochs: 100
student_distill_epochs: 200

# 损失权重调整
combined_loss:
  mae_weight: 1.0
  distill_weight: 2.0    # 增加蒸馏权重
  contrast_weight: 0.5
```

## 🧪 测试和验证

### 1. 组件测试

```bash
# 测试所有组件
python test_components.py

# 测试MMFi整合
python test_mmfi_integration.py
```

### 2. 模型测试

```bash
# 测试训练好的模型
python example.py test /path/to/model.pth C:/tangyx/MMFi_Dataset/filtered_mmwave/filtered_mmwave config.yaml \
    --output_dir ./test_results
```

### 3. 单样本推理

```bash
# 推理单个CSI文件
python example.py infer /path/to/model.pth /path/to/csi_file.mat \
    --output_dir ./inference_results
```

## 💾 模型保存和管理

### 1. 自动保存机制

**训练过程中自动保存：**
- ✅ **定期保存**：每10个epoch自动保存检查点
- ✅ **最佳模型保存**：验证损失最低时自动保存为`best_model.pth`
- ✅ **完整状态保存**：包含模型参数、优化器状态、学习率调度器

**保存位置：**
```
./outputs/
├── teacher_checkpoints/          # 老师模型检查点
│   ├── teacher_epoch_10.pth
│   ├── teacher_epoch_20.pth
│   └── best_model.pth
├── student_checkpoints/          # 学生模型检查点
│   ├── student_epoch_10.pth
│   ├── student_epoch_20.pth
│   └── best_model.pth
└── tensorboard_logs/            # TensorBoard日志
```

### 2. 模型管理工具

**使用模型管理脚本：**

```bash
# 列出所有检查点
python model_manager.py list ./outputs

# 查看检查点详细信息
python model_manager.py info ./outputs/student_checkpoints/best_model.pth

# 导出完整模型（包含训练状态）
python model_manager.py export ./outputs/student_checkpoints/best_model.pth ./exported_model.pth

# 仅导出模型权重（用于推理）
python model_manager.py weights ./outputs/student_checkpoints/best_model.pth ./model_weights.pth

# 导出为ONNX格式（用于部署）
python model_manager.py onnx ./outputs/student_checkpoints/best_model.pth ./model.onnx
```

### 3. 保存内容详解

**完整检查点包含：**
```python
checkpoint = {
    'epoch': 50,                           # 训练轮次
    'model_state_dict': {...},             # 模型参数
    'optimizer_state_dict': {...},         # 优化器状态
    'scheduler_state_dict': {...},         # 学习率调度器状态
    'loss': 0.1234,                        # 当前损失
    'timestamp': 1234567890,               # 保存时间戳
    'model_info': {                        # 模型信息
        'total_params': 12345678,          # 总参数数量
        'trainable_params': 12345678,      # 可训练参数数量
        'model_size_mb': 45.6,             # 模型大小(MB)
        'model_class': 'StudentModel'      # 模型类型
    }
}
```

**仅权重文件包含：**
```python
weights = {
    'model_state_dict': {...},             # 模型参数
    'model_class': 'StudentModel',         # 模型类型
    'timestamp': 1234567890                # 保存时间戳
}
```

### 4. 模型加载和恢复

**恢复训练：**
```bash
python train.py /path/to/dataset config.yaml --resume ./outputs/student_checkpoints/best_model.pth
```

**加载模型进行推理：**
```python
from utils import load_checkpoint
from models import StudentModel

# 加载检查点
checkpoint = torch.load('./outputs/student_checkpoints/best_model.pth')
model = StudentModel(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**使用example.py进行推理：**
```bash
python example.py test ./outputs/student_checkpoints/best_model.pth /path/to/test_data
python example.py infer ./outputs/student_checkpoints/best_model.pth /path/to/csi_file.mat
```

### 5. 模型部署选项

**PyTorch格式（推荐）：**
- ✅ 完整功能支持
- ✅ 易于调试和修改
- ❌ 需要PyTorch环境

**ONNX格式（部署推荐）：**
- ✅ 跨平台兼容
- ✅ 推理速度快
- ✅ 支持多种推理引擎
- ❌ 功能可能受限

**权重文件（轻量级）：**
- ✅ 文件最小
- ✅ 加载速度快
- ❌ 需要重新创建模型结构

## 📈 训练监控

### 1. 实时监控 - TensorBoard

**启动TensorBoard可视化：**

```bash
# 方法1: 使用启动脚本 (推荐)
python start_tensorboard.py

# 方法2: 直接启动
tensorboard --logdir ./outputs/tensorboard_logs --port 6006

# 方法3: 指定自定义目录
python start_tensorboard.py /path/to/your/logs --port 6007
```

**访问TensorBoard：**
- 打开浏览器访问: `http://localhost:6006`
- 实时查看训练进度，无需等待训练完成

**TensorBoard监控内容：**
- **Teacher模型**:
  - `Teacher/Train_Loss`: 老师模型训练损失
  - `Teacher/Val_Loss`: 老师模型验证损失  
  - `Teacher/Learning_Rate`: 学习率变化
- **Student模型**:
  - `Student/Train_Total_Loss`: 总训练损失
  - `Student/Train_MAE_Loss`: MAE损失
  - `Student/Train_Distill_Loss`: 蒸馏损失
  - `Student/Train_Contrast_Loss`: 对比学习损失
  - `Student/Val_Loss`: 验证损失
  - `Student/Val_MPJPE`: 平均关节位置误差
  - `Student/Val_PCK@*`: 正确关键点百分比
  - `Student/Learning_Rate`: 学习率变化

### 2. 控制台监控

训练过程中会显示：
- `teacher_pretrain_loss`: 老师模型MAE损失
- `mae_total_loss`: 学生模型MAE损失
- `distill_loss`: 知识蒸馏损失
- `contrast_loss`: 对比学习损失
- `total_loss`: 总损失

### 3. 评估指标

- `MPJPE`: Mean Per Joint Position Error (平均关节位置误差)
- `PCK@α`: Percentage of Correct Keypoints (正确关键点百分比)

### 4. 可视化输出

训练完成后会自动生成：
- `training_curves.png`: 训练曲线
- `test_sample_*.png`: 测试样本可视化
- `training_log.json`: 详细训练日志
- `tensorboard_logs/`: TensorBoard日志目录

## 🚨 常见问题解决

### 1. 数据加载错误

```
错误: FileNotFoundError: 数据集根目录不存在
解决: 检查MMFi数据集路径是否正确
```

```
错误: 数据形状不匹配
解决: 检查CSI预处理器参数是否与实际数据匹配
```

### 2. 内存不足

```
错误: CUDA out of memory
解决: 
- 减小batch_size (如从32改为16)
- 减小模型embed_dim (如从768改为384)
- 使用更小的patch_size
```

### 3. 收敛问题

```
问题: 损失不收敛
解决:
- 调整学习率 (减小lr)
- 调整损失权重 (增加distill_weight)
- 检查数据预处理是否正确
```

## 📝 最佳实践

### 1. 数据准备

- 确保MMFi数据集完整下载
- 检查数据文件是否损坏
- 验证RGB和CSI数据时间对齐

### 2. 训练策略

- 先用小模型和少量数据测试
- 逐步增加模型规模和训练时间
- 定期保存检查点

### 3. 性能优化

- 使用多GPU训练（如果可用）
- 启用混合精度训练
- 优化数据加载（调整num_workers）

## 🎯 下一步发展

1. **模型改进**: 尝试不同的Transformer架构
2. **数据增强**: 添加CSI和RGB数据增强技术
3. **多模态融合**: 探索更好的跨模态融合方法
4. **实时推理**: 优化模型用于实时应用

## 📞 支持

如果遇到问题，请检查：
1. 数据集路径和格式是否正确
2. 依赖库是否完整安装
3. 配置文件是否正确设置
4. 硬件资源是否足够

---

**祝你训练顺利！🎉**








