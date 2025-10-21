"""
Enhanced Multi-Modal DMAE for Cross-Modal Contrastive Learning
增强型多模态DMAE，用于CSI-RGB跨模态对比学习

主要组件:
- CSI预处理模块 (STFT时频谱转换)
- 老师模型 (RGB骨骼点MAE预训练)
- 学生模型 (CSI多分支架构)
- 三重损失函数 (MAE + 蒸馏 + 对比)
"""

__version__ = "1.0.0"
__author__ = "MMFi-MoCov2 Team"

from .models import *
from .losses import *
from .utils import *
from .data_processing import *

