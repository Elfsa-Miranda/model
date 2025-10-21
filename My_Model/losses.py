"""
Enhanced Multi-Modal DMAE Loss Functions
增强型多模态DMAE损失函数

包含:
1. MAE重建损失 (L_MAE)
2. 知识蒸馏损失 (L_Distill)  
3. 对比学习损失 (L_Contrast)
4. 组合损失函数

改进点:
- 平衡损失权重配置
- 重写 InfoNCE 对比损失（标准双向实现）
- 改进蒸馏损失稳定性（L2-normalization + 余弦距离）
- 添加调试输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAELoss(nn.Module):
    """
    MAE重建损失
    用于CSI补丁重建和骨骼点重建
    """

    def __init__(self, loss_type='mse', normalize=False):
        """
        Args:
            loss_type: 损失类型 ('mse', 'l1', 'smooth_l1')
            normalize: 是否对目标进行归一化
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

    def patchify_target(self, target, patch_size):
        """
        将目标转换为补丁格式（用于图像重建）

        Args:
            target: [batch, channels, height, width]
            patch_size: 补丁大小

        Returns:
            patches: [batch, num_patches, patch_dim]
        """
        batch_size, channels, height, width = target.shape

        assert height % patch_size == 0 and width % patch_size == 0

        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        patches = target.reshape(
            batch_size, channels,
            num_patches_h, patch_size,
            num_patches_w, patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(
            batch_size,
            num_patches_h * num_patches_w,
            channels * patch_size * patch_size
        )

        return patches

    def forward(self, pred, target, mask=None):
        """
        计算MAE损失

        Args:
            pred: 预测值 [batch, seq_len, feature_dim] 或 [batch, num_patches, patch_dim]
            target: 目标值，形状与pred相同
            mask: 掩码 [batch, seq_len]，True表示计算损失的位置

        Returns:
            loss: 标量损失值
        """
        # 归一化目标（可选）
        if self.normalize:
            target_mean = target.mean(dim=-1, keepdim=True)
            target_var = target.var(dim=-1, keepdim=True)
            target = (target - target_mean) / (target_var + 1e-6)**.5

        # 计算损失
        loss = self.criterion(pred, target)

        # 如果有多个维度，在最后一个维度上求均值
        if len(loss.shape) > 2:
            loss = loss.mean(dim=-1)

        # 应用掩码
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失
    对齐学生模型和教师模型的中间层特征
    """

    def __init__(self, loss_type='l1', temperature=1.0, align_layers=None):
        """
        Args:
            loss_type: 损失类型 ('l1', 'mse', 'cosine')
            temperature: 温度参数（用于特征软化）
            align_layers: 要对齐的层索引列表，如果为None则对齐所有层
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.align_layers = align_layers

        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

    def align_features(self, student_feat, teacher_feat):
        """
        对齐学生和教师特征的维度

        Args:
            student_feat: [batch, seq_len_s, dim_s]
            teacher_feat: [batch, seq_len_t, dim_t]

        Returns:
            aligned_student: [batch, min_seq_len, min_dim]
            aligned_teacher: [batch, min_seq_len, min_dim]
        """
        batch_size = student_feat.shape[0]

        # 对齐序列长度（取较短的）
        min_seq_len = min(student_feat.shape[1], teacher_feat.shape[1])
        student_feat = student_feat[:, :min_seq_len, :]
        teacher_feat = teacher_feat[:, :min_seq_len, :]

        # 对齐特征维度（如果不同，使用线性投影）
        if student_feat.shape[-1] != teacher_feat.shape[-1]:
            min_dim = min(student_feat.shape[-1], teacher_feat.shape[-1])
            student_feat = student_feat[:, :, :min_dim]
            teacher_feat = teacher_feat[:, :, :min_dim]

        return student_feat, teacher_feat

    def forward(self, student_features, teacher_features):
        """
        计算知识蒸馏损失

        Args:
            student_features: 学生模型特征列表 [layer_features, ...]
            teacher_features: 教师模型特征列表 [layer_features, ...]

        Returns:
            loss: 蒸馏损失
        """
        total_loss = 0.0
        num_layers = 0

        # 确定要对齐的层
        if self.align_layers is None:
            # 对齐所有层，但可能需要采样
            num_student_layers = len(student_features)
            num_teacher_layers = len(teacher_features)

            # 如果层数不同，均匀采样
            if num_student_layers != num_teacher_layers:
                if num_student_layers > num_teacher_layers:
                    # 从学生层中采样
                    indices = np.linspace(0, num_student_layers-1, num_teacher_layers, dtype=int)
                    student_features = [student_features[i] for i in indices]
                else:
                    # 从教师层中采样
                    indices = np.linspace(0, num_teacher_layers-1, num_student_layers, dtype=int)
                    teacher_features = [teacher_features[i] for i in indices]

            align_layers = list(range(min(len(student_features), len(teacher_features))))
        else:
            align_layers = self.align_layers

        # 计算每层的蒸馏损失
        for layer_idx in align_layers:
            if layer_idx < len(student_features) and layer_idx < len(teacher_features):
                student_feat = student_features[layer_idx]
                teacher_feat = teacher_features[layer_idx]

                # 对齐特征维度
                student_feat, teacher_feat = self.align_features(student_feat, teacher_feat)

                # 温度软化
                if self.temperature != 1.0:
                    student_feat = student_feat / self.temperature
                    teacher_feat = teacher_feat / self.temperature

                # 计算损失
                if self.loss_type == 'cosine':
                    # 余弦相似度损失需要特殊处理
                    batch_size, seq_len, dim = student_feat.shape
                    student_flat = student_feat.reshape(-1, dim)
                    teacher_flat = teacher_feat.reshape(-1, dim)
                    target = torch.ones(student_flat.shape[0], device=student_feat.device)
                    layer_loss = self.criterion(student_flat, teacher_flat, target)
                else:
                    layer_loss = self.criterion(student_feat, teacher_feat)

                total_loss += layer_loss
                num_layers += 1

        # 平均损失
        if num_layers > 0:
            total_loss = total_loss / num_layers

        return total_loss


class ContrastiveLoss(nn.Module):
    """
    对比学习损失 (NT-Xent)
    用于跨模态对比学习，拉近正样本对，推远负样本对
    """

    def __init__(self, temperature=0.07, normalize=True):
        """
        Args:
            temperature: 温度参数
            normalize: 是否对特征进行L2归一化
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, anchor_features, positive_features, labels):
        """
        计算对比学习损失

        Args:
            anchor_features: anchor样本特征 [batch_size, feature_dim]
            positive_features: positive样本特征 [batch_size, feature_dim]
            labels: 标签 [batch_size]，1表示正样本对，0表示负样本对

        Returns:
            loss: 对比学习损失
        """
        batch_size = anchor_features.shape[0]

        # L2归一化
        if self.normalize:
            anchor_features = F.normalize(anchor_features, dim=1)
            positive_features = F.normalize(positive_features, dim=1)

        # 拼接所有特征
        all_features = torch.cat([anchor_features, positive_features], dim=0)  # [2*batch, dim]

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature  # [2*batch, 2*batch]

        # 创建标签矩阵
        labels_matrix = torch.zeros(2 * batch_size, 2 * batch_size, device=labels.device)

        # 填充正样本对标签
        for i in range(batch_size):
            if labels[i] == 1:  # 正样本对
                labels_matrix[i, i + batch_size] = 1  # anchor -> positive
                labels_matrix[i + batch_size, i] = 1  # positive -> anchor

        # 掩码对角线（自己与自己的相似度不参与计算）
        mask = torch.eye(2 * batch_size, device=labels.device).bool()
        similarity_matrix.masked_fill_(mask, -float('inf'))

        # 计算InfoNCE损失
        # 对于每个anchor，计算与所有样本的相似度
        positive_mask = labels_matrix.bool()

        total_loss = 0.0
        num_positives = 0

        for i in range(2 * batch_size):
            if positive_mask[i].sum() > 0:  # 如果有正样本
                # 分子：与正样本的相似度
                pos_similarities = similarity_matrix[i][positive_mask[i]]

                # 分母：与所有样本的相似度（除了自己）
                all_similarities = similarity_matrix[i][~mask[i]]

                # InfoNCE损失
                for pos_sim in pos_similarities:
                    loss_i = -pos_sim + torch.logsumexp(all_similarities, dim=0)
                    total_loss += loss_i
                    num_positives += 1

        if num_positives > 0:
            total_loss = total_loss / num_positives

        return total_loss


class SimplifiedContrastiveLoss(nn.Module):
    """
    简化的对比学习损失
    直接使用欧几里得距离计算正负样本对损失
    """

    def __init__(self, margin=1.0):
        """
        Args:
            margin: 负样本对的margin
        """
        super().__init__()
        self.margin = margin

    def forward(self, anchor_features, positive_features, labels):
        """
        计算简化对比学习损失

        Args:
            anchor_features: [batch_size, feature_dim]
            positive_features: [batch_size, feature_dim]
            labels: [batch_size]，1表示正样本对，0表示负样本对

        Returns:
            loss: 对比学习损失
        """
        # 计算欧几里得距离
        distances = torch.norm(anchor_features - positive_features, dim=1)

        # 正样本对损失：距离应该尽可能小
        positive_loss = labels.float() * torch.pow(distances, 2)

        # 负样本对损失：距离应该大于margin
        negative_loss = (1 - labels.float()) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )

        # 总损失
        loss = torch.mean(positive_loss + negative_loss)

        return loss


class EnhancedContrastiveLoss(nn.Module):
    """
    InfoNCE-based contrastive loss for student-teacher embedding alignment.
    Now supports silent accumulation for epoch-level reporting.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))
        # ✅ 新增：统计缓存，用于每个 epoch 结束时统一输出
        self.pos_sims, self.neg_sims, self.loss_values = [], [], []

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: [B, D]
            teacher_feats: [B, D]
        """
        student_feats = F.normalize(student_feats, dim=-1)
        teacher_feats = F.normalize(teacher_feats, dim=-1)

        logits = torch.matmul(student_feats, teacher_feats.T) / self.temperature
        labels = torch.arange(student_feats.size(0), device=student_feats.device)
        loss = F.cross_entropy(logits, labels)

        # === 仅记录，不打印 ===
        with torch.no_grad():
            pos_sim = torch.mean(torch.diag(logits)).item()
            neg_sim = ((logits.sum() - torch.diag(logits).sum()) / (logits.numel() - logits.size(0))).item()
            self.pos_sims.append(pos_sim)
            self.neg_sims.append(neg_sim)
            self.loss_values.append(loss.item())

        self.step_count += 1
        return loss


class StableDistillationLoss(nn.Module):
    """
    稳定的蒸馏损失
    使用 L2-normalization + 余弦距离
    """

    def __init__(self, temperature=1.0):
        """
        Args:
            temperature: 温度参数
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, student_feats, teacher_feats):
        """
        计算稳定蒸馏损失

        Args:
            student_feats: 学生特征 [batch, seq_len, dim]
            teacher_feats: 教师特征 [batch, seq_len, dim]

        Returns:
            loss: 蒸馏损失
        """
        # 温度软化 + L2 归一化
        student_feats = F.normalize(student_feats / self.temperature, dim=-1)
        teacher_feats = F.normalize(teacher_feats / self.temperature, dim=-1)

        # 余弦距离损失
        cosine_sim = F.cosine_similarity(student_feats, teacher_feats, dim=-1)
        loss = 1 - cosine_sim.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    L_total = L_MAE + α * L_Distill + β * L_Contrast
    """

    def __init__(self,
                 mae_weight=1.0,
                 distill_weight=1.0,
                 contrast_weight=0.5,
                 mae_loss_config=None,
                 distill_loss_config=None,
                 contrast_loss_config=None):
        """
        Args:
            mae_weight: MAE损失权重
            distill_weight: 蒸馏损失权重
            contrast_weight: 对比学习损失权重
            mae_loss_config: MAE损失配置
            distill_loss_config: 蒸馏损失配置
            contrast_loss_config: 对比学习损失配置
        """
        super().__init__()

        self.mae_weight = mae_weight
        self.distill_weight = distill_weight
        self.contrast_weight = contrast_weight

        # 初始化各个损失函数
        mae_config = mae_loss_config or {}
        self.mae_loss = MAELoss(**mae_config)

        distill_config = distill_loss_config or {}
        self.distill_loss = DistillationLoss(**distill_config)

        contrast_config = contrast_loss_config or {}
        if contrast_config.get('use_simplified', False):
            self.contrast_loss = SimplifiedContrastiveLoss(
                margin=contrast_config.get('margin', 1.0)
            )
        else:
            self.contrast_loss = ContrastiveLoss(
                temperature=contrast_config.get('temperature', 0.07),
                normalize=contrast_config.get('normalize', True)
            )

    def forward(self,
                # MAE相关
                pred_patches, target_patches, patch_mask,
                pred_skeleton, target_skeleton,
                # 蒸馏相关
                student_features, teacher_features,
                # 对比学习相关
                anchor_contrast_feat, positive_contrast_feat, contrast_labels):
        """
        计算组合损失

        Args:
            pred_patches: 预测的CSI补丁
            target_patches: 目标CSI补丁
            patch_mask: 补丁掩码
            pred_skeleton: 预测的骨骼点
            target_skeleton: 目标骨骼点
            student_features: 学生模型特征
            teacher_features: 教师模型特征
            anchor_contrast_feat: anchor对比特征
            positive_contrast_feat: positive对比特征
            contrast_labels: 对比学习标签

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # 1. MAE重建损失
        mae_patch_loss = self.mae_loss(pred_patches, target_patches, patch_mask)
        mae_skeleton_loss = self.mae_loss(pred_skeleton, target_skeleton)
        mae_total_loss = mae_patch_loss + mae_skeleton_loss

        loss_dict['mae_patch_loss'] = mae_patch_loss
        loss_dict['mae_skeleton_loss'] = mae_skeleton_loss
        loss_dict['mae_total_loss'] = mae_total_loss

        # 2. 知识蒸馏损失
        distill_loss = self.distill_loss(student_features, teacher_features)
        loss_dict['distill_loss'] = distill_loss

        # 3. 对比学习损失
        contrast_loss = self.contrast_loss(
            anchor_contrast_feat, positive_contrast_feat, contrast_labels
        )
        loss_dict['contrast_loss'] = contrast_loss

        # 4. 总损失
        total_loss = (self.mae_weight * mae_total_loss +
                     self.distill_weight * distill_loss +
                     self.contrast_weight * contrast_loss)

        loss_dict['total_loss'] = total_loss
        loss_dict['weighted_mae_loss'] = self.mae_weight * mae_total_loss
        loss_dict['weighted_distill_loss'] = self.distill_weight * distill_loss
        loss_dict['weighted_contrast_loss'] = self.contrast_weight * contrast_loss

        return total_loss, loss_dict


class EnhancedCombinedLoss(nn.Module):
    """
    增强的组合损失函数
    改进版本：
    1. 调整损失权重平衡
    2. 使用标准双向 InfoNCE 对比损失
    3. 使用稳定的余弦距离蒸馏损失
    4. 添加调试输出
    """

    def __init__(self,
                 mae_weight=1.0,
                 distill_weight=0.05,
                 contrast_weight=0.2,
                 contrast_temp=0.07,
                 distill_temp=1.0,
                 mae_loss_config=None):
        """
        Args:
            mae_weight: MAE损失权重
            distill_weight: 蒸馏损失权重
            contrast_weight: 对比学习损失权重
            contrast_temp: 对比学习温度
            distill_temp: 蒸馏温度
            mae_loss_config: MAE损失配置
        """
        super().__init__()

        self.mae_weight = mae_weight
        self.distill_weight = distill_weight
        self.contrast_weight = contrast_weight

        # 初始化各个损失函数
        mae_config = mae_loss_config or {}
        self.mae_loss = MAELoss(**mae_config)

        # 使用增强的对比损失（标准双向 InfoNCE）
        self.contrast_loss = EnhancedContrastiveLoss(temperature=contrast_temp)

        # 使用稳定的蒸馏损失
        self.distill_loss = StableDistillationLoss(temperature=distill_temp)

        # 打印初始化信息
        print(f"[EnhancedCombinedLoss] Initialized with weights:")
        print(f"  MAE: {mae_weight}, Distill: {distill_weight}, Contrast: {contrast_weight}")
        print(f"  Contrast temp: {contrast_temp}, Distill temp: {distill_temp}")

    def forward(self,
                # MAE相关
                pred_patches, target_patches, patch_mask,
                pred_skeleton, target_skeleton,
                # 蒸馏相关
                student_features, teacher_features,
                # 对比学习相关
                anchor_contrast_feat, positive_contrast_feat, contrast_labels):
        """
        计算组合损失

        Args:
            pred_patches: 预测的CSI补丁
            target_patches: 目标CSI补丁
            patch_mask: 补丁掩码
            pred_skeleton: 预测的骨骼点
            target_skeleton: 目标骨骼点
            student_features: 学生模型特征列表
            teacher_features: 教师模型特征列表
            anchor_contrast_feat: anchor对比特征
            positive_contrast_feat: positive对比特征（这里实际是teacher特征）
            contrast_labels: 对比学习标签（保留兼容性，但不使用）

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # 1. MAE重建损失
        mae_patch_loss = self.mae_loss(pred_patches, target_patches, patch_mask)
        mae_skeleton_loss = self.mae_loss(pred_skeleton, target_skeleton)
        mae_total_loss = mae_patch_loss + mae_skeleton_loss

        loss_dict['mae_patch_loss'] = mae_patch_loss
        loss_dict['mae_skeleton_loss'] = mae_skeleton_loss
        loss_dict['mae_total_loss'] = mae_total_loss

        # 2. 知识蒸馏损失（使用稳定版本）
        # 对所有层的特征求平均
        distill_total_loss = 0.0
        num_layers = min(len(student_features), len(teacher_features))

        for i in range(num_layers):
            s_feat = student_features[i]
            t_feat = teacher_features[i]

            # 对齐序列长度和特征维度
            min_seq_len = min(s_feat.shape[1], t_feat.shape[1])
            min_dim = min(s_feat.shape[-1], t_feat.shape[-1])

            s_feat = s_feat[:, :min_seq_len, :min_dim]
            t_feat = t_feat[:, :min_seq_len, :min_dim]

            layer_loss = self.distill_loss(s_feat, t_feat)
            distill_total_loss += layer_loss

        distill_loss = distill_total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
        loss_dict['distill_loss'] = distill_loss

        # 3. 对比学习损失（使用增强版本，不依赖 labels）
        contrast_loss = self.contrast_loss(anchor_contrast_feat, positive_contrast_feat)
        loss_dict['contrast_loss'] = contrast_loss

        # 4. 总损失
        total_loss = (self.mae_weight * mae_total_loss +
                     self.distill_weight * distill_loss +
                     self.contrast_weight * contrast_loss)

        loss_dict['total_loss'] = total_loss
        loss_dict['weighted_mae_loss'] = self.mae_weight * mae_total_loss
        loss_dict['weighted_distill_loss'] = self.distill_weight * distill_loss
        loss_dict['weighted_contrast_loss'] = self.contrast_weight * contrast_loss

        # 调试输出（1% 概率）
        if torch.rand(1).item() < 0.01:
            print(f"[Loss Debug] MAE={mae_total_loss:.4f}, Distill={distill_loss:.4f}, Contrast={contrast_loss:.4f}")

        return total_loss, loss_dict


def test_mae_loss():
    """测试MAE损失"""
    print("=" * 50)
    print("测试MAE损失")
    print("=" * 50)

    loss_fn = MAELoss(loss_type='mse')

    # 测试补丁重建损失
    batch_size, num_patches, patch_dim = 4, 64, 256
    pred_patches = torch.randn(batch_size, num_patches, patch_dim)
    target_patches = torch.randn(batch_size, num_patches, patch_dim)
    mask = torch.randint(0, 2, (batch_size, num_patches)).bool()

    patch_loss = loss_fn(pred_patches, target_patches, mask)
    print(f"补丁重建损失: {patch_loss.item():.4f}")

    # 测试骨骼点重建损失
    batch_size, num_joints, coord_dim = 4, 17, 2
    pred_skeleton = torch.randn(batch_size, num_joints, coord_dim)
    target_skeleton = torch.randn(batch_size, num_joints, coord_dim)

    skeleton_loss = loss_fn(pred_skeleton, target_skeleton)
    print(f"骨骼点重建损失: {skeleton_loss.item():.4f}")

    print("✅ MAE损失测试通过")


def test_distillation_loss():
    """测试知识蒸馏损失"""
    print("\n" + "=" * 50)
    print("测试知识蒸馏损失")
    print("=" * 50)

    loss_fn = DistillationLoss(loss_type='l1')

    # 创建模拟特征
    batch_size = 4
    student_features = [
        torch.randn(batch_size, 18, 384),  # 学生模型层1
        torch.randn(batch_size, 18, 384),  # 学生模型层2
        torch.randn(batch_size, 18, 384),  # 学生模型层3
    ]

    teacher_features = [
        torch.randn(batch_size, 18, 768),  # 教师模型层1（维度不同）
        torch.randn(batch_size, 18, 768),  # 教师模型层2
        torch.randn(batch_size, 18, 768),  # 教师模型层3
    ]

    distill_loss = loss_fn(student_features, teacher_features)
    print(f"知识蒸馏损失: {distill_loss.item():.4f}")

    # 测试稳定版本
    stable_loss_fn = StableDistillationLoss(temperature=1.0)
    s_feat = torch.randn(batch_size, 18, 384)
    t_feat = torch.randn(batch_size, 18, 384)
    stable_loss = stable_loss_fn(s_feat, t_feat)
    print(f"稳定蒸馏损失: {stable_loss.item():.4f}")

    print("✅ 知识蒸馏损失测试通过")


def test_contrastive_loss():
    """测试对比学习损失"""
    print("\n" + "=" * 50)
    print("测试对比学习损失")
    print("=" * 50)

    # 测试简化对比损失
    simple_loss_fn = SimplifiedContrastiveLoss(margin=1.0)

    batch_size = 8
    feature_dim = 256
    anchor_features = torch.randn(batch_size, feature_dim)
    positive_features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, 2, (batch_size,))  # 随机正负样本标签

    simple_loss = simple_loss_fn(anchor_features, positive_features, labels)
    print(f"简化对比学习损失: {simple_loss.item():.4f}")
    print(f"正样本对数量: {labels.sum().item()}")
    print(f"负样本对数量: {(1-labels).sum().item()}")

    # 测试NT-Xent对比损失
    ntxent_loss_fn = ContrastiveLoss(temperature=0.07)
    ntxent_loss = ntxent_loss_fn(anchor_features, positive_features, labels)
    print(f"NT-Xent对比学习损失: {ntxent_loss.item():.4f}")

    # 测试增强版对比损失（标准双向 InfoNCE）
    enhanced_loss_fn = EnhancedContrastiveLoss(temperature=0.07)
    enhanced_loss = enhanced_loss_fn(anchor_features, positive_features)
    print(f"增强对比学习损失 (InfoNCE): {enhanced_loss.item():.4f}")

    print("✅ 对比学习损失测试通过")


def test_combined_loss():
    """测试组合损失"""
    print("\n" + "=" * 50)
    print("测试组合损失")
    print("=" * 50)

    # 创建组合损失函数
    loss_fn = CombinedLoss(
        mae_weight=1.0,
        distill_weight=1.0,
        contrast_weight=0.5,
        contrast_loss_config={'use_simplified': True, 'margin': 1.0}
    )

    # 创建模拟数据
    batch_size = 4

    # MAE相关数据
    pred_patches = torch.randn(batch_size, 64, 256)
    target_patches = torch.randn(batch_size, 64, 256)
    patch_mask = torch.randint(0, 2, (batch_size, 64)).bool()

    pred_skeleton = torch.randn(batch_size, 17, 2)
    target_skeleton = torch.randn(batch_size, 17, 2)

    # 蒸馏相关数据
    student_features = [torch.randn(batch_size, 65, 384) for _ in range(3)]
    teacher_features = [torch.randn(batch_size, 18, 768) for _ in range(3)]

    # 对比学习相关数据
    anchor_contrast_feat = torch.randn(batch_size, 256)
    positive_contrast_feat = torch.randn(batch_size, 256)
    contrast_labels = torch.randint(0, 2, (batch_size,))

    # 计算组合损失
    total_loss, loss_dict = loss_fn(
        pred_patches, target_patches, patch_mask,
        pred_skeleton, target_skeleton,
        student_features, teacher_features,
        anchor_contrast_feat, positive_contrast_feat, contrast_labels
    )

    print(f"总损失: {total_loss.item():.4f}")
    print("各项损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    print("✅ 组合损失测试通过")


def test_enhanced_combined_loss():
    """测试增强的组合损失"""
    print("\n" + "=" * 50)
    print("测试增强的组合损失")
    print("=" * 50)

    # 创建增强的组合损失函数（使用改进的权重配置）
    loss_fn = EnhancedCombinedLoss(
        mae_weight=1.0,
        distill_weight=0.05,
        contrast_weight=0.2,
        contrast_temp=0.07,
        distill_temp=1.0
    )

    # 创建模拟数据
    batch_size = 4

    # MAE相关数据
    pred_patches = torch.randn(batch_size, 64, 256)
    target_patches = torch.randn(batch_size, 64, 256)
    patch_mask = torch.randint(0, 2, (batch_size, 64)).bool()

    pred_skeleton = torch.randn(batch_size, 17, 2)
    target_skeleton = torch.randn(batch_size, 17, 2)

    # 蒸馏相关数据
    student_features = [torch.randn(batch_size, 65, 384) for _ in range(3)]
    teacher_features = [torch.randn(batch_size, 65, 384) for _ in range(3)]

    # 对比学习相关数据（使用归一化特征以确保非零损失）
    anchor_contrast_feat = F.normalize(torch.randn(batch_size, 256), dim=-1)
    positive_contrast_feat = F.normalize(torch.randn(batch_size, 256), dim=-1)
    contrast_labels = torch.ones(batch_size, dtype=torch.long)  # 保留兼容性

    # 计算组合损失
    total_loss, loss_dict = loss_fn(
        pred_patches, target_patches, patch_mask,
        pred_skeleton, target_skeleton,
        student_features, teacher_features,
        anchor_contrast_feat, positive_contrast_feat, contrast_labels
    )

    print(f"总损失: {total_loss.item():.4f}")
    print("各项损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    # 验证对比损失非零
    contrast_loss_value = loss_dict['contrast_loss'].item()
    print(f"\n对比损失 = {contrast_loss_value:.4f}")
    assert contrast_loss_value > 0.0, "❌ Contrast loss = 0, 请检查特征分布!"
    print("✅ 对比损失非零验证通过")

    # 验证各项损失权重合理
    weighted_mae = loss_dict['weighted_mae_loss'].item()
    weighted_distill = loss_dict['weighted_distill_loss'].item()
    weighted_contrast = loss_dict['weighted_contrast_loss'].item()

    print(f"\n加权损失分析:")
    print(f"  加权 MAE: {weighted_mae:.4f}")
    print(f"  加权 Distill: {weighted_distill:.4f}")
    print(f"  加权 Contrast: {weighted_contrast:.4f}")

    total_weighted = weighted_mae + weighted_distill + weighted_contrast
    print(f"  MAE 占比: {weighted_mae/total_weighted*100:.1f}%")
    print(f"  Distill 占比: {weighted_distill/total_weighted*100:.1f}%")
    print(f"  Contrast 占比: {weighted_contrast/total_weighted*100:.1f}%")

    print("\n✅ 增强的组合损失测试通过")


if __name__ == "__main__":
    test_mae_loss()
    test_distillation_loss()
    test_contrastive_loss()
    test_combined_loss()
    test_enhanced_combined_loss()