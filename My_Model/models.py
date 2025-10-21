"""
Enhanced Multi-Modal DMAE Models - 完整修复版
增强型多模态DMAE模型实现

修复内容：
1. ✅ 修复 StudentModel 初始化顺序错误
2. ✅ MultiPathAttention 添加安全检查（处理 num_patches 不能整除的情况）
3. ✅ 完全兼容你的训练框架（RGB骨骼点 + CSI信号）
4. ✅ 一键开关：use_multi_attn=True/False
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from timm.models.vision_transformer import Block, Mlp

try:
    from .data_processing import create_random_mask
except ImportError:
    try:
        from data_processing import create_random_mask
    except ImportError:
        def create_random_mask(seq_len, mask_ratio=0.75):
            mask_len = int(seq_len * mask_ratio)
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask_indices = torch.randperm(seq_len)[:mask_len]
            mask[mask_indices] = True
            return mask


def get_1d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    """生成1D正弦余弦位置编码"""
    pos = np.arange(seq_len, dtype=np.float32)

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)

    return emb


class SkeletonEmbedding(nn.Module):
    """骨骼点嵌入层"""

    def __init__(self, num_joints=17, coord_dim=2, embed_dim=768):
        super().__init__()
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim

        self.proj = nn.Linear(coord_dim, embed_dim)
        self.joint_embed = nn.Parameter(torch.zeros(num_joints, embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.joint_embed, std=.02)
        torch.nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            torch.nn.init.constant_(self.proj.bias, 0)

    def forward(self, skeleton):
        batch_size = skeleton.shape[0]
        embeddings = self.proj(skeleton)
        joint_embeds = self.joint_embed.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = embeddings + joint_embeds
        return embeddings


class MultiPathAttention(nn.Module):
    """
    多路径注意力模块 - 完整修复版本 v2

    关键修复：
    1. ✅ 正确处理 CLS token（不参与天线分组）
    2. ✅ 验证 patch tokens 能被 num_antennas 整除
    3. ✅ 保持序列长度不变（展平天线维度而非取平均）
    4. ✅ 自动降级为标准注意力（如果条件不满足）

    工作原理：
    - 将 patch tokens 按天线分组
    - 对每个天线独立计算注意力
    - 展平天线维度重新组合（保持原始序列长度）
    """
    def __init__(self, embed_dim, num_heads=8, num_antennas=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_antennas = num_antennas

        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # QKV投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, antenna_aware=False):
        """
        Args:
            x: [batch, seq_len, embed_dim] (包含CLS token)
            antenna_aware: 是否使用多路径注意力
        """
        B, N, C = x.shape

        # === 安全检查：是否可以使用多路径注意力 ===
        # ✅ 新增：分离CLS token和patch tokens
        has_cls = N > 1  # 假设第一个token是CLS
        effective_N = N - 1 if has_cls else N  # 实际的patch数量

        can_use_multipath = (
                antenna_aware and
                self.num_antennas > 1 and
                effective_N > 0 and
                effective_N % self.num_antennas == 0  # ✅ 改为检查 effective_N
        )

        if can_use_multipath:
            patches_per_antenna = effective_N // self.num_antennas  # ✅ 使用 effective_N

            # 额外检查：每个天线的补丁数必须 > 0
            if patches_per_antenna == 0:
                can_use_multipath = False


        if can_use_multipath:
            # === 多路径注意力模式 ===
            # ✅ 新增：分离CLS token
            cls_token = x[:, :1, :]  # [B, 1, C]
            patch_tokens = x[:, 1:, :]  # [B, effective_N, C]

            patches_per_antenna = effective_N // self.num_antennas
            patch_tokens = patch_tokens.view(B, self.num_antennas, patches_per_antenna, C)  # ✅ 只处理patch_tokens

            attn_per_antenna = []
            for ant in range(self.num_antennas):
                ant_feat = patch_tokens[:, ant]  # [B, patches_per_antenna, C]  # ✅ 改为 patch_tokens

                qkv = self.qkv(ant_feat).reshape(
                    B, patches_per_antenna, 3, self.num_heads, self.head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                ant_out = (attn @ v).transpose(1, 2).reshape(B, patches_per_antenna, C)
                attn_per_antenna.append(ant_out)

            # ✅ 修复：跨天线融合 - 展平天线维度（保持序列长度）
            patch_tokens = torch.stack(attn_per_antenna, dim=1)  # [B, num_antennas, patches_per_antenna, C]
            patch_tokens = patch_tokens.reshape(B, effective_N, C)  # [B, effective_N, C]

            # ✅ 新增：重新拼接CLS token
            x = torch.cat([cls_token, patch_tokens], dim=1)  # [B, N, C]

        else:
            # === 标准注意力模式 ===
            # if antenna_aware and effective_N > 0:  # ✅ 改为 effective_N
            #     divisible = effective_N % self.num_antennas == 0  # ✅ 改为 effective_N
            #     print(
            #         f"⚠️  MultiPath disabled: N={N}, effective_N={effective_N}, antennas={self.num_antennas}, divisible={divisible}")  # ✅ 更详细的日志

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TeacherModel(nn.Module):
    """
    老师模型：RGB骨骼点MAE预训练
    输入：RGB骨骼点 [batch, 17, 2]
    输出：loss, pred, mask
    """

    def __init__(self,
                 num_joints=17,
                 coord_dim=2,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm,
                 mask_ratio=0.75):
        super().__init__()

        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # 骨骼点嵌入
        self.skeleton_embed = SkeletonEmbedding(num_joints, coord_dim, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_joints + 1, embed_dim),
            requires_grad=False
        )

        # Transformer编码器
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # MAE解码器
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, coord_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.num_joints, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.num_joints, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        batch_size, seq_len, embed_dim = x.shape
        len_keep = int(seq_len * (1 - mask_ratio))

        noise = torch.rand(batch_size, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, embed_dim))

        mask = torch.ones([batch_size, seq_len], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, skeleton, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        x = self.skeleton_embed(skeleton)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, skeleton, pred, mask):
        target = skeleton
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, skeleton, mask_ratio=None):
        latent, mask, ids_restore = self.forward_encoder(skeleton, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(skeleton, pred, mask)
        return loss, pred, mask

    def forward_features(self, skeleton, mask_ratio=None):
        """提取特征（用于知识蒸馏）"""
        x = self.skeleton_embed(skeleton)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio is not None and mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x)

        x = self.norm(x)
        features.append(x)

        return features


class PatchEmbedding(nn.Module):
    """补丁嵌入层"""

    def __init__(self, patch_dim, embed_dim):
        super().__init__()
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, patches):
        return self.proj(patches)


class StudentModel(nn.Module):
    """
    学生模型：CSI多分支学习 - 完整修复版

    修复内容：
    1. ✅ 修复初始化顺序（decoder_pos_embed 在 initialize_weights 之前定义）
    2. ✅ 添加多径注意力安全检查
    3. ✅ 完全兼容训练框架
    """

    def __init__(self,
                 num_patches,
                 patch_dim,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_joints=17,
                 coord_dim=2,
                 contrast_dim=256,
                 mask_ratio=0.75,
                 num_antennas=3,
                 use_multi_attn=True):
        super().__init__()

        # === ✅ 新增：支持延迟初始化 ===
        if num_patches is None:
            raise ValueError(
                "StudentModel requires num_patches to be specified. "
                "Please run CSIPreprocessor on a sample input first to determine num_patches, "
                "then create StudentModel with that value."
            )

        # === 1. 基础参数 ===
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.mask_ratio = mask_ratio
        self.num_antennas = num_antennas
        self.use_multi_attn = use_multi_attn

        # print(f"StudentModel: num_patches={num_patches}, use_multi_attn={use_multi_attn}")

        # === 2. CSI编码器组件 ===
        self.patch_embed = PatchEmbedding(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )

        # Transformer编码器
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # === 3. 分支1: CSI重建解码器（关键：所有参数必须在initialize_weights之前定义）===
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # ✅ 关键修复：decoder_pos_embed 必须在这里定义
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)

        # === 4. 分支2-4: 预测头 ===
        self.skeleton_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_joints * coord_dim)
        )

        self.distill_proj = nn.Linear(embed_dim, embed_dim)

        self.contrast_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, contrast_dim)
        )

        # === 5. 多径注意力（可选，在所有必需参数定义之后）===
        if use_multi_attn:
            self.multi_attn = MultiPathAttention(embed_dim, num_heads=8, num_antennas=num_antennas)
        #     print(f"  MultiPath Attention enabled for {num_antennas} antennas")
        # else:
        #     print(f"  Using standard attention")

        # === 6. 最后初始化权重 ===
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.num_patches, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        batch_size, seq_len, embed_dim = x.shape
        len_keep = int(seq_len * (1 - mask_ratio))

        noise = torch.rand(batch_size, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, embed_dim))

        mask = torch.ones([batch_size, seq_len], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, patches, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        x = self.patch_embed(patches)
        # 动态对齐 pos_embed，只在 patch 数小变动时插值
        N_new = x.shape[1]
        N_init = self.pos_embed.shape[1] - 1
        if N_new != N_init:
            rel_diff = abs(N_new - N_init) / max(N_init, 1)
            if rel_diff <= 0.1:
                # print(f"⚠️ pos_embed mismatch {N_init}->{N_new}, interpolate safely...")
                cls_tok = self.pos_embed[:, :1, :]
                patch_pos = self.pos_embed[:, 1:, :].reshape(1, int(N_init ** 0.5),
                                                             int(N_init ** 0.5), -1).permute(0, 3, 1, 2)
                patch_interp = torch.nn.functional.interpolate(
                    patch_pos, size=(int(N_new ** 0.5), int(N_new ** 0.5)),
                    mode='bilinear', align_corners=False)
                patch_interp = patch_interp.permute(0, 2, 3, 1).reshape(1, N_new, -1)
                pos_embed = torch.cat([cls_tok, patch_interp], dim=1)
            else:
                raise RuntimeError(f"❌ pos_embed patch count {N_new} vs {N_init}, check data pipeline.")
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed[:, 1:, :]


        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x)

        x = self.norm(x)

        # 多径注意力融合（可选）
        if self.use_multi_attn:
            x = self.multi_attn(x, antenna_aware=True)

        features.append(x)

        return x, mask, ids_restore, features

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward(self, patches, mask_ratio=None):
        """
        完整前向传播

        Args:
            patches: [batch_size, num_patches, patch_dim]

        Returns:
            outputs: 包含4个分支的字典
        """
        latent, mask, ids_restore, features = self.forward_encoder(patches, mask_ratio)

        # 分支1: CSI重建
        reconstructed_patches = self.forward_decoder(latent, ids_restore)

        # 分支2: 骨骼点预测
        cls_token = latent[:, 0]
        skeleton_pred = self.skeleton_head(cls_token)
        skeleton_pred = skeleton_pred.reshape(-1, self.num_joints, self.coord_dim)

        # 分支3: 知识蒸馏特征
        distill_features = [self.distill_proj(feat) for feat in features]

        # 分支4: 对比学习特征
        contrast_features = self.contrast_proj(cls_token)

        outputs = {
            'reconstructed_patches': reconstructed_patches,
            'skeleton_pred': skeleton_pred,
            'distill_features': distill_features,
            'contrast_features': contrast_features,
            'mask': mask,
            'cls_token': cls_token
        }

        return outputs


def test_teacher_model():
    print("=" * 50)
    print("测试老师模型")
    print("=" * 50)

    model = TeacherModel(embed_dim=384, depth=6, num_heads=6)
    skeleton = torch.randn(4, 17, 2)

    loss, pred, mask = model(skeleton)
    print(f"✅ Teacher: loss={loss.item():.4f}, pred={pred.shape}, mask={mask.shape}")

    features = model.forward_features(skeleton, mask_ratio=0.0)
    print(f"✅ Features: {len(features)} layers")


def test_student_model():
    print("\n" + "=" * 50)
    print("测试学生模型")
    print("=" * 50)

    # 测试不同的num_patches配置
    test_configs = [
        (60, 64, 3, "60 patches, 3 antennas (divisible)"),
        (64, 64, 3, "64 patches, 3 antennas (not divisible, will fall back)"),
        (0, 64, 3, "0 patches (edge case)"),
    ]

    for num_patches, patch_dim, num_antennas, desc in test_configs:
        print(f"\n测试配置: {desc}")
        try:
            model = StudentModel(
                num_patches=num_patches,
                patch_dim=patch_dim,
                embed_dim=256,
                depth=4,
                num_heads=4,
                num_antennas=num_antennas,
                use_multi_attn=True
            )

            if num_patches > 0:
                patches = torch.randn(2, num_patches, patch_dim)
                outputs = model(patches)
                print(f"✅ Student: skeleton_pred={outputs['skeleton_pred'].shape}")
            else:
                print(f"⚠️  Skipped (num_patches=0)")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_teacher_model()
    test_student_model()

