"""
CSI数据预处理模块 - 修复维度问题
将CSI数据转换为时频谱图并进行补丁划分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
import math


class CSIPreprocessor(nn.Module):
    """
    CSI数据预处理器
    将原始CSI数据转换为时频谱图，并划分为补丁用于Transformer处理
    """

    def __init__(self,
                 num_antennas=3,
                 num_subcarriers=30,
                 time_length=297,
                 stft_window=32,
                 stft_hop=16,
                 patch_size=16,
                 normalize=True):
        """
        Args:
            num_antennas: 天线数量
            num_subcarriers: 子载波数量
            time_length: 时间序列长度
            stft_window: STFT窗口大小
            stft_hop: STFT跳跃步长
            patch_size: 补丁大小
            normalize: 是否归一化
        """
        super().__init__()
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.time_length = time_length
        self.stft_window = stft_window
        self.stft_hop = stft_hop
        self.patch_size = patch_size
        self.normalize = normalize

        # 🔑 修复：需要根据实际输入动态计算维度
        self.freq_bins = None
        self.time_bins = None
        self.patches_per_freq = None
        self.patches_per_time = None
        self.num_patches = None
        self.patch_dim = patch_size * patch_size

        print(f"CSI预处理器初始化:")
        print(f"  输入维度: [{num_antennas}, {num_subcarriers}, {time_length}]")
        print(f"  补丁大小: {patch_size}")
        print(f"  补丁维度: {self.patch_dim}")
        print(f"  ⚠️  实际维度将在第一次前向传播时确定")

    def _initialize_dimensions(self, freq_bins, time_bins):
        """根据实际输入初始化维度"""
        if self.freq_bins is None:
            self.freq_bins = freq_bins
            self.time_bins = time_bins

            # 计算补丁数量
            self.patches_per_freq = self.freq_bins // self.patch_size
            self.patches_per_time = self.time_bins // self.patch_size
            self.num_patches = self.patches_per_freq * self.patches_per_time * self.num_antennas

            print(f"\n✅ 维度已确定:")
            print(f"  频率bins: {self.freq_bins}")
            print(f"  时间bins: {self.time_bins}")
            print(f"  补丁数量: {self.num_patches} ({self.num_antennas} antennas × {self.patches_per_freq} freq × {self.patches_per_time} time)")


    def csi_to_spectrogram(self, csi_data):
        """
        统一CSI输入格式: [B, antennas, freq, time]
        如果数据已是频域，不做STFT，保持原语义。
        只有当输入是时域数据时才执行 STFT。
        """
        import torch.nn.functional as F
        x = csi_data
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        B = x.shape[0]
        shape = list(x.shape)

        # --- 1. 自动识别维度，统一到 [B, num_antennas, freq, time] ---
        if len(shape) == 4 and shape[1] == self.num_antennas:
            if shape[2] == self.num_subcarriers:  # [B, 3, 114, 10]
                spect = x
            elif shape[3] == self.num_subcarriers:  # [B, 3, 10, 114]
                spect = x.permute(0, 1, 3, 2)
            else:
                spect = x
        elif len(shape) == 4 and shape[2] == self.num_antennas:  # [B, time, 3, freq]
            spect = x.permute(0, 2, 3, 1)
        elif len(shape) == 4 and shape[3] == self.num_antennas:  # [B, time, freq, 3]
            spect = x.permute(0, 3, 2, 1)
        else:
            raise ValueError(f"Unexpected CSI shape {shape}")

        # --- 2. 判断是否已是频域数据 ---
        _, _, dim2, dim3 = spect.shape
        is_freq_input = (dim2 == self.num_subcarriers)
        if is_freq_input:
            return spect, {'is_freq_input': True, 'freq_bins': dim2, 'time_bins': dim3}

        # --- 3. 否则执行 STFT（仅限时域输入） ---
        B, A, Freq, Time = spect.shape
        n_fft = self.stft_window
        hop = self.stft_hop
        stft_in = spect.reshape(B * A, Time)
        stft_res = torch.stft(stft_in, n_fft=n_fft, hop_length=hop,
                              return_complex=True, center=False)
        stft_mag = torch.abs(stft_res).reshape(B, A, stft_res.shape[1], stft_res.shape[2])
        return stft_mag, {'is_freq_input': False, 'freq_bins': stft_res.shape[1], 'time_bins': stft_res.shape[2]}



    def normalize_spectrogram(self, spectrogram):
        """归一化时频谱图（逐样本逐天线 min-max）"""
        # spectrogram: [B, A, F, T]
        spec_min = spectrogram.amin(dim=(2, 3), keepdim=True)
        spec_max = spectrogram.amax(dim=(2, 3), keepdim=True)
        denom = (spec_max - spec_min).clamp(min=1e-6)
        spectrogram = (spectrogram - spec_min) / denom
        return spectrogram


    def patchify(self, spectrogram):
        """
        把 [B, antennas, freq, time] 转为补丁。
        - 正常情况下不插值
        - 只有极端情况下（freq<patch_size）才最小插值，避免0patch
        """
        import torch.nn.functional as F
        B, A, Freq, Time = spectrogram.shape
        P = self.patch_size

        patches_per_freq = Freq // P
        patches_per_time = Time // P

        # --- 仅在极少数异常时插值 ---
        if patches_per_freq == 0 or patches_per_time == 0:
            print(f"⚠️ Warning: freq={Freq}, time={Time} too small, "
                  f"upsample minimally to patch={P}")
            up_Freq, up_Time = max(P, Freq), max(P, Time)
            spectrogram = F.interpolate(spectrogram.reshape(B * A, 1, Freq, Time),
                                        size=(up_Freq, up_Time),
                                        mode='bilinear', align_corners=False
                                        ).reshape(B, A, up_Freq, up_Time)
            Freq, Time = up_Freq, up_Time
            patches_per_freq = Freq // P
            patches_per_time = Time // P

        # --- 裁剪到整除区域 ---
        Freq_crop = patches_per_freq * P
        Time_crop = patches_per_time * P
        spectrogram = spectrogram[:, :, :Freq_crop, :Time_crop]

        # --- 划分 patch ---
        spectrogram = spectrogram.reshape(B, A, patches_per_freq, P, patches_per_time, P)
        spectrogram = spectrogram.permute(0, 1, 2, 4, 3, 5)
        patches = spectrogram.reshape(B, A * patches_per_freq * patches_per_time, P * P)

        # --- 保存 patch 信息 ---
        self.patch_grid = (patches_per_freq, patches_per_time)
        self.num_patches = A * patches_per_freq * patches_per_time
        return patches

    def unpatchify(self, patches):
        """
        将补丁重构为时频谱图

        Args:
            patches: [batch_size, num_patches, patch_dim]

        Returns:
            spectrogram: [batch_size, num_antennas, freq_bins, time_bins]
        """
        batch_size = patches.shape[0]

        # 🔑 修复：使用实际计算的维度
        if self.patches_per_freq is None or self.patches_per_time is None:
            raise ValueError("必须先调用 patchify 初始化维度")

        # 重塑补丁
        patches = patches.reshape(
            batch_size,
            self.num_antennas,
            self.patches_per_freq,
            self.patches_per_time,
            self.patch_size,
            self.patch_size
        )

        # 调整维度顺序
        patches = patches.permute(0, 1, 2, 4, 3, 5)

        # 重构时频谱图
        spectrogram = patches.reshape(
            batch_size,
            self.num_antennas,
            self.patches_per_freq * self.patch_size,
            self.patches_per_time * self.patch_size
        )

        return spectrogram


    def forward(self, x):
        """
        CSI数据预处理前向传播。
        仅在第一次运行时打印形状信息，以免打断 tqdm 进度条。
        """
        # Step 1: 统一格式并生成时频谱
        spectrogram, meta = self.csi_to_spectrogram(x)

        # Step 2: 生成补丁
        patches = self.patchify(spectrogram)

        # Step 3: 正则化（如果配置中启用）
        if self.normalize:
            mean = patches.mean(dim=-1, keepdim=True)
            std = patches.std(dim=-1, keepdim=True) + 1e-6
            patches = (patches - mean) / std

        # ---- 打印信息（只在第一次 forward 时输出） ----
        if not hasattr(self, "_printed_once"):
            print(f"\n🔄 CSI预处理 - 输入形状: {x.shape}")
            print(f"  时频谱形状: {spectrogram.shape}")
            print(f"✅ 最终patches形状: {patches.shape}")
            if meta.get('is_freq_input', False):
                print(f"  ⚙️ 检测到频域输入: freq_bins={meta['freq_bins']}, time_bins={meta['time_bins']}")
            else:
                print(f"  ⚙️ STFT转换完成: freq_bins={meta['freq_bins']}, time_bins={meta['time_bins']}")
            self._printed_once = True  # 标记已打印

        return patches, meta


class SkeletonPreprocessor(nn.Module):
    """
    骨骼点数据预处理器
    处理RGB骨骼点数据，进行归一化和序列化
    """

    def __init__(self, num_joints=17, coord_dim=2, normalize=True):
        """
        Args:
            num_joints: 关节点数量
            coord_dim: 坐标维度 (2D或3D)
            normalize: 是否归一化
        """
        super().__init__()
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.normalize = normalize

        # 序列长度
        self.seq_len = num_joints
        self.feature_dim = coord_dim

    def normalize_skeleton(self, skeleton):
        """
        归一化骨骼点坐标

        Args:
            skeleton: [batch_size, num_joints, coord_dim]

        Returns:
            normalized_skeleton: [batch_size, num_joints, coord_dim]
        """
        if not self.normalize:
            return skeleton

        batch_size = skeleton.shape[0]
        normalized = skeleton.clone()

        for b in range(batch_size):
            skel = skeleton[b]  # [num_joints, coord_dim]

            # 找到有效关节点（非零点）
            valid_mask = (skel != 0).any(dim=1)
            if valid_mask.sum() > 0:
                valid_joints = skel[valid_mask]

                # 计算边界框
                min_coords = valid_joints.min(dim=0)[0]
                max_coords = valid_joints.max(dim=0)[0]

                # 归一化到[0, 1]
                range_coords = max_coords - min_coords
                range_coords[range_coords == 0] = 1  # 避免除零

                normalized[b] = (skel - min_coords) / range_coords

        return normalized

    def forward(self, skeleton):
        """
        完整的骨骼点预处理流程

        Args:
            skeleton: [batch_size, num_joints, coord_dim]

        Returns:
            processed_skeleton: [batch_size, num_joints, feature_dim]
        """
        # 归一化
        skeleton = self.normalize_skeleton(skeleton)

        return skeleton


def create_random_mask(seq_len, mask_ratio=0.75):
    """
    创建随机掩码

    Args:
        seq_len: 序列长度
        mask_ratio: 掩码比例

    Returns:
        mask: [seq_len] 布尔掩码，True表示被掩码
        ids_restore: [seq_len] 恢复原始顺序的索引
    """
    len_keep = int(seq_len * (1 - mask_ratio))

    noise = torch.rand(seq_len)
    ids_shuffle = torch.argsort(noise)
    ids_restore = torch.argsort(ids_shuffle)

    # 保留前len_keep个，其余掩码
    mask = torch.ones(seq_len, dtype=torch.bool)
    mask[:len_keep] = False
    mask = mask[ids_shuffle]

    return mask, ids_restore


def test_csi_preprocessor():
    """测试CSI预处理器"""
    print("=" * 50)
    print("测试CSI预处理器")
    print("=" * 50)

    # 测试1: MMFi格式数据（已经是频域）
    print("\n测试1: MMFi频域数据")
    batch_size = 2
    freq_bins = 128  # MMFi的频率bins
    time_bins = 32   # MMFi的时间bins
    num_antennas = 3

    # 模拟MMFi格式: [batch, freq, time, antennas]
    csi_mmfi = torch.randn(batch_size, freq_bins, time_bins, num_antennas)
    print(f"输入MMFi数据形状: {csi_mmfi.shape}")

    preprocessor1 = CSIPreprocessor(
        num_antennas=num_antennas,
        patch_size=16
    )

    patches1, spectrogram1 = preprocessor1(csi_mmfi)
    print(f"✅ 输出patches: {patches1.shape}")

    # 测试2: 原始CSI数据（需要STFT）
    print("\n测试2: 原始CSI数据")
    num_subcarriers = 30
    time_length = 297

    csi_raw = torch.randn(batch_size, num_antennas, num_subcarriers, time_length)
    print(f"输入原始CSI形状: {csi_raw.shape}")

    preprocessor2 = CSIPreprocessor(
        num_antennas=num_antennas,
        num_subcarriers=num_subcarriers,
        time_length=time_length,
        patch_size=8
    )

    patches2, spectrogram2 = preprocessor2(csi_raw)
    print(f"✅ 输出patches: {patches2.shape}")


def test_skeleton_preprocessor():
    """测试骨骼点预处理器"""
    print("\n" + "=" * 50)
    print("测试骨骼点预处理器")
    print("=" * 50)

    batch_size = 2
    num_joints = 17
    coord_dim = 2

    skeleton_data = torch.randn(batch_size, num_joints, coord_dim) * 100
    print(f"输入骨骼点数据形状: {skeleton_data.shape}")

    preprocessor = SkeletonPreprocessor(
        num_joints=num_joints,
        coord_dim=coord_dim,
        normalize=True
    )

    processed_skeleton = preprocessor(skeleton_data)
    print(f"✅ 输出骨骼点形状: {processed_skeleton.shape}")


if __name__ == "__main__":
    test_csi_preprocessor()
    test_skeleton_preprocessor()