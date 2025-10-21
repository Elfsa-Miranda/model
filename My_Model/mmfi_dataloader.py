"""
MMFi数据加载器 - 专门用于Enhanced Multi-Modal DMAE
整合MMFi_dataset中的数据处理逻辑，为My_Model提供RGB骨骼点和CSI数据
"""

import os
import sys
import torch
import numpy as np
import scipy.io as scio
import glob
from torch.utils.data import Dataset, DataLoader

# 添加MMFi_dataset路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MMFi_dataset'))

from mmfi_lib.mmfi import MMFi_Database, decode_config


class EnhancedMMFiDataset(Dataset):
    """
    增强型MMFi数据集，专门用于Enhanced Multi-Modal DMAE
    提供RGB骨骼点和CSI数据的配对
    """
    
    def __init__(self, data_root, config, is_training=True):
        """
        Args:
            data_root: MMFi数据集根目录
            config: 配置字典
            is_training: 是否为训练模式
        """
        self.data_root = data_root
        self.config = config
        self.is_training = is_training
        
        # 创建MMFi数据库
        self.database = MMFi_Database(data_root)
        
        # 解码配置
        dataset_config = decode_config(config)
        
        # 选择训练或验证配置
        if is_training:
            self.split_config = dataset_config['train_dataset']
        else:
            self.split_config = dataset_config['val_dataset']
        
        # 确保使用rgb和wifi-csi模态
        self.modality = ['rgb', 'wifi-csi']
        self.data_unit = config.get('data_unit', 'frame')
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        print(f"✅ Enhanced MMFi Dataset {'训练' if is_training else '验证'}集加载完成")
        print(f"   数据样本数: {len(self.data_list)}")
        print(f"   数据单位: {self.data_unit}")
        print(f"   模态: {self.modality}")
    
    def _load_data_list(self):
        """加载数据列表"""
        data_list = []
        data_form = self.split_config['data_form']
        
        for subject, actions in data_form.items():
            for action in actions:
                scene = self._get_scene(subject)
                
                if self.data_unit == 'sequence':
                    # 序列数据模式
                    data_dict = {
                        'scene': scene,
                        'subject': subject,
                        'action': action,
                        'gt_path': os.path.join(self.data_root, scene, subject, action, 'ground_truth.npy'),
                        'rgb_path': os.path.join(self.data_root, scene, subject, action, 'rgb'),
                        'csi_path': os.path.join(self.data_root, scene, subject, action, 'wifi-csi')
                    }
                    
                    # 检查路径是否存在
                    if (os.path.exists(data_dict['gt_path']) and 
                        os.path.exists(data_dict['rgb_path']) and 
                        os.path.exists(data_dict['csi_path'])):
                        data_list.append(data_dict)
                
                elif self.data_unit == 'frame':
                    # 单帧数据模式
                    frame_num = 297  # MMFi数据集每个动作有297帧
                    
                    for idx in range(frame_num):
                        frame_id = f"frame{idx+1:03d}"
                        
                        data_dict = {
                            'scene': scene,
                            'subject': subject,
                            'action': action,
                            'idx': idx,
                            'gt_path': os.path.join(self.data_root, scene, subject, action, 'ground_truth.npy'),
                            'rgb_path': os.path.join(self.data_root, scene, subject, action, 'rgb', f'{frame_id}.npy'),
                            'csi_path': os.path.join(self.data_root, scene, subject, action, 'wifi-csi', f'{frame_id}.mat')
                        }
                        
                        # 检查文件是否存在且非空
                        if (os.path.exists(data_dict['gt_path']) and 
                            os.path.exists(data_dict['rgb_path']) and 
                            os.path.exists(data_dict['csi_path']) and
                            os.path.getsize(data_dict['rgb_path']) > 0 and
                            os.path.getsize(data_dict['csi_path']) > 0):
                            data_list.append(data_dict)
        
        return data_list
    
    def _get_scene(self, subject):
        """根据受试者获取场景"""
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError(f'Subject {subject} does not exist in this dataset.')
    
    def _load_rgb_sequence(self, rgb_dir):
        """加载RGB骨骼点序列数据"""
        data = []
        for arr_file in sorted(glob.glob(os.path.join(rgb_dir, "frame*.npy"))):
            arr = np.load(arr_file)
            data.append(arr)
        return np.array(data)  # [num_frames, num_joints, coord_dim]
    
    def _load_csi_sequence(self, csi_dir):
        """加载CSI序列数据"""
        data = []
        for csi_mat in sorted(glob.glob(os.path.join(csi_dir, "frame*.mat"))):
            data_mat = scio.loadmat(csi_mat)['CSIamp']
            # 处理无穷值和NaN
            data_mat[np.isinf(data_mat)] = np.nan
            
            # 填充NaN值
            for i in range(data_mat.shape[-1]):  # 遍历天线
                temp_col = data_mat[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    if len(temp_not_nan_col) > 0:
                        temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
            
            # 归一化到[0,1]
            data_min = np.min(data_mat)
            data_max = np.max(data_mat)
            if data_max > data_min:
                data_mat = (data_mat - data_min) / (data_max - data_min)
            
            data.append(data_mat)
        
        return np.array(data)  # [num_frames, freq, time, antennas]
    
    def _load_rgb_frame(self, rgb_path):
        """加载单帧RGB骨骼点数据"""
        return np.load(rgb_path)  # [num_joints, coord_dim]
    
    def _load_csi_frame(self, csi_path):
        """加载单帧CSI数据"""
        data_mat = scio.loadmat(csi_path)['CSIamp']
        
        # 处理无穷值和NaN
        data_mat[np.isinf(data_mat)] = np.nan
        
        # 填充NaN值
        for i in range(data_mat.shape[-1]):  # 遍历天线
            temp_col = data_mat[:, :, i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                if len(temp_not_nan_col) > 0:
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
        
        # 归一化到[0,1]
        data_min = np.min(data_mat)
        data_max = np.max(data_mat)
        if data_max > data_min:
            data_mat = (data_mat - data_min) / (data_max - data_min)
        
        return data_mat  # [freq, time, antennas]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        item = self.data_list[idx]
        
        # 加载ground truth (3D骨骼点)
        gt_data = np.load(item['gt_path'])  # [num_frames, num_joints, 3]
        
        if self.data_unit == 'sequence':
            # 序列数据
            rgb_data = self._load_rgb_sequence(item['rgb_path'])  # [num_frames, num_joints, 2]
            csi_data = self._load_csi_sequence(item['csi_path'])  # [num_frames, freq, time, antennas]
            gt_target = gt_data  # [num_frames, num_joints, 3]
            
            sample = {
                'scene': item['scene'],
                'subject': item['subject'],
                'action': item['action'],
                'rgb_skeleton': torch.FloatTensor(rgb_data),
                'csi_data': torch.FloatTensor(csi_data),
                'gt_skeleton': torch.FloatTensor(gt_target),
                'data_unit': 'sequence'
            }
            
        elif self.data_unit == 'frame':
            # 单帧数据
            rgb_data = self._load_rgb_frame(item['rgb_path'])  # [num_joints, 2]
            csi_data = self._load_csi_frame(item['csi_path'])  # [freq, time, antennas]
            gt_target = gt_data[item['idx']]  # [num_joints, 3]
            
            sample = {
                'scene': item['scene'],
                'subject': item['subject'],
                'action': item['action'],
                'idx': item['idx'],
                'rgb_skeleton': torch.FloatTensor(rgb_data),
                'csi_data': torch.FloatTensor(csi_data),
                'gt_skeleton': torch.FloatTensor(gt_target),
                'data_unit': 'frame'
            }
        
        return sample


def enhanced_collate_fn(batch):
    """
    Enhanced MMFi数据的批处理函数
    """
    batch_data = {
        'scene': [sample['scene'] for sample in batch],
        'subject': [sample['subject'] for sample in batch],
        'action': [sample['action'] for sample in batch],
        'data_unit': batch[0]['data_unit']
    }
    
    # 处理索引（仅单帧模式有）
    if 'idx' in batch[0]:
        batch_data['idx'] = [sample['idx'] for sample in batch]
    
    # 批处理RGB骨骼点数据
    rgb_skeletons = [sample['rgb_skeleton'] for sample in batch]
    batch_data['rgb_skeleton'] = torch.stack(rgb_skeletons, dim=0)
    
    # 批处理CSI数据
    csi_data_list = [sample['csi_data'] for sample in batch]
    batch_data['csi_data'] = torch.stack(csi_data_list, dim=0)
    
    # 批处理ground truth
    gt_skeletons = [sample['gt_skeleton'] for sample in batch]
    batch_data['gt_skeleton'] = torch.stack(gt_skeletons, dim=0)
    
    return batch_data


def create_enhanced_mmfi_dataloaders(data_root, config_path, batch_size=32, num_workers=4):
    """
    创建Enhanced MMFi数据加载器
    
    Args:
        data_root: MMFi数据集根目录
        config_path: 配置文件路径
        batch_size: 批次大小
        num_workers: 工作进程数
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    import yaml
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建数据集
    train_dataset = EnhancedMMFiDataset(data_root, config, is_training=True)
    val_dataset = EnhancedMMFiDataset(data_root, config, is_training=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=enhanced_collate_fn,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=enhanced_collate_fn,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_enhanced_mmfi_dataloader():
    """测试Enhanced MMFi数据加载器"""
    print("=" * 60)
    print("测试Enhanced MMFi数据加载器")
    print("=" * 60)
    
    # 测试配置
    data_root = "C:\\tangyx\\MMFi_Dataset\\filtered_mmwave\\filtered_mmwave"  # 请修改为你的数据集路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MMFi_dataset', 'config.yaml')
    
    if not os.path.exists(data_root):
        print(f"⚠️  数据集路径不存在: {data_root}")
        print("请修改data_root为你的MMFi数据集路径")
        return
    
    if not os.path.exists(config_path):
        print(f"⚠️  配置文件不存在: {config_path}")
        return
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_enhanced_mmfi_dataloaders(
            data_root, config_path, batch_size=4
        )
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 测试一个批次
        print("\n测试训练集批次...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"批次 {batch_idx + 1}:")
            print(f"  RGB骨骼点形状: {batch['rgb_skeleton'].shape}")
            print(f"  CSI数据形状: {batch['csi_data'].shape}")
            print(f"  GT骨骼点形状: {batch['gt_skeleton'].shape}")
            print(f"  数据单位: {batch['data_unit']}")
            print(f"  场景: {batch['scene'][:2]}...")
            print(f"  动作: {batch['action'][:2]}...")
            
            if batch_idx >= 2:  # 只测试前3个批次
                break
        
        print("\n✅ Enhanced MMFi数据加载器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_mmfi_dataloader()

