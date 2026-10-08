"""
BAIR Robot Pushing Dataset Loader

数据集规格：
- 原始帧率：60 FPS
- 降采样到：25 FPS (每 2.4 帧取 1 帧，实际取整为每 2 帧取 1 帧 ≈ 30 FPS，或每 3 帧取 1 帧 ≈ 20 FPS)
- 跳跃距离 k：Uniform{1..10}，对应 0.04s ~ 0.4s (at 25 FPS)
- 输出元组：(o_t, o_k, delta_t, action)
"""

import os
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


@dataclass
class BAIRConfig:
    """BAIR 数据集配置"""
    data_dir: str = "data/bair"
    split: str = "train"  # train / val / test
    img_size: Tuple[int, int] = (64, 64)
    downsample_factor: int = 2  # 60 FPS → 30 FPS (每 2 帧取 1)
    max_k: int = 10  # 最大跳跃距离
    min_k: int = 1   # 最小跳跃距离
    sequence_length: int = 30  # 每条序列的帧数（降采样后）
    action_dim: int = 4  # BAIR 动作维度（实际是 4D: x, y, z, gripper）
    use_tfds: bool = False  # 是否使用 TensorFlow Datasets（默认从磁盘加载）


class BAIRDataset(Dataset):
    """
    BAIR Robot Pushing Dataset
    
    每个样本返回：
    - o_t: 起始帧 (C, H, W)
    - o_k: 跳跃后帧 (C, H, W)
    - delta_t: 时间差（归一化到 [0, 1]），float
    - action: 动作序列 (k, action_dim)，从 t 到 t+k 的动作
    """
    
    def __init__(self, cfg: BAIRConfig):
        self.cfg = cfg
        self.sequences = []
        self.actions = []
        
        if cfg.use_tfds:
            self._load_from_tfds()
        else:
            self._load_from_disk()
    
    def _load_from_tfds(self):
        """从 TensorFlow Datasets 加载"""
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow_datasets not installed. "
                "Run: pip install tensorflow-datasets"
            )
        
        # BAIR 数据集在 TFDS 中的名称是 'bair_robot_pushing_small'
        dataset_name = 'bair_robot_pushing_small'
        
        # 加载数据集
        ds, info = tfds.load(
            dataset_name,
            split=self.cfg.split,
            data_dir=self.cfg.data_dir,
            with_info=True,
            download=True
        )
        
        print(f"Loading BAIR {self.cfg.split} split from TFDS...")
        print(f"Dataset info: {info}")
        
        # 转换为 numpy 数组列表
        for example in ds:
            # example['image_main']: (T, H, W, C)，原始 60 FPS
            frames = example['image_main'].numpy()
            actions = example['action'].numpy()  # (T, action_dim)
            
            # 降采样：每 downsample_factor 帧取 1 帧
            frames = frames[::self.cfg.downsample_factor]
            actions = actions[::self.cfg.downsample_factor]
            
            # 裁剪到指定长度
            if len(frames) >= self.cfg.sequence_length:
                frames = frames[:self.cfg.sequence_length]
                actions = actions[:self.cfg.sequence_length]
                
                self.sequences.append(frames)
                self.actions.append(actions)
        
        print(f"Loaded {len(self.sequences)} sequences")
    
    def _load_from_disk(self):
        """从磁盘加载原始 BAIR 数据（push_train / push_test 格式）"""
        import json
        from glob import glob
        
        # 确定数据目录
        if self.cfg.split == "train":
            split_dir = os.path.join(self.cfg.data_dir, "push_train")
        elif self.cfg.split in ["test", "validation"]:
            split_dir = os.path.join(self.cfg.data_dir, "push_test")
        else:
            raise ValueError(f"Unknown split: {self.cfg.split}")
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Data directory not found: {split_dir}\n"
                "Run: python scripts/download_bair.py"
            )
        
        # 扫描所有轨迹目录
        traj_dirs = sorted(glob(os.path.join(split_dir, "traj_*")))
        
        if len(traj_dirs) == 0:
            raise FileNotFoundError(
                f"No trajectory directories found in {split_dir}"
            )
        
        print(f"Loading BAIR {self.cfg.split} split from {split_dir}...")
        print(f"Found {len(traj_dirs)} trajectories")
        
        # 加载每条轨迹
        for traj_dir in tqdm(traj_dirs, desc="Loading trajectories"):
            # 加载图像帧
            frames = []
            for i in range(30):  # BAIR 每条轨迹 30 帧
                img_path = os.path.join(traj_dir, f"{i}.png")
                if not os.path.exists(img_path):
                    break
                img = Image.open(img_path)
                frames.append(np.array(img))
            
            if len(frames) < self.cfg.sequence_length:
                continue  # 跳过不完整的轨迹
            
            frames = np.stack(frames)  # (T, H, W, C)
            
            # 加载动作
            actions_path = os.path.join(traj_dir, "actions.npy")
            if not os.path.exists(actions_path):
                # 如果没有 .npy，尝试 .json
                actions_path = os.path.join(traj_dir, "actions.json")
                if os.path.exists(actions_path):
                    with open(actions_path, 'r') as f:
                        actions_data = json.load(f)
                    actions = np.array(actions_data)
                else:
                    # 没有动作数据，使用零向量
                    actions = np.zeros((len(frames), self.cfg.action_dim))
            else:
                actions = np.load(actions_path)
            
            # 降采样
            frames = frames[::self.cfg.downsample_factor]
            actions = actions[::self.cfg.downsample_factor]
            
            # 裁剪到指定长度
            frames = frames[:self.cfg.sequence_length]
            actions = actions[:self.cfg.sequence_length]
            
            self.sequences.append(frames)
            self.actions.append(actions)
        
        print(f"✓ Loaded {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        返回：
        - o_t: (3, H, W) 起始帧
        - o_k: (3, H, W) 目标帧
        - delta_t: float，归一化时间差 [0, 1]
        - action: (k, action_dim) 动作序列
        """
        seq = self.sequences[idx]  # (T, H, W, C)
        act = self.actions[idx]    # (T, action_dim)
        
        # 随机选择 t 和 k
        T = len(seq)
        k = np.random.randint(self.cfg.min_k, self.cfg.max_k + 1)
        
        # 确保 t + k < T
        max_t = T - k
        if max_t <= 0:
            # 序列太短，回退到 k=1
            k = 1
            max_t = T - 1
        
        t = np.random.randint(0, max_t)
        
        # 提取帧
        o_t = seq[t]  # (H, W, C)
        o_k = seq[t + k]
        
        # 提取动作（从 t 到 t+k）
        action = act[t:t+k]  # (k, action_dim)
        
        # Resize 到目标尺寸
        o_t = self._resize_frame(o_t)
        o_k = self._resize_frame(o_k)
        
        # 转换为 torch tensor
        o_t = self._to_tensor(o_t)  # (3, H, W)
        o_k = self._to_tensor(o_k)
        action = torch.from_numpy(action).float()
        
        # 归一化时间差（假设降采样后是 30 FPS，k=10 对应 0.33s）
        # 这里用 k / max_k 作为归一化值
        delta_t = float(k) / self.cfg.max_k
        
        return o_t, o_k, delta_t, action
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize 帧到目标尺寸"""
        img = Image.fromarray(frame)
        img = img.resize(self.cfg.img_size, Image.BILINEAR)
        return np.array(img)
    
    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """转换为 PyTorch tensor 并归一化到 [0, 1]"""
        # (H, W, C) → (C, H, W)
        frame = frame.transpose(2, 0, 1)
        # uint8 [0, 255] → float32 [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return torch.from_numpy(frame)


def create_dataloader(
    cfg: BAIRConfig,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """创建 DataLoader"""
    dataset = BAIRDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


# ==================== 数据管道验证脚本 ====================

def visualize_samples(cfg: BAIRConfig, num_samples: int = 4):
    """可视化样本"""
    import matplotlib.pyplot as plt
    
    dataset = BAIRDataset(cfg)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        o_t, o_k, delta_t, action = dataset[i]
        
        # 转换为 (H, W, C) 格式用于显示
        o_t_img = o_t.permute(1, 2, 0).numpy()
        o_k_img = o_k.permute(1, 2, 0).numpy()
        
        axes[i, 0].imshow(o_t_img)
        axes[i, 0].set_title(f"Frame t (sample {i})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(o_k_img)
        axes[i, 1].set_title(f"Frame t+k (Δt={delta_t:.2f}, k={len(action)})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("data/bair/sample_visualization.png", dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to data/bair/sample_visualization.png")
    plt.close()


def check_action_integrity(cfg: BAIRConfig):
    """检查动作标签完整性"""
    dataset = BAIRDataset(cfg)
    
    print("\n=== Action Integrity Check ===")
    print(f"Dataset size: {len(dataset)}")
    
    # 采样检查
    num_check = min(100, len(dataset))
    action_shapes = []
    action_ranges = []
    
    for i in range(num_check):
        o_t, o_k, delta_t, action = dataset[i]
        action_shapes.append(action.shape[0])  # k
        action_ranges.append((action.min().item(), action.max().item()))
    
    print(f"Action sequence lengths (k): min={min(action_shapes)}, max={max(action_shapes)}, mean={np.mean(action_shapes):.1f}")
    
    all_mins = [r[0] for r in action_ranges]
    all_maxs = [r[1] for r in action_ranges]
    print(f"Action value ranges: min={min(all_mins):.3f}, max={max(all_maxs):.3f}")
    
    # 检查是否有 NaN
    has_nan = False
    for i in range(num_check):
        _, _, _, action = dataset[i]
        if torch.isnan(action).any():
            has_nan = True
            print(f"⚠ Warning: NaN detected in sample {i}")
    
    if not has_nan:
        print("✓ No NaN values detected")
    
    print("=== Check Complete ===\n")


if __name__ == "__main__":
    """验证脚本"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--visualize", action="store_true", help="生成可视化样本")
    parser.add_argument("--check_actions", action="store_true", help="检查动作完整性")
    args = parser.parse_args()
    
    # 配置
    cfg = BAIRConfig(split=args.split)
    
    # 创建 DataLoader
    loader = create_dataloader(cfg, batch_size=args.batch_size, shuffle=True)
    
    print(f"Created DataLoader for {args.split} split")
    print(f"  Dataset size: {len(loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {len(loader)}")
    
    # 测试一个 batch
    o_t, o_k, delta_t, action = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  o_t: {o_t.shape}")
    print(f"  o_k: {o_k.shape}")
    print(f"  delta_t: {delta_t.shape}")
    print(f"  action: {action.shape}")
    
    # 可视化
    if args.visualize:
        visualize_samples(cfg, num_samples=4)
    
    # 检查动作
    if args.check_actions:
        check_action_integrity(cfg)
