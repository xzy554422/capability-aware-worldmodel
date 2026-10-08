"""
创建 Mock BAIR 数据集用于快速开发测试

在真实数据下载完成前，可以用这个 mock 数据集验证训练流程。
"""

import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm


def create_mock_bair_dataset(
    output_dir: str = "data/bair",
    num_train: int = 100,
    num_test: int = 20,
    frames_per_traj: int = 30,
    img_size: int = 64,
    action_dim: int = 4
):
    """
    创建 Mock BAIR 数据集
    
    生成随机图像和动作，遵循 BAIR 原始格式：
    - push_train/traj_0/ ... traj_N/
      - 0.png, 1.png, ..., 29.png
      - actions.npy
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Creating Mock BAIR Dataset ===\n")
    
    # 训练集
    train_dir = os.path.join(output_dir, "push_train")
    os.makedirs(train_dir, exist_ok=True)
    
    print(f"Generating {num_train} training trajectories...")
    for i in tqdm(range(num_train)):
        traj_dir = os.path.join(train_dir, f"traj_{i}")
        os.makedirs(traj_dir, exist_ok=True)
        
        # 生成图像帧（渐变模式，模拟机械臂推动）
        base_color = np.random.randint(0, 255, size=3)
        for t in range(frames_per_traj):
            # 创建渐变图像（从 base_color 渐变到白色）
            alpha = t / frames_per_traj
            color = (base_color * (1 - alpha) + 255 * alpha).astype(np.uint8)
            
            # 添加噪声
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * color
            noise = np.random.randint(-20, 20, size=(img_size, img_size, 3))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # 添加"机械臂"（简单矩形）
            arm_x = int(img_size * 0.3 + t * img_size * 0.4 / frames_per_traj)
            arm_y = int(img_size * 0.5)
            img[arm_y-5:arm_y+5, arm_x-3:arm_x+3] = [255, 0, 0]  # 红色
            
            # 添加"物体"（圆形）
            obj_x = int(img_size * 0.4 + t * img_size * 0.3 / frames_per_traj)
            obj_y = int(img_size * 0.5)
            for dy in range(-8, 8):
                for dx in range(-8, 8):
                    if dx**2 + dy**2 < 64:  # 圆形
                        py, px = obj_y + dy, obj_x + dx
                        if 0 <= py < img_size and 0 <= px < img_size:
                            img[py, px] = [0, 255, 0]  # 绿色
            
            # 保存图像
            Image.fromarray(img).save(os.path.join(traj_dir, f"{t}.png"))
        
        # 生成动作（随机游走）
        actions = np.cumsum(np.random.randn(frames_per_traj, action_dim) * 0.1, axis=0)
        actions = np.clip(actions, -1, 1)  # 归一化到 [-1, 1]
        np.save(os.path.join(traj_dir, "actions.npy"), actions.astype(np.float32))
    
    # 测试集
    test_dir = os.path.join(output_dir, "push_test")
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\nGenerating {num_test} test trajectories...")
    for i in tqdm(range(num_test)):
        traj_dir = os.path.join(test_dir, f"traj_{i}")
        os.makedirs(traj_dir, exist_ok=True)
        
        base_color = np.random.randint(0, 255, size=3)
        for t in range(frames_per_traj):
            alpha = t / frames_per_traj
            color = (base_color * (1 - alpha) + 255 * alpha).astype(np.uint8)
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * color
            noise = np.random.randint(-20, 20, size=(img_size, img_size, 3))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            arm_x = int(img_size * 0.3 + t * img_size * 0.4 / frames_per_traj)
            arm_y = int(img_size * 0.5)
            img[arm_y-5:arm_y+5, arm_x-3:arm_x+3] = [255, 0, 0]
            
            obj_x = int(img_size * 0.4 + t * img_size * 0.3 / frames_per_traj)
            obj_y = int(img_size * 0.5)
            for dy in range(-8, 8):
                for dx in range(-8, 8):
                    if dx**2 + dy**2 < 64:
                        py, px = obj_y + dy, obj_x + dx
                        if 0 <= py < img_size and 0 <= px < img_size:
                            img[py, px] = [0, 255, 0]
            
            Image.fromarray(img).save(os.path.join(traj_dir, f"{t}.png"))
        
        actions = np.cumsum(np.random.randn(frames_per_traj, action_dim) * 0.1, axis=0)
        actions = np.clip(actions, -1, 1)
        np.save(os.path.join(traj_dir, "actions.npy"), actions.astype(np.float32))
    
    print("\n=== Mock Dataset Created ===")
    print(f"Training trajectories: {num_train} ({train_dir})")
    print(f"Test trajectories: {num_test} ({test_dir})")
    print(f"\n✓ You can now run: python code/verify_data.py")
    print("  (This mock dataset is for development only. Replace with real BAIR data for training.)")


if __name__ == "__main__":
    create_mock_bair_dataset()
