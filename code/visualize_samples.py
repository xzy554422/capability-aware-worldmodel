"""可视化 BAIR 数据集样本"""
import sys
sys.path.insert(0, '.')

from code.bair_loader import BAIRConfig, BAIRDataset
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Loading BAIR dataset...")
    cfg = BAIRConfig(split="train")
    dataset = BAIRDataset(cfg)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 可视化 4 个样本
    num_samples = 4
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    for i in range(num_samples):
        o_t, o_k, delta_t, action = dataset[np.random.randint(len(dataset))]
        
        # 转换为 (H, W, C) 格式
        o_t_img = o_t.permute(1, 2, 0).numpy()
        o_k_img = o_k.permute(1, 2, 0).numpy()
        
        axes[i, 0].imshow(o_t_img)
        axes[i, 0].set_title(f"Frame t (sample {i})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(o_k_img)
        axes[i, 1].set_title(f"Frame t+k (Δt={delta_t:.2f}, k={len(action)})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    output_path = "data/bair/sample_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()
    
    # 动作完整性检查
    print("\n=== Action Integrity Check ===")
    num_check = min(100, len(dataset))
    action_shapes = []
    action_ranges = []
    
    for i in range(num_check):
        o_t, o_k, delta_t, action = dataset[i]
        action_shapes.append(action.shape[0])
        action_ranges.append((action.min().item(), action.max().item()))
    
    print(f"Checked {num_check} samples")
    print(f"Action sequence lengths (k): min={min(action_shapes)}, max={max(action_shapes)}, mean={np.mean(action_shapes):.1f}")
    
    all_mins = [r[0] for r in action_ranges]
    all_maxs = [r[1] for r in action_ranges]
    print(f"Action value ranges: min={min(all_mins):.3f}, max={max(all_maxs):.3f}")
    
    # 检查 NaN
    has_nan = False
    for i in range(num_check):
        import torch
        _, _, _, action = dataset[i]
        if torch.isnan(action).any():
            has_nan = True
            print(f"⚠ Warning: NaN detected in sample {i}")
    
    if not has_nan:
        print("✓ No NaN values detected")
    
    print("\n✓ Verification complete!")

if __name__ == "__main__":
    main()
