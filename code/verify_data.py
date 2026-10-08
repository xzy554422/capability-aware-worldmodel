"""快速验证数据加载器"""
import sys
sys.path.insert(0, '.')

from code.bair_loader import BAIRConfig, create_dataloader

def main():
    print("=== BAIR Data Loader Verification ===\n")
    
    # 配置
    cfg = BAIRConfig(split="train")
    
    print(f"Config:")
    print(f"  Data dir: {cfg.data_dir}")
    print(f"  Image size: {cfg.img_size}")
    print(f"  Downsample factor: {cfg.downsample_factor} (60 FPS → {60 // cfg.downsample_factor} FPS)")
    print(f"  k range: [{cfg.min_k}, {cfg.max_k}]")
    print(f"  Sequence length: {cfg.sequence_length} frames (after downsampling)")
    print()
    
    # 创建 DataLoader
    try:
        loader = create_dataloader(cfg, batch_size=16, shuffle=True, num_workers=0)
        print(f"✓ DataLoader created successfully")
        print(f"  Dataset size: {len(loader.dataset)}")
        print(f"  Batch size: 16")
        print(f"  Num batches: {len(loader)}")
        print()
    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        return
    
    # 测试加载一个 batch
    print("Testing batch loading...")
    try:
        o_t, o_k, delta_t, action = next(iter(loader))
        print(f"✓ Batch loaded successfully")
        print(f"\nBatch shapes:")
        print(f"  o_t:     {tuple(o_t.shape)}  (batch, channels, height, width)")
        print(f"  o_k:     {tuple(o_k.shape)}")
        print(f"  delta_t: {tuple(delta_t.shape)}  (batch,)")
        print(f"  action:  {tuple(action.shape)}  (batch, k, action_dim)")
        
        print(f"\nValue ranges:")
        print(f"  o_t:     [{o_t.min():.3f}, {o_t.max():.3f}]")
        print(f"  o_k:     [{o_k.min():.3f}, {o_k.max():.3f}]")
        print(f"  delta_t: [{delta_t.min():.3f}, {delta_t.max():.3f}]")
        print(f"  action:  [{action.min():.3f}, {action.max():.3f}]")
        
        # 检查 k 值分布
        k_values = [a.shape[0] for a in action]
        print(f"\nk distribution in this batch: min={min(k_values)}, max={max(k_values)}, mean={sum(k_values)/len(k_values):.1f}")
        
        print("\n✓ All checks passed!")
        
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
