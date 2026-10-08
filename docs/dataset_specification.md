# BAIR Robot Pushing Dataset Specification

## 数据集概述

**BAIR Robot Pushing** 是 UC Berkeley 机器人实验室发布的视频预测数据集，包含机械臂推动桌面物体的视频序列。

- **发布机构**: UC Berkeley BAIR Lab
- **数据量**: ~43,000 条轨迹（每条 30 帧）
- **数据大小**: ~4 GB（小版本：bair_robot_pushing_small）
- **帧率**: 60 FPS（原始）
- **分辨率**: 64×64 RGB
- **动作空间**: 4D（x, y, z 位移 + gripper 状态）

## 数据集特点

### 1. 天然遮挡场景
- 机械臂推动物体时，**手会挡住物体**
- 为 Phase 1 实验（空间不确定性地图）提供天然标注
- 无需人工标注遮挡区域

### 2. 动作条件
- 每帧配有 4D 动作向量：`(dx, dy, dz, gripper)`
- 支持 Phase 2/3 实验（主动探索、动作条件预测）

### 3. 轻量级
- 相比 Something-Something-v2 (~20GB)，体积小 5x
- 适合 RTX 4060 8GB 显存约束

## 数据加载流程

### 1. 帧降采样

**原始**: 60 FPS, 每条轨迹 30 帧 → 0.5s  
**降采样**: 30 FPS (每 2 帧取 1) → 15 帧 / 轨迹  
**目的**: 让 k=10 对应 ~0.33s（而非原始的 0.17s）

```python
# 降采样因子
downsample_factor = 2

# 应用降采样
frames = frames[::downsample_factor]  # 60 FPS → 30 FPS
actions = actions[::downsample_factor]
```

### 2. 随机跳跃采样

每个训练样本随机选择：
- **起始帧 t**: Uniform[0, T-k)
- **跳跃距离 k**: Uniform{1, 2, ..., 10}
- **目标帧**: t + k

```python
k = np.random.randint(min_k, max_k + 1)  # k ∈ [1, 10]
max_t = T - k
t = np.random.randint(0, max_t)

o_t = frames[t]
o_k = frames[t + k]
```

### 3. 输出格式

每个样本返回 4 个元素：

```python
(o_t, o_k, delta_t, action)
```

- **o_t**: `(3, 64, 64)` Tensor，起始帧，归一化到 [0, 1]
- **o_k**: `(3, 64, 64)` Tensor，目标帧
- **delta_t**: `float`，归一化时间差 `k / max_k` ∈ [0, 1]
- **action**: `(k, 4)` Tensor，从 t 到 t+k 的动作序列

### 4. 批处理

**注意**: 由于 k 是随机的，每个样本的 `action` 长度不同。

**解决方案**（两种）:
1. **Padding**: 将所有 action pad 到 max_k 长度（默认）
2. **List[Tensor]**: 返回变长 Tensor 列表（需要自定义 collate_fn）

当前实现采用方案 1（固定长度，简化训练循环）。

## 数据分割

- **train**: ~40,000 条轨迹
- **validation**: ~2,000 条
- **test**: ~1,000 条

## 加载示例

```python
from code.bair_loader import BAIRConfig, create_dataloader

# 配置
cfg = BAIRConfig(
    data_dir="data/bair",
    split="train",
    img_size=(64, 64),
    downsample_factor=2,  # 60 FPS → 30 FPS
    max_k=10
)

# 创建 DataLoader
loader = create_dataloader(
    cfg,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

# 迭代
for o_t, o_k, delta_t, action in loader:
    # o_t: (B, 3, 64, 64)
    # o_k: (B, 3, 64, 64)
    # delta_t: (B,)
    # action: (B, k, 4) 或 (B, max_k, 4) if padded
    pass
```

## 可视化验证

运行验证脚本：

```bash
python code/bair_loader.py --visualize --check_actions
```

生成：
- `data/bair/sample_visualization.png`: 4 对 (o_t, o_k) 样本
- 动作完整性检查报告（控制台输出）

## 与 TinyWM 架构的对接

TinyWM 期望输入：
```python
forward(o_t, o_k, delta_t)
```

从 DataLoader 加载的数据可以直接使用：
```python
for o_t, o_k, delta_t, action in loader:
    o_t, o_k = o_t.to(device), o_k.to(device)
    delta_t = delta_t.to(device)
    
    # 前向传播
    s_t, ens_mu, mu_mean, mu, logvar = model(o_t, o_k, delta_t)
    
    # 计算损失
    losses = compute_losses(
        (s_t, ens_mu, mu_mean, mu, logvar),
        o_k,  # 目标帧
        delta_t.unsqueeze(1),  # (B,) → (B, 1)
        cfg
    )
```

`action` 暂时未使用（Phase 1 只做离线校准），Phase 2/3 将用于动作条件预测。

## 数据增强（可选）

当前未实现，未来可以添加：
- 随机裁剪（spatial jitter）
- 颜色抖动（color jitter）
- 随机水平翻转

但注意：**遮挡检测实验 (RQ2) 需要保持原始遮挡模式**，增强可能干扰结果。建议先在无增强条件下完成 Phase 1。

## 参考文献

- **论文**: Finn et al., "Unsupervised Learning for Physical Interaction through Video Prediction", NeurIPS 2016
- **数据集**: https://sites.google.com/view/sna-visual-mpc/
- **TFDS 文档**: https://www.tensorflow.org/datasets/catalog/bair_robot_pushing
