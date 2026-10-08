# Capability-Aware World Model

**让世界模型知道自己哪里不准** — 为视频预测模型注入元认知能力

## 项目简介

本项目探索世界模型的"能力自知"（Capability Self-Knowledge），让模型显式学习对哪些预测有把握。核心贡献：

- **空间维度 (Where)**：像素级不确定性热图，可视化"哪里不准"
- **时间维度 (When)**：继承 HAUWM 的 HCU 损失，校准随时域增长的不确定性
- **内省维度 (Why)**：激活引导（从 LLM 内省迁移），学习元认知表征

## TinyWM-v0 架构

针对 **RTX 4060 8GB** 设计的轻量级世界模型：

- **参数量**: 0.731M (encoder 0.173M / ensemble heads 0.252M / decoder 0.306M)
- **峰值显存**: 0.29GB @ batch=128
- **训练时长**: 60k steps ≈ 1.2 小时
- **核心创新**: 
  - 集成不确定性建模 (M=5 个独立 MLP 头)
  - 空间异方差解码器 (输出 μ 和 log σ²)
  - 异方差 NLL 损失 + 防膨胀退化修复

详见 [`模型架构规格_TinyWM.md`](./模型架构规格_TinyWM.md)

## 快速开始

### 环境要求

- **硬件**: NVIDIA GPU with ≥8GB VRAM (tested on RTX 4060 Laptop)
- **软件**: Python 3.11+, CUDA 11.8+, PyTorch 2.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行冒烟测试

```bash
python code/tinywm.py
```

预期输出：
```
参数量: 0.731 M
batch=128  单步耗时=70.1 ms  吞吐=1826 frames/s
峰值显存=0.29 GB / 总 8.6 GB
```

## 项目结构

```
.
├── code/
│   └── tinywm.py           # TinyWM-v0 模型实现 + 冒烟测试
├── data/                   # 数据集目录 (gitignored)
├── checkpoints/            # 模型检查点 (gitignored)
├── logs/                   # 训练日志 (gitignored)
├── 模型架构规格_TinyWM.md   # 完整架构规格书
├── 实验方案_空间不确定性地图.md  # Phase 1 实验方案
└── README.md
```

## 数据集

使用 **BAIR Robot Pushing** (~4GB):
- 推物体时手挡住物体 → 天然遮挡场景
- 7 维动作标签 → 支持主动探索
- 帧降采样到 25FPS (k=10 对应 0.4s)

下载方式（待补充）

## 训练

```bash
# Phase 1: 空间不确定性地图 (60k steps)
python train.py --config configs/spatial_uncertainty.yaml
```

## 实验计划

**Phase 1**: 空间不确定性地图 (3周)
- RQ1: 不确定性 vs 误差 Spearman ρ > 0.7
- RQ2: 遮挡区域不确定性显著高于可见区域
- 6 组消融实验

**Phase 2**: 时间校准 (继承 HAUWM HCU)  
**Phase 3**: 内省机制 (激活引导)

详见 [`实验方案_空间不确定性地图.md`](./实验方案_空间不确定性地图.md)

## 关键文献

- **HAUWM** (ICLR 2026): Hierarchically Calibrated Uncertainty for World Models
- **VJEPA** (arXiv 2026.01): 概率化 JEPA，变分预测分布
- **LLM Introspection** (Transformer Circuits 2025.10): 概念注入测试框架

## 长期目标

- **投稿目标**: ICLR 2027
- **开源计划**: 训练代码 + 预训练权重 + 可视化工具

## License

MIT

## 引用

（待发表后补充 BibTeX）
