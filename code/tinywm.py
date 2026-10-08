"""
TinyWM-v0 — 空间不确定性世界模型的最小可运行实现

设计要点（与 HAUWM 对齐，但把 RSSM 递推从预训练阶段移除）：
  1. 无循环。编码器一次前向得到 s_t = Enc(o_t)，集成分歧头在 Δt_k 条件下
     直接预测未来隐状态。HAUWM 的预训练阶段本就是这种"直接跳跃预测"，
     不需要 RSSM 递推 —— RSSM 只在下游 RL 微调阶段才需要。
  2. 无时序维度。训练样本是 (o_t, o_{t+k}, k) 三元组，k 每样本独立，
     所以整个 batch 一次前向即可，没有 T 维循环。
  3. 解码器双头：μ (重建) + log σ² (空间不确定性图)。

作者备注：本文件只依赖 torch，可直接 `python tinywm.py` 做冒烟测试 + 参数量
与单步耗时基准，不需要数据集。
"""

import math
import time
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class TinyWMConfig:
    # 输入
    in_channels: int = 3
    image_size: int = 64
    k_max: int = 10

    # 编码器: 64 -> 32 -> 16 -> 8 -> 4 (stride 2, kernel 4, pad 1)
    enc_channels: Tuple[int, ...] = (16, 32, 64, 64)
    enc_out_dim: int = 256          # squeeze 到 2x2x64

    # 隐状态
    stoch_dim: int = 64             # z_t 维度（解码器输入）

    # 集成分歧头
    n_heads: int = 5                # M
    dt_embed_dim: int = 64
    head_embed_dim: int = 8
    head_hidden: int = 128

    # 解码器
    dec_latent_hw: int = 4          # 4x4x128
    dec_latent_ch: int = 128
    dec_channels: Tuple[int, ...] = (64, 32, 16)

    # 损失权重
    lam_hcu: float = 1.0            # 时域分歧增长（HAUWM 的 HCU）
    lam_spatial: float = 1.0        # 空间校准（误差-σ² 相关性）
    lam_tv: float = 0.01            # 不确定性图 TV 正则
    # σ² 膨胀稳定器：把 log σ² 拉向目标值，防止 NLL 靠"全图大方差"作弊
    lam_var_prior: float = 0.05
    var_prior_target: float = -3.0  # ≈ σ² ≈ 0.05

    # 训练
    lr: float = 3e-4
    batch_size: int = 128


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def dt_embedding(k: torch.Tensor, dim: int) -> torch.Tensor:
    """相对时域嵌入：编码 k 步的时间间隔，而非序列位置。

    k: (B,) int64 -> (B, dim)
    """
    half = dim // 2
    device = k.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
    )
    ang = k.float().unsqueeze(1) * freqs.unsqueeze(0)     # (B, half)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐样本可微 Pearson 相关系数。a, b: (B, ...) -> (B,)"""
    a = a.flatten(1)
    b = b.flatten(1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + eps
    return num / den


def _mlp(sizes, act=nn.ELU, out_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# 编码器
# --------------------------------------------------------------------------
class Encoder(nn.Module):
    """Conv 堆叠 + 空间 squeeze，输出 s_t（同时充当预测头的输入与 z_t）。"""

    def __init__(self, cfg: TinyWMConfig):
        super().__init__()
        chs = cfg.enc_channels
        layers = []
        c_in = cfg.in_channels
        for c_out in chs:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.GroupNorm(8, c_out),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        # 4x4xC -> 2x2x64
        self.squeeze = nn.Sequential(
            nn.Conv2d(chs[-1], cfg.enc_out_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) -> (B, enc_out_dim)"""
        h = self.net(obs)
        h = self.squeeze(h)
        return h.flatten(1)


# --------------------------------------------------------------------------
# 集成分歧动力学头
# --------------------------------------------------------------------------
class EnsembleHead(nn.Module):
    """单个动力学头：s_t + Δt_k^e (+ head embedding) -> 未来隐状态 μ_i。"""

    def __init__(self, cfg: TinyWMConfig):
        super().__init__()
        in_dim = cfg.enc_out_dim + cfg.dt_embed_dim + cfg.head_embed_dim
        self.net = _mlp([in_dim, cfg.head_hidden, cfg.stoch_dim])

    def forward(self, s_t: torch.Tensor, dt: torch.Tensor, h_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_t, dt, h_emb], dim=-1))


# --------------------------------------------------------------------------
# 双头解码器：μ + log σ²
# --------------------------------------------------------------------------
class SpatialUncertaintyDecoder(nn.Module):
    def __init__(self, cfg: TinyWMConfig):
        super().__init__()
        self.cfg = cfg
        hw, c = cfg.dec_latent_hw, cfg.dec_latent_ch
        self.inp = nn.Linear(cfg.stoch_dim, hw * hw * c)
        c_in = c
        ups = []
        for c_out in cfg.dec_channels:
            ups += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.GroupNorm(8, c_out),
            ]
            c_in = c_out
        self.ups = nn.Sequential(*ups)
        # 32x32x16 -> 64x64x16 用最近邻插值（比再上一层 deconv 便宜 ~7x）
        self.up_final = nn.Upsample(scale_factor=2, mode="nearest")
        self.mu_head = nn.Conv2d(c_in, cfg.in_channels, kernel_size=3, stride=1, padding=1)
        self.logvar_head = nn.Conv2d(c_in, 1, kernel_size=3, stride=1, padding=1)
        self.img_size = cfg.image_size

    def forward(self, z: torch.Tensor):
        h = self.inp(z).view(
            -1, self.cfg.dec_latent_ch, self.cfg.dec_latent_hw, self.cfg.dec_latent_hw
        )
        h = self.ups(h)
        h = self.up_final(h)
        if h.shape[-1] != self.img_size:
            h = F.interpolate(h, size=(self.img_size, self.img_size), mode="nearest")
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


# --------------------------------------------------------------------------
# 完整模型
# --------------------------------------------------------------------------
class SpatialUncertaintyWM(nn.Module):
    """TinyWM-v0：无循环、可变时域、集成分歧 + 空间不确定性双头。"""

    def __init__(self, cfg: TinyWMConfig = None):
        super().__init__()
        self.cfg = cfg or TinyWMConfig()
        self.encoder = Encoder(self.cfg)
        self.head_emb = nn.Embedding(self.cfg.n_heads, self.cfg.head_embed_dim)
        self.heads = nn.ModuleList([EnsembleHead(self.cfg) for _ in range(self.cfg.n_heads)])
        self.decoder = SpatialUncertaintyDecoder(self.cfg)

    def forward(self, obs_t: torch.Tensor, k: torch.Tensor):
        """
        obs_t: (B, C, H, W)  当前帧
        k:     (B,)          预测时域（1..k_max）
        returns dict: s_t, ens_mu (B,M,D), mu_mean (B,D), mu (B,3,H,W), logvar (B,1,H,W)
        """
        B = obs_t.shape[0]
        s_t = self.encoder(obs_t)
        dt = dt_embedding(k, self.cfg.dt_embed_dim)

        mus = []
        for i, head in enumerate(self.heads):
            h = self.head_emb.weight[i].unsqueeze(0).expand(B, -1)
            mus.append(head(s_t, dt, h))
        ens_mu = torch.stack(mus, dim=1)                 # (B, M, D)
        mu_mean = ens_mu.mean(dim=1)                     # (B, D)

        mu, logvar = self.decoder(mu_mean)
        return {
            "s_t": s_t,
            "ens_mu": ens_mu,
            "mu_mean": mu_mean,
            "mu": mu,
            "logvar": logvar,
        }


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------
def compute_losses(out: dict, target: torch.Tensor, k: torch.Tensor, cfg: TinyWMConfig):
    mu, logvar, ens_mu = out["mu"], out["logvar"], out["ens_mu"]

    # L1 异方差 NLL（像素级）：误差大方差小则重罚，防止 σ² 膨胀
    logvar_c = logvar.clamp(min=-8.0, max=4.0)
    var = logvar_c.exp()
    l_recon = 0.5 * (((target - mu) ** 2) / var + logvar_c).mean()

    # L2 空间校准：最大化 |误差| 与 σ² 的相关性
    err_map = (target - mu).pow(2).mean(dim=1, keepdim=True)   # (B,1,H,W)
    l_spatial = -pearson_corr(err_map, var).mean()

    # L3 时域分歧增长（HAUWM HCU）：最小化它 = 最大化被 k 放大的集成分歧
    mu_bar = ens_mu.mean(dim=1, keepdim=True)
    disagree = (ens_mu - mu_bar).pow(2).sum(-1).mean(-1)       # (B,)
    l_hcu = -(k.float() * disagree).mean()

    # L4 TV 正则（在 log 空间做，尺度无关）
    d_h = (logvar_c[:, :, 1:, :] - logvar_c[:, :, :-1, :]).abs().mean()
    d_w = (logvar_c[:, :, :, 1:] - logvar_c[:, :, :, :-1]).abs().mean()
    l_tv = d_h + d_w

    # L5 σ² 膨胀稳定器
    l_prior = (logvar_c - cfg.var_prior_target).pow(2).mean()

    total = (
        l_recon
        + cfg.lam_spatial * l_spatial
        + cfg.lam_hcu * l_hcu
        + cfg.lam_tv * l_tv
        + cfg.lam_var_prior * l_prior
    )
    parts = {
        "total": total,
        "recon": l_recon,
        "spatial": l_spatial,
        "hcu": l_hcu,
        "tv": l_tv,
        "var_prior": l_prior,
        "sigma2_mean": var.mean().detach(),
        "disagree": disagree.mean().detach(),
    }
    return total, parts


# --------------------------------------------------------------------------
# 冒烟测试 + 参数量 / 耗时基准
# --------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    cfg = TinyWMConfig()
    model = SpatialUncertaintyWM(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 62)
    print(f"参数量: {n_params/1e6:.3f} M")
    print("=" * 62)
    for name, mod in [
        ("encoder", model.encoder),
        ("head_emb", model.head_emb),
        ("heads(ensemble)", model.heads),
        ("decoder", model.decoder),
    ]:
        c = sum(p.numel() for p in mod.parameters())
        print(f"  {name:20s} {c/1e6:7.3f} M  ({100*c/n_params:4.1f}%)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    print(f"\n设备: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    B = cfg.batch_size
    x0 = torch.rand(B, 3, cfg.image_size, cfg.image_size, device=dev)
    xk = torch.rand(B, 3, cfg.image_size, cfg.image_size, device=dev)
    k = torch.randint(1, cfg.k_max + 1, (B,), device=dev)

    out = model(x0, k)
    print("\n前向输出形状:")
    for key in ["s_t", "ens_mu", "mu_mean", "mu", "logvar"]:
        print(f"  {key:10s} {tuple(out[key].shape)}")

    loss, parts = compute_losses(out, xk, k, cfg)
    loss.backward()
    print("\n损失分量:")
    for key, v in parts.items():
        print(f"  {key:12s} {float(v): .5f}")

    # ---- 单步耗时基准 ----
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def one_step():
        o = model(x0, k)
        l, _ = compute_losses(o, xk, k, cfg)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()

    for _ in range(10):
        one_step()
    if dev == "cuda":
        torch.cuda.synchronize()

    N = 100
    t0 = time.time()
    for _ in range(N):
        one_step()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt_step = (time.time() - t0) / N

    mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    frames = B
    print("\n" + "=" * 62)
    print(f"batch={B}  单步耗时={dt_step*1000:.1f} ms  吞吐={frames/dt_step:.0f} frames/s")
    if dev == "cuda":
        print(f"峰值显存={mem:.2f} GB / 总 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"100k 步预计: {dt_step*100_000/3600:.2f} 小时")
    print(f"等价 epoch (BAIR 44k 帧): {100_000/(44_000/frames):.0f} epochs")
    print("=" * 62)


if __name__ == "__main__":
    main()