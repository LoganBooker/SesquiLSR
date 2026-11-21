"""
SesquiLSR — Multi-scale latent upscaler.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# I like Mish, but any decent activation (like SiLU) works fine.
Activation = nn.Mish

# Per-pixel channel RMSNorm with IR-style local RMS residual scaling (https://arxiv.org/abs/2504.06629).
# Normalizing over channels only (dim=1) instead of (C, H, W) eliminates global brightness
# coupling and makes full-frame vs tiled inference match bit-for-bit inside overlap.
class LocalIRRMSNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        inv_rms = torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-6)
        rms = 1.0 / inv_rms
        y = (x * inv_rms) * self.weight.view(1, -1, 1, 1)
        return y, rms

# Inspired by WDSR / wide activation: https://arxiv.org/abs/1808.08718
# Uses a lopsided gate design to maximise values, 8:1 ratio works well
# in practise.
class GatedWDSR(nn.Module):
    def __init__(
        self,
        channels: int,
        expand_ratio: float = 2.0,
        shrink_ratio: float = 0.8,
        values_per_gate: int = 8,
    ):
        super().__init__()
        if values_per_gate < 1:
            raise ValueError(f"values_per_gate must be >= 1, got {values_per_gate}")

        gate_ch = math.ceil(expand_ratio * channels / (values_per_gate + 1))
        gate_ch = math.ceil(gate_ch / values_per_gate) * values_per_gate
        value_ch = gate_ch * values_per_gate
        mid_ch = math.ceil(channels * shrink_ratio)
        hidden = value_ch + gate_ch

        self.value_ch = value_ch
        self.gate_ch = gate_ch
        self.values_per_gate = values_per_gate

        self.norm = LocalIRRMSNorm(channels)
        self.up = nn.Conv2d(channels, hidden, 1, bias=False)
        self.down = nn.Conv2d(value_ch, mid_ch, 1, bias=False)
        self.mix = nn.Conv2d(
            mid_ch, channels, kernel_size=3, padding=1,
            padding_mode="reflect", bias=False,
        )
        self.act = Activation()

    def forward(self, x: Tensor) -> Tensor:
        y, s = self.norm(x)
        v, g = torch.split(self.up(y), [self.value_ch, self.gate_ch], dim=1)
        v = self.act(v)
        g = torch.tanh(0.5 * g)  # Bound residual and slow down saturation.

        B, _, H, W = v.shape
        v = v.view(B, self.gate_ch, self.values_per_gate, H, W)
        y = (v * g[:, :, None, :, :]).reshape(B, self.value_ch, H, W)

        y = self.down(y)
        y = self.mix(y)
        return x + y * s

# The "sesqui" part. Inspired by rational/fractional resampling, we use PS to get to 2x,
# then use a learned downsample to reach our target. The original model was a fixed 1.5x;
# sesqui means "one and a half". While the fixed upscale is gone, the name remains.
#
# Evaluates a continuous 1D kernel K(d) at tap displacements d = tap_offset - frac in [-2, 2]:
#   - Rank 0 starts from the exact cubic B-spline B_3(d) (exact partition of unity and centroid frac).
#   - Learned per-tap residuals are multiplied by the smooth C^1 support envelope (1 - (d/2)^2)^2,
#     which vanishes identically at |d| = 2 so integer grid crossings have zero seam jump.
class LowRankReassemblyHead(nn.Module):
    def __init__(
        self,
        channels: int,
        refine_expand: float = 1.5,
        rank: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.K = 4
        self.register_buffer(
            "tap_offsets", torch.tensor([-1, 0, 1, 2], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "tap_offsets_f", torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32),
            persistent=False,
        )

        self.ps_expand = nn.Conv2d(channels, channels * 4, 1, bias=True)
        self._init_pixel_shuffle()

        lift_ch = min(64, channels // 2)
        self.lift_expand = nn.Conv2d(channels, lift_ch * 4, 1, bias=False)
        self.lift_mix = nn.Conv2d(lift_ch, channels, 3, padding=1, padding_mode="reflect", bias=False)
        nn.init.trunc_normal_(self.lift_expand.weight, std=0.02)
        nn.init.trunc_normal_(self.lift_mix.weight, std=0.02)

        # You could probably use a simple conv stack here, but another resblock works well so why not.
        self.refine = GatedWDSR(channels, expand_ratio=refine_expand, shrink_ratio=1.0, values_per_gate=8)

        # 7 continuous displacement features [d/2, cos/sin(pi*d/2), cos/sin(pi*d), cos/sin(2*pi*d)]
        # + log_footprint + axis_emb(8) + rank_emb(8) = 24
        self.axis_emb = nn.Embedding(2, 8)
        self.rank_emb = nn.Embedding(rank, 8)
        cond_dim = 7 + 1 + 8 + 8
        self.filter_net = nn.Sequential(
            nn.Linear(cond_dim, 64, bias=True),
            Activation(),
            nn.Linear(64, channels, bias=True),
        )
        nn.init.trunc_normal_(self.filter_net[-1].weight, std=0.02)
        nn.init.zeros_(self.filter_net[-1].bias)

        self.rank_gain = nn.Parameter(torch.tensor([2.0] + [0.0] * (rank - 1), dtype=torch.float32))
    def _init_pixel_shuffle(self) -> None:
        with torch.no_grad():
            self.ps_expand.weight.zero_()
            self.ps_expand.bias.zero_()
            w = self.ps_expand.weight.view(self.channels, 4, self.channels)
            idx = torch.arange(self.channels, device=w.device)
            w[idx, :, idx] = 1.0

    @staticmethod
    def _axis_coords(n_src: int, n_tgt: int, device: torch.device) -> tuple[Tensor, Tensor]:
        pos = (torch.arange(n_tgt, device=device, dtype=torch.float32) + 0.5)
        pos = pos * (n_src / n_tgt) - 0.5
        base = torch.floor(pos).long().clamp_(0, n_src - 1)
        frac = pos - base.float()
        return frac, base

    @staticmethod
    def _bspline_base(frac: Tensor) -> Tensor:
        u = 1.0 - frac
        return torch.stack([
            (u * u * u) / 6.0,
            (3.0 * frac * frac * frac - 6.0 * frac * frac + 4.0) / 6.0,
            (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
            (frac * frac * frac) / 6.0,
        ], dim=-1)

    def _predict_weights(self, fy: Tensor, fx: Tensor,
                         log_foot_y: float, log_foot_x: float,
                         batch: int) -> tuple[Tensor, Tensor]:
        """Compute seamless tap weights for all ranks/axes in a single MLP forward pass.

        Returns (ky, kx), each shaped [R, B, C*n_out, K].
        """
        ny, nx = fy.numel(), fx.numel()
        R = self.rank
        n_axis = ny + nx

        frac_axis = torch.cat([fy, fx])
        d = self.tap_offsets_f.unsqueeze(0) - frac_axis.unsqueeze(1)  # [n_axis, K] in [-2, 2]
        env = (1.0 - 0.25 * d * d).clamp_min(0.0).square()            # C^1 zero at |d| = 2

        half_pi_d = (0.5 * math.pi) * d
        pi_d = math.pi * d
        two_pi_d = (2.0 * math.pi) * d
        d_feat = torch.stack([
            0.5 * d,
            half_pi_d.cos(), half_pi_d.sin(),
            pi_d.cos(), pi_d.sin(),
            two_pi_d.cos(), two_pi_d.sin(),
        ], dim=-1)

        log_foot = fy.new_full((n_axis, self.K, 1), log_foot_x)
        log_foot[:ny] = log_foot_y
        axis_id = torch.empty((n_axis, self.K), dtype=torch.long, device=fy.device)
        axis_id[:ny] = 0
        axis_id[ny:] = 1

        cond_axis = torch.cat([d_feat, log_foot, self.axis_emb(axis_id)], dim=-1)
        rank_id = torch.arange(R, device=fy.device).view(R, 1, 1).expand(R, n_axis, self.K)
        cond = torch.cat([
            cond_axis.unsqueeze(0).expand(R, -1, -1, -1),
            self.rank_emb(rank_id),
        ], dim=-1)
        cond = cond.to(dtype=self.filter_net[0].weight.dtype)

        raw = self.filter_net(cond)                                    # [R, n_axis, K, C]
        res = (0.25 * torch.tanh(raw)) * env.unsqueeze(0).unsqueeze(-1)
        w = res.permute(0, 1, 3, 2)                                    # [R, n_axis, C, K]

        # Envelope-weighted zero-sum projection preserves exact 0 at |d| = 2
        env_w = env.unsqueeze(0).unsqueeze(2).to(dtype=w.dtype)
        w = w - (w.sum(dim=-1, keepdim=True) / env_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)) * env_w

        base_k = self._bspline_base(frac_axis).to(dtype=w.dtype).unsqueeze(1)
        w = torch.cat([(w[0] + base_k).unsqueeze(0), w[1:]], dim=0)
        w = w.to(dtype=cond.dtype)  # fp32 maths, emitted in the model's dtype

        wy = w[:, :ny]
        wx = w[:, ny:]

        ky = wy.permute(0, 2, 1, 3).reshape(R, self.channels * ny, self.K).unsqueeze(1).expand(-1, batch, -1, -1)
        kx = wx.permute(0, 2, 1, 3).reshape(R, self.channels * nx, self.K).unsqueeze(1).expand(-1, batch, -1, -1)
        return ky, kx

    def _filter_axis(self, x: Tensor, centers: Tensor, weights: Tensor, dim: int) -> Tensor:
        B, C = x.shape[0], x.shape[1]
        n_out = centers.numel()
        limit = x.shape[dim] - 1

        idx = (centers.unsqueeze(0) + self.tap_offsets.unsqueeze(1)).clamp(0, limit)
        w = weights.view(B, C, n_out, self.K)

        if dim == 2:
            idx_exp = idx.view(1, 1, self.K * n_out, 1).expand(B, C, -1, x.shape[3])
            sample = torch.gather(x, 2, idx_exp).view(B, C, self.K, n_out, x.shape[3])
            return (sample * w.permute(0, 1, 3, 2).unsqueeze(-1)).sum(dim=2)

        idx_exp = idx.view(1, 1, 1, self.K * n_out).expand(B, C, x.shape[2], -1)
        sample = torch.gather(x, 3, idx_exp).view(B, C, x.shape[2], self.K, n_out)
        return (sample * w.permute(0, 1, 3, 2).unsqueeze(2)).sum(dim=3)

    def forward(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:
        h_tgt, w_tgt = target_size
        batch = x.shape[0]

        x_base = F.pixel_shuffle(self.ps_expand(x), 2)
        x_lift = self.lift_mix(F.pixel_shuffle(self.lift_expand(x), 2))
        x = x_base + x_lift
        h_src, w_src = x.shape[-2:]

        x = self.refine(x)

        fy, iy = self._axis_coords(h_src, h_tgt, x.device)
        fx, ix = self._axis_coords(w_src, w_tgt, x.device)

        ky, kx = self._predict_weights(
            fy, fx,
            math.log2(max(h_src / h_tgt, 0.01)),
            math.log2(max(w_src / w_tgt, 0.01)),
            batch,
        )

        gains = self.rank_gain.sigmoid()
        out = torch.zeros(batch, self.channels, h_tgt, w_tgt, device=x.device, dtype=x.dtype)
        for r in range(self.rank):
            x_r = self._filter_axis(x, ix, kx[r], dim=3)
            x_r = self._filter_axis(x_r, iy, ky[r], dim=2)
            out = out + gains[r] * x_r

        return out

# Some models/VAE can get away with very little in terms of trunks. For example, the Qwen/Wan21 VAE
# was completely happy with 2/4 instead of 8/6. However, to keep the arch consistent, we use the same
# structure for all upscalers.
class LatentUpscaler(nn.Module):
    def __init__(
        self,
        in_depth: int = 8,
        out_depth: int = 6,
        in_channels: int = 4,
        width: int = 128,
        in_expand: float = 1.5,
        out_expand: float = 2.0,
    ):
        super().__init__()

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, width // 2, 3, padding=1, padding_mode="reflect", bias=False),
            Activation(),
            nn.Conv2d(width // 2, width, 3, padding=1, padding_mode="reflect", bias=True),
        )

        self.in_blocks = nn.Sequential(*[
            GatedWDSR(width, expand_ratio=in_expand, shrink_ratio=0.75, values_per_gate=8)
            for _ in range(in_depth)
        ])
        self.upsample = LowRankReassemblyHead(width, refine_expand=2.0)

        # Traditional SR says do most of the work at LR, then upscale and output. The problem is
        # latents are delicate structures where no amount of front-loaded work can make up for the
        # damage introduced by upscaling. Post-correction is a must, and all variants tested without
        # post-correction were much worse.
        self.out_blocks = nn.Sequential(*[
            GatedWDSR(width, expand_ratio=out_expand, shrink_ratio=1.0, values_per_gate=8)
            for _ in range(out_depth)
        ])
        self.tail_conv = nn.Sequential(
            nn.Conv2d(width, width // 2, 1, bias=False),
            Activation(),
            nn.Conv2d(width // 2, in_channels, 3, padding=1, padding_mode="reflect", bias=True),
        )
        # Give the model a way to do light correction on the skip so the trunk/head can focus on
        # detail/damage control.
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 5, padding=2, padding_mode="reflect", bias=False)
        nn.init.dirac_(self.skip_conv.weight)

    def forward(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:

        h = self.head_conv(x)
        h = self.in_blocks(h)
        h = self.upsample(h, target_size)
        h = self.out_blocks(h)
        h = self.tail_conv(h)

        # Magic sauce. Learn correction on crude upsample.
        x_up = self.skip_conv(F.interpolate(x, size=target_size, mode="bicubic", align_corners=False))
        return h + x_up
