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


# IR LayerNorm from https://arxiv.org/abs/2504.06629
# Removed bias to reduce chances of mean drift, also it didn't really make
# much of a difference with it in.
class IRLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
        std = torch.sqrt(var + 1e-6)
        y = (x - mean) / std * self.weight.view(1, -1, 1, 1)
        return y, std


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

        self.norm = IRLayerNorm(channels)
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


# Originally the model uses AdaLN in every block for scale conditioning, but this
# ended up exploding. Turns out we can just do conditioning in the head, and the
# model works fine.
class PeriodicEncoder1D(nn.Module):
    def __init__(self, n_freqs: int = 4, max_freq: float = 4.0):
        super().__init__()
        self.out_dim = 2 * n_freqs
        self.register_buffer("freqs", 2.0 ** torch.linspace(0, math.log2(max_freq), n_freqs))

    def forward(self, x: Tensor) -> Tensor:
        angles = x.unsqueeze(-1) * self.freqs * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


# The "sesqui" part. Inspired by rational/fractional resampling, we use PS to get to 2x,
# then use a learned downsample to reach our target. The original model was a fixed 1.5x;
# sesqui means "one and a half". While the fixed upscale is gone, the name remains.
class LowRankReassemblyHead(nn.Module):
    def __init__(self, channels: int, refine_expand: float = 1.5, rank: int = 2):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.K = 4

        self.register_buffer(
            "tap_offsets", torch.tensor([-1, 0, 1, 2], dtype=torch.long),
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

        taps = torch.zeros(rank, 2, channels, self.K)
        taps[0, :, :, :] = torch.tensor([-0.0625, 0.5625, 0.5625, -0.0625], dtype=taps.dtype)  # ~Bicubic start
        self.base_taps = nn.Parameter(taps)

        self.pe = PeriodicEncoder1D(n_freqs=4)
        self.axis_emb = nn.Embedding(2, 8)
        self.rank_emb = nn.Embedding(rank, 8)
        cond_dim = self.pe.out_dim + 1 + 8 + 8
        self.filter_net = nn.Sequential(
            nn.Linear(cond_dim, 64, bias=True),
            Activation(),
            nn.Linear(64, channels * self.K, bias=True),
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

    def _predict_weights(self, fy: Tensor, fx: Tensor,
                         log_foot_y: float, log_foot_x: float,
                         batch: int) -> tuple[Tensor, Tensor]:
        ny, nx = fy.numel(), fx.numel()
        R = self.rank
        n_axis = ny + nx

        frac_axis = torch.cat([fy, fx])
        log_foot = fy.new_full((n_axis, 1), log_foot_x)
        log_foot[:ny] = log_foot_y
        axis_id = torch.empty(n_axis, dtype=torch.long, device=fy.device)
        axis_id[:ny] = 0
        axis_id[ny:] = 1

        cond_axis = torch.cat([
            self.pe(frac_axis), log_foot, self.axis_emb(axis_id),
        ], dim=-1)

        rank_id = torch.arange(R, device=fy.device).repeat_interleave(n_axis)
        cond = torch.cat([cond_axis.repeat(R, 1), self.rank_emb(rank_id)], dim=-1)
        cond = cond.to(dtype=self.base_taps.dtype)

        residuals = self.filter_net(cond).view(R, n_axis, self.channels, self.K)
        residuals = 0.2 * torch.tanh(residuals)  # Again, bound the residual otherwise kaboom...

        residuals = residuals - residuals.mean(dim=-1, keepdim=True)

        wy = residuals[:, :ny] + self.base_taps[:, 0].unsqueeze(1)
        wx = residuals[:, ny:] + self.base_taps[:, 1].unsqueeze(1)

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


# Some models/VAEs can get away with very little in terms of trunks. For example, the Qwen/Wan21 VAE
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
        # No difference with a slight bottleneck here and smaller expand, saves a few parameters.
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
