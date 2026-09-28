"""Qwen Image 2.1 latent transcoder for theNoise."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from thenoise.upscale.base import LatentUpscaler


class RMSNorm2D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(1, keepdim=True) + self.eps)
        return x * scale.to(x.dtype) * self.weight.to(x.dtype).view(1, -1, 1, 1)


class Qwen21SwiGLUStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, expansion: float = 1.0):
        super().__init__()
        hidden = max(1, round(out_channels * expansion))
        self.norm = RMSNorm2D(in_channels)
        self.input = nn.Conv2d(in_channels, 2 * hidden, 1, bias=False)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.output = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(F.mish(self.norm(x))).chunk(2, 1)
        return self.shortcut(x) + self.output(self.spatial(F.mish(value) * gate))


class Qwen21FeatureTranscoder(nn.Module):
    """The fixed 384-wide 2× bridge paired with the exported weights."""

    def __init__(self):
        super().__init__()
        feature_channels, latent_channels, width = 1152, 64, 384
        depth, expansion = 6, 1.0
        self.head = nn.Conv2d(feature_channels, width, 1)
        self.in_body = nn.Sequential(*(
            Qwen21SwiGLUStage(latent_channels if i == 0 else width, width, expansion=expansion)
            for i in range(depth)
        ))
        self.body = nn.Sequential(*(
            Qwen21SwiGLUStage(width, width, expansion=expansion)
            for _ in range(depth)
        ))
        self.out = nn.Conv2d(width, latent_channels, 3, padding=1)
        self.skip = nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False)

    def forward(self, feature: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        size = feature.shape[-2:]
        if size != (2 * latent.shape[-2], 2 * latent.shape[-1]):
            raise ValueError("decoder-prefix feature must be exactly 2× the latent grid")
        trunk = F.interpolate(self.in_body(latent), size=size, mode="bicubic", align_corners=False)
        hidden = self.body(self.head(feature) + trunk)
        skip = F.interpolate(latent, size=size, mode="bicubic", align_corners=False)
        return self.skip(skip) + self.out(hidden)


class Qwen21TranscodeUpscaler(LatentUpscaler):
    """Normalized Qwen latent [B,64,H,W] → normalized 2× latent."""

    scale = 2

    def __init__(self, vae: Any, checkpoint_path: str | Path, *, device, dtype):
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype
        self.bridge = Qwen21FeatureTranscoder()
        self.bridge.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
        self.bridge.to(self.device, dtype).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(self.device, self.dtype)
        if z.ndim != 4 or z.shape[1] != 64:
            raise ValueError(f"expected [B,64,H,W], got {tuple(z.shape)}")
        mean = self.vae._latents_mean.to(self.device, z.dtype)
        inv_std = self.vae._latents_inv_std.to(self.device, z.dtype)
        raw = z / inv_std + mean
        feature = self.vae.conv2(raw)
        feature = self.vae.decoder.conv1(feature)
        feature = self.vae.decoder.middle(feature)
        feature = self.vae.decoder.upsamples[0](feature)
        return self.bridge(
            feature.to(dtype=self.dtype),
            z.to(dtype=self.dtype),
        )


__all__ = ["Qwen21TranscodeUpscaler"]
