"""Qwen Image 2.1 decoder-feature latent transcoder for theNoise."""
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
    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNorm2D(channels)
        self.input = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.spatial = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            padding_mode="reflect",
            groups=channels,
            bias=False,
        )
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(F.mish(self.norm(x))).chunk(2, 1)
        value = self.spatial(F.mish(value) * 2.0 * torch.tanh(0.5 * gate))
        return x + self.output(value)


class Qwen21FeatureTranscoder(nn.Module):
    """Translate Qwen decoder-prefix features into a normalized 2× latent."""

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(1152, 384, 1)
        self.body = nn.Sequential(*(Qwen21SwiGLUStage(384) for _ in range(6)))
        self.out = nn.Conv2d(384, 64, 3, padding=1, padding_mode="reflect")
        self.skip = nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect")

    def forward(self, feature: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        size = (2 * latent.shape[-2], 2 * latent.shape[-1])
        if feature.shape[-2:] != size:
            raise ValueError("decoder-prefix feature must be exactly 2× the latent grid")
        hidden = self.body(self.head(feature))
        skip = F.interpolate(latent, size=size, mode="bicubic", align_corners=False)
        return self.skip(skip + self.out(hidden))


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
        return self.bridge(feature.to(self.dtype), z)


__all__ = ["Qwen21TranscodeUpscaler"]
