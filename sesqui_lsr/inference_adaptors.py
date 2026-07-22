"""Latent-format adaptors for Sesqui inference.

Each adaptor converts between a pipeline's external latent space (what the
diffusion model passes around) and the canonical VAE latent space that
Sesqui models are trained on (as defined by the VAE adaptor).

Adaptors are only needed when the pipeline modifies the VAE output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ── Patchify helpers (shared by Flux2-family formats) ─────────────────────


def _patchify(z: Tensor) -> Tensor:
    """32ch -> 128ch via 2x2 spatial packing."""
    B, C, H, W = z.shape
    return (
        z.float()
        .reshape(B, C, H // 2, 2, W // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(B, C * 4, H // 2, W // 2)
    )


def _unpatchify(z: Tensor) -> Tensor:
    """128ch -> 32ch via 2x2 spatial unpacking."""
    B, C, H, W = z.shape
    return (
        z.float()
        .reshape(B, C // 4, 2, 2, H, W)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(B, C // 4, H * 2, W * 2)
    )


# ── Base class ────────────────────────────────────────────────────────────


class LatentFormatAdaptor:
    """Convert between an external pipeline's latent and VAE latent.

    Subclass or instantiate via the ``make_*`` constructors below.

    Parameters
    ----------
    external_channels : int
        Channel count of the external/pipeline latent.
    spatial_scale : int, default 1
        Spatial multiplier between external and VAE-latent coords.
        Flux2-family formats use ``2`` (VAE latent is 2x spatially larger
        because the pipeline packs 2x2 into channels).  All others use ``1``.
    """

    def __init__(self, external_channels: int, spatial_scale: int = 1):
        self.external_channels = external_channels
        self.spatial_scale = spatial_scale

    def to_vae_latent(self, z: Tensor) -> Tensor:
        """External/pipeline latent -> VAE latent."""
        return z.float()

    def from_vae_latent(self, z: Tensor) -> Tensor:
        """VAE latent -> external/pipeline latent."""
        return z

    def vae_target_size(self, target_size: tuple[int, int]) -> tuple[int, int]:
        """Target size in external coords -> VAE-latent coords."""
        h, w = target_size
        s = self.spatial_scale
        return (h * s, w * s) if s != 1 else target_size


# ── Constructors ──────────────────────────────────────────────────────────


def make_identity(external_channels: int) -> LatentFormatAdaptor:
    """Adaptor for formats where the external latent IS the model latent.

    Wan 2.1, Anima, Qwen Image, Krea 2, and any other model where ComfyUI's
    latent_format does not apply a scaling transform (process_in/out are
    identity or cancel out before reaching custom LATENT nodes).
    """
    return LatentFormatAdaptor(external_channels=external_channels)


def make_sdxl() -> LatentFormatAdaptor:
    """SDXL - 4ch VAE latent with 0.13025 scaling.

    ComfyUI stores raw VAE latents between nodes; KSampler applies
    process_in (x0.13025) / process_out (/0.13025) around the model.
    Sesqui was trained on the scaled latents.
    """
    return _AffineAdaptor(external_channels=4, scale=0.13025, shift=0.0)


def make_flux() -> LatentFormatAdaptor:
    """Flux / Z-Image Turbo / Lumina - 16ch VAE latent with shift/scale.

    ComfyUI stores raw VAE latents between nodes; KSampler applies
    process_in ((z - 0.1159) x 0.3611) / process_out (z / 0.3611 + 0.1159).
    Sesqui was trained on the scaled latents.
    """
    return _AffineAdaptor(external_channels=16, scale=0.3611, shift=0.1159)


def make_flux2(
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    eps: float = 1e-4,
) -> LatentFormatAdaptor:
    """Flux2 - 32ch raw VAE ↔ 128ch patched + BN pipeline latent.

    Uses BN statistics from the Flux2 VAE's ``bn`` module.
    Defaults are extracted from the official Flux2 VAE checkpoint.
    """
    if running_mean is None:
        running_mean = torch.tensor([
            -0.06762, -0.07152, -0.07534, -0.07449,  0.02228,  0.01800,  0.01420,  0.01836,
            -0.00006, -0.00625, -0.00021, -0.00314, -0.02720, -0.02811, -0.02765, -0.02903,
            -0.07689, -0.06717, -0.09019, -0.08921,  0.01684,  0.01521,  0.00790,  0.00858,
             0.00835,  0.00154,  0.00026, -0.00428, -0.04388, -0.04190, -0.04378, -0.04315,
            -0.01025, -0.01319, -0.00662, -0.00477, -0.03106, -0.03055, -0.02790, -0.01795,
             0.00302,  0.00150,  0.01259,  0.01447,  0.03472,  0.03377,  0.03366,  0.02830,
             0.00198,  0.00473,  0.00465,  0.00496,  0.01227,  0.00810,  0.00806,  0.01458,
             0.06811,  0.06790,  0.07665,  0.07319, -0.04621, -0.04739, -0.03919, -0.05109,
            -0.05278, -0.04774, -0.04700, -0.05172, -0.03171, -0.03163, -0.03447, -0.02826,
             0.05097,  0.04450,  0.05781,  0.04580, -0.04116, -0.04583, -0.04874, -0.04674,
            -0.00884, -0.01063, -0.00881, -0.00461, -0.03758, -0.04322, -0.04357, -0.04989,
             0.01185,  0.01664,  0.02028,  0.02790,  0.01127,  0.01290,  0.00156,  0.00716,
            -0.01180, -0.00184, -0.01414, -0.00537, -0.00910, -0.01380, -0.01447, -0.01870,
             0.03225,  0.03050,  0.02587,  0.02996,  0.05400,  0.06144,  0.04954,  0.05899,
            -0.05108, -0.06033, -0.04778, -0.05240, -0.02268, -0.02742, -0.01537, -0.02546,
            -0.05721, -0.05648, -0.05176, -0.04956,  0.01159,  0.00542,  0.01630,  0.01038,
        ])
    if running_var is None:
        running_var = torch.tensor([
            3.25021,  3.16341,  3.19243,  3.18137,  3.13891,  3.09414,  3.10118,  3.05509,
            3.00518,  3.01795,  3.00676,  3.00764,  3.46902,  3.43252,  3.47023,  3.45539,
            3.09498,  3.07138,  3.08192,  3.09134,  3.01471,  3.02746,  3.01199,  3.02529,
            3.00746,  2.97413,  3.02488,  2.99405,  3.08042,  3.06691,  3.08315,  3.05815,
            3.40362,  3.40553,  3.44087,  3.43550,  3.32671,  3.17300,  3.18745,  3.22017,
            3.25698,  3.19532,  3.13096,  3.12421,  3.16203,  3.12096,  3.21296,  3.18538,
            3.09027,  3.03003,  3.05658,  3.01625,  3.22585,  3.23913,  3.21108,  3.21309,
            3.16103,  3.14950,  3.14238,  3.15017,  3.07164,  3.04400,  3.11775,  3.06079,
            3.15937,  3.13995,  3.17299,  3.17302,  3.29846,  3.24451,  3.24831,  3.25173,
            3.07203,  3.00360,  3.08447,  3.05619,  3.10095,  3.06496,  3.12614,  3.10201,
            3.12051,  3.07826,  3.17810,  3.14189,  3.20242,  3.23967,  3.19097,  3.15400,
            3.10219,  3.10638,  3.08341,  3.08930,  3.16211,  3.12266,  3.17199,  3.16812,
            2.95874,  2.91292,  2.98084,  2.92094,  3.16569,  3.08972,  3.06321,  3.04655,
            3.09284,  3.06227,  3.07098,  3.01419,  3.10315,  3.08778,  3.04287,  3.03801,
            3.06550,  3.10085,  3.10954,  3.10174,  2.97687,  2.93585,  2.99999,  2.96735,
            3.12007,  3.10587,  3.13934,  3.12008,  3.04748,  3.04194,  3.08653,  3.07292,
        ])
    return _Flux2BNAdaptor(running_mean=running_mean, running_var=running_var, eps=eps)


def make_ideogram4(
    shift: Tensor | None = None,
    scale: Tensor | None = None,
) -> LatentFormatAdaptor:
    """Ideogram 4 - 32ch raw VAE ↔ 128ch patched + shift/scale pipeline latent.

    Uses the same Flux2 VAE with 2x2 patchify, but applies per-channel
    shift+scale normalization instead of BatchNorm.
    """
    if shift is None:
        shift = torch.tensor([
            0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549,
            0.05021099, -0.15083604, -0.15360136, -0.20131799, 0.01922352, 0.06226260,
            0.10140969, -0.06739428, 0.37582610, -0.23371200, 0.35164491, -0.02590912,
            -0.02719350, -0.10833897, -0.14768480, -0.01130957, -0.22983720, 0.23526423,
            -0.10893522, 0.11957631, 0.04047799, 0.31345890, -0.17225064, -0.18646109,
            -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294, 0.26952152,
            -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.15485020, -0.18971131,
            0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937,
            0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137,
            -0.21863696, 0.23228983, -0.11739769, 0.11693044, 0.02563311, 0.31356594,
            -0.17420591, -0.19006285, -0.34905377, -0.04025005, 0.01924137, 0.07652984,
            0.29956080, 0.26280570, -0.22011674, -0.12715361, 0.04879879, -0.14075719,
            -0.15935895, -0.21235840, 0.01974813, 0.05523547, 0.10011992, -0.06428964,
            0.37781868, -0.21491644, 0.34254215, -0.03153528, -0.03100820, -0.10761415,
            -0.14730405, -0.02475182, -0.22855880, 0.25150810, -0.10445128, 0.12446000,
            0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.01291770,
            0.02578168, 0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155,
            0.04059549, -0.14043529, -0.16111187, -0.20203050, 0.02602069, 0.04852717,
            0.10432153, -0.06309942, 0.38402443, -0.22397003, 0.34814481, -0.03774432,
            -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016, 0.24872463,
            -0.11399775, 0.12226870, 0.05620835, 0.30917800, -0.18065738, -0.19401479,
            -0.34495114, -0.01760592,
        ])
    if scale is None:
        scale = torch.tensor([
            1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.66753160, 1.69059584,
            1.56853198, 1.62314944, 1.89106626, 1.58086668, 1.60822129, 1.60962993,
            1.63322129, 1.56074359, 1.73419528, 1.79192650, 1.64040632, 1.66802808,
            1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046,
            1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641,
            1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756, 1.91310663,
            1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539,
            1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606,
            1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667,
            1.61659342, 1.59722044, 1.64103121, 1.54085310, 1.68610394, 1.67772755,
            1.78998563, 1.66621713, 1.65458955, 1.66041308, 1.64710857, 1.68163503,
            1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
            1.87618195, 1.55843750, 1.57438785, 1.61711053, 1.63094305, 1.55644029,
            1.73124302, 1.80666627, 1.64636210, 1.65932006, 1.60816188, 1.75682671,
            1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068,
            1.65534289, 1.67132281, 1.80317197, 1.67673140, 1.65700938, 1.68426259,
            1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117,
            1.57309060, 1.61928553, 1.87148809, 1.56244866, 1.56697152, 1.61584394,
            1.62759496, 1.55480378, 1.73484107, 1.79055143, 1.64688773, 1.66121492,
            1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592, 1.60792883,
            1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
            1.65655173, 1.68533454,
        ])
    return _ShiftScalePatchAdaptor(shift=shift, scale=scale)


def make_wan21(
    latents_mean: Tensor | None = None,
    latents_std: Tensor | None = None,
) -> LatentFormatAdaptor:
    """Wan21 / Anima / QwenImage - per-channel z-score.

    Pipeline latent = ``(VAE_raw - mean) / std``.
    Default stats from the Wan21 latent format.
    """
    if latents_mean is None:
        latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
             0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
        ])
    if latents_std is None:
        latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ])
    return _ZScoreAdaptor(
        external_channels=16,
        mean=latents_mean,
        std=latents_std,
    )


# ── Internal implementations ──────────────────────────────────────────────


class _AffineAdaptor(LatentFormatAdaptor):
    """Affine transform: (z - shift) * scale and inverse.

    Used by SDXL and Flux where ComfyUI stores raw VAE latents between nodes
    but the diffusion model (and Sesqui) operates on scaled latents.
    """

    def __init__(self, external_channels: int, scale: float, shift: float):
        super().__init__(external_channels=external_channels)
        self.scale = scale
        self.shift = shift

    def to_vae_latent(self, z: Tensor) -> Tensor:
        return (z.float() - self.shift) * self.scale

    def from_vae_latent(self, z: Tensor) -> Tensor:
        return z.float() / self.scale + self.shift


class _ZScoreAdaptor(LatentFormatAdaptor):
    """Per-channel z-score normalization (Wan21 / Anima / QwenImage)."""

    def __init__(self, external_channels: int, mean: Tensor, std: Tensor):
        super().__init__(external_channels=external_channels)
        self.mean = mean.clone().view(1, -1, 1, 1)
        self.std = std.clone().view(1, -1, 1, 1)

    def _cast(self, z: Tensor) -> tuple[Tensor, Tensor]:
        d = z.device
        return (
            self.mean.to(dtype=z.dtype, device=d),
            self.std.to(dtype=z.dtype, device=d),
        )

    def to_vae_latent(self, z: Tensor) -> Tensor:
        # Pipeline latent = (VAE_raw - mean) / std
        # -> VAE_raw = z * std + mean
        m, s = self._cast(z)
        return z.float() * s + m

    def from_vae_latent(self, z: Tensor) -> Tensor:
        m, s = self._cast(z)
        return (z.float() - m) / s


class _Flux2BNAdaptor(LatentFormatAdaptor):
    """Flux2: 32ch raw VAE ↔ 128ch patched + BN-normalized pipeline."""

    def __init__(self, running_mean: Tensor, running_var: Tensor, eps: float = 1e-4):
        super().__init__(external_channels=128, spatial_scale=2)
        self.bn_mean = running_mean.clone().float().view(1, -1, 1, 1)
        self.bn_var = running_var.clone().float().view(1, -1, 1, 1)
        self.bn_eps = eps

    def to_vae_latent(self, z: Tensor) -> Tensor:
        # Pipeline = BN(patchify(VAE_raw))
        # Undo BN: x * sqrt(var) + mean
        var = self.bn_var.to(dtype=z.dtype, device=z.device)
        mean = self.bn_mean.to(dtype=z.dtype, device=z.device)
        z = z.float() * torch.sqrt(var + self.bn_eps) + mean
        return _unpatchify(z)

    def from_vae_latent(self, z: Tensor) -> Tensor:
        z = _patchify(z)
        var = self.bn_var.to(dtype=z.dtype, device=z.device)
        mean = self.bn_mean.to(dtype=z.dtype, device=z.device)
        return F.batch_norm(
            z,
            running_mean=mean.squeeze(),
            running_var=var.squeeze(),
            momentum=0.0,
            eps=self.bn_eps,
        )


class _ShiftScalePatchAdaptor(LatentFormatAdaptor):
    """Ideogram 4: 32ch raw VAE <-> 128ch patched + per-channel shift/scale.

    Pipeline latent = ``patchify(VAE_raw) * scale + shift``.
    """

    def __init__(self, shift: Tensor, scale: Tensor):
        super().__init__(external_channels=128, spatial_scale=2)
        self.shift = shift.clone().float().view(1, -1, 1, 1)
        self.scale = scale.clone().float().view(1, -1, 1, 1)

    def _cast(self, z: Tensor) -> tuple[Tensor, Tensor]:
        d = z.device
        return (
            self.shift.to(dtype=z.dtype, device=d),
            self.scale.to(dtype=z.dtype, device=d),
        )

    def to_vae_latent(self, z: Tensor) -> Tensor:
        # Pipeline = patchify(VAE_raw) * scale + shift
        # -> VAE_raw = unpatchify((z - shift) / scale)
        s, sc = self._cast(z)
        return _unpatchify((z.float() - s) / sc)

    def from_vae_latent(self, z: Tensor) -> Tensor:
        s, sc = self._cast(z)
        return _patchify(z) * sc + s
