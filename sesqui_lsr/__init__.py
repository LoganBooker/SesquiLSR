"""
SesquiLSR - Latent Super-Resolution Upscaler.

Operates on VAE latents at arbitrary scales in [1x, 2x].
"""

from .inference_adaptors import (
    LatentFormatAdaptor,
    make_flux,
    make_flux2,
    make_ideogram4,
    make_identity,
    make_sdxl,
    make_wan21,
)
from .upscaler import LatentUpscaler

__all__ = [
    "LatentUpscaler",
    "LatentFormatAdaptor",
    "make_identity",
    "make_sdxl",
    "make_flux",
    "make_flux2",
    "make_ideogram4",
    "make_wan21",
]
