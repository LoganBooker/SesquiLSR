"""SesquiLSR ComfyUI node"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure the sesqui_lsr package is importable when ComfyUI loads this node.
_node_dir = Path(__file__).resolve().parent
if str(_node_dir) not in sys.path:
    sys.path.insert(0, str(_node_dir))

from sesqui_lsr import LatentUpscaler, make_flux, make_flux2, make_identity, make_ideogram4, make_sdxl
from sesqui_lsr.inference_adaptors import LatentFormatAdaptor

# ── Model / adaptor registry ────────────────────────────────────────────

_FORMAT_CFG: dict[str, dict] = {
    "SDXL": {
        "model_file": "upscaler_SDXL.safetensors",
        "in_channels": 4,
        "adaptor_fn": lambda: make_sdxl(),
    },
    "Flux": {
        "model_file": "upscaler_Flux.safetensors",
        "in_channels": 16,
        "adaptor_fn": lambda: make_flux(),
    },
    "Flux2": {
        "model_file": "upscaler_Flux2.safetensors",
        "in_channels": 32,
        "adaptor_fn": lambda: make_flux2(),
    },
    "Ideogram 4": {
        # Same 32ch VAE as Flux2; pipeline uses shift/scale patchify.
        "model_file": "upscaler_Flux2.safetensors",
        "in_channels": 32,
        "adaptor_fn": lambda: make_ideogram4(),
    },
    "Wan 2.1": {
        "model_file": "upscaler_Wan21.safetensors",
        "in_channels": 16,
        # ComfyUI LATENT nodes receive Wan/Qwen/Anima/Krea latents after
        # latent_format.process_out(...), so they are already back in raw
        # VAE latent space.
        "adaptor_fn": lambda: make_identity(16),
    },
}

# Single active model cache. Sesqui checkpoints are small (~3M params) and
# typical workflows only use one model family at a time, so we keep exactly
# one loaded copy and reload on format/dtype changes.
_active_key: tuple[str, torch.dtype] | None = None
_active_model: LatentUpscaler | None = None
_adaptor_cache: dict[str, LatentFormatAdaptor] = {}


# ── Helpers ──────────────────────────────────────────────────────────────


def _resolve_dtype(half_precision: bool, device: torch.device) -> torch.dtype:
    if not half_precision or device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _format_channel_hint(channels: int) -> str:
    mapping = {
        4: "SDXL",
        16: "Flux or Wan 2.1",
        32: "Flux2 VAE space",
        128: "Flux2 packed",
    }
    return mapping.get(channels, f"{channels}-channel")


def _flatten_to_4d(z: torch.Tensor, expected_channels: int) -> tuple[torch.Tensor, tuple]:
    """Flatten 4D/5D latents to 4D for the model, returning a layout tag for restoration."""
    if z.ndim == 4:
        return z, ("4d",)

    if z.ndim != 5:
        raise ValueError(f"Unsupported latent rank {z.ndim}; expected 4D or 5D tensor")

    # Video latents: [B, C, T, H, W] or [B, T, C, H, W].
    if z.shape[1] == expected_channels:
        b, c, t, h, w = z.shape
        return z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), ("bcthw", (b, c, t))

    if z.shape[2] == expected_channels:
        b, t, c, h, w = z.shape
        return z.reshape(b * t, c, h, w), ("btchw", (b, t, c))

    raise ValueError(
        f"Could not find {expected_channels}-channel axis "
        f"({_format_channel_hint(expected_channels)}) "
        f"in latent shape {tuple(z.shape)}"
    )


def _restore_from_4d(z4: torch.Tensor, layout: tuple) -> torch.Tensor:
    """Undo _flatten_to_4d, letting spatial dims come from the actual output."""
    kind = layout[0]
    if kind == "4d":
        return z4

    if kind == "bcthw":
        b, c, t = layout[1]
        return z4.reshape(b, t, c, *z4.shape[-2:]).permute(0, 2, 1, 3, 4)

    if kind == "btchw":
        b, t, c = layout[1]
        return z4.reshape(b, t, c, *z4.shape[-2:])

    raise ValueError(f"Unknown latent layout {kind!r}")


def _resize_mask(mask: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a noise mask, preserving any leading dimensions."""
    if mask.shape[-2:] == target_hw:
        return mask
    flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
    flat = torch.nn.functional.interpolate(flat, size=target_hw, mode="nearest")
    return flat.reshape(*mask.shape[:-2], *target_hw).to(mask.dtype)


def _find_model(filename: str) -> Path:
    """Locate a weight file: folder_paths -> models/ -> node root."""
    try:
        import folder_paths
        for d in folder_paths.get_folder_paths("sesqui_lsr"):
            p = Path(d) / filename
            if p.is_file():
                return p
    except (ImportError, KeyError):
        pass

    for d in (_node_dir / "models", _node_dir):
        p = d / filename
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"SesquiLSR weight file '{filename}' not found. "
        f"Place it in {_node_dir / 'models'} or add a 'sesqui_lsr' folder path."
    )


def _load_model(fmt: str, dtype: torch.dtype) -> tuple[LatentUpscaler, LatentFormatAdaptor]:
    """Load (or retrieve from cache) the model and adaptor for *fmt*."""
    global _active_key, _active_model

    key = (fmt, dtype)
    if _active_key != key or _active_model is None:
        cfg = _FORMAT_CFG[fmt]
        path = _find_model(cfg["model_file"])

        from safetensors.torch import load_file

        state_dict = load_file(str(path))
        model = LatentUpscaler(in_channels=cfg["in_channels"])
        model.load_state_dict(state_dict)
        model.to(dtype=dtype).eval().requires_grad_(False)

        _active_model = model
        _active_key = key

        if fmt not in _adaptor_cache:
            _adaptor_cache[fmt] = cfg["adaptor_fn"]()

    return _active_model, _adaptor_cache[fmt]


# ── ComfyUI node ─────────────────────────────────────────────────────────


class SesquiLatentUpscale:
    """Upscale latents with SesquiLSR at arbitrary scales in [1x, 2x]."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_format": (
                    ["SDXL", "Flux", "Flux2", "Ideogram 4", "Wan 2.1"],
                    {
                        "default": "SDXL",
                        "tooltip": (
                            "VAE latent configuration:\n"
                            "• SDXL (not compatible with SD 1.5)\n"
                            "• Flux (Flux, Z-Image Turbo, Lumina)\n"
                            "• Flux2 (Flux2, Flux2 Klein: BN-packed latent)\n"
                            "• Ideogram 4 (Flux2: shift/scale-packed latent)\n"
                            "• Wan 2.1 (Wan 2.x, Krea 2, Anima, Qwen Image)"
                        ),
                    },
                ),
                "scale": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05},
                ),
                "half_precision": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "latent/upscaling"
    OUTPUT_NODE = False

    def upscale(
        self,
        latent: dict,
        model_format: str,
        scale: float,
        half_precision: bool,
    ) -> tuple[dict]:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = _resolve_dtype(half_precision, device)
            model, adaptor = _load_model(model_format, dtype)

            samples = latent["samples"]
            if samples.ndim not in (4, 5):
                raise ValueError(
                    f"Expected a 4D or 5D latent, got shape {tuple(samples.shape)}"
                )

            # Target size in external (ComfyUI latent) coordinates.
            h, w = samples.shape[-2:]
            target_h, target_w = round(h * scale), round(w * scale)

            # Flatten to 4D so the adaptor always sees (B, C, H, W).
            ext_ch = adaptor.external_channels
            samples_4d, layout = _flatten_to_4d(samples, ext_ch)
            samples_fp32 = samples_4d.to(device=device, dtype=torch.float32)

            # Adaptor math in fp32, model inference in selected dtype.
            z_fp32 = adaptor.to_vae_latent(samples_fp32)
            vae_target = adaptor.vae_target_size((target_h, target_w))

            z = z_fp32.to(dtype=dtype)
            model.to(device=device)
            with torch.no_grad():
                z_up = model(z, vae_target)

            # Back through adaptor in fp32, then restore original layout.
            result_4d = adaptor.from_vae_latent(z_up.float()).to(
                device=samples.device, dtype=samples.dtype,
            )
            result = _restore_from_4d(result_4d, layout)

            out = {"samples": result}
            if "noise_mask" in latent:
                out["noise_mask"] = _resize_mask(latent["noise_mask"], result.shape[-2:])
            return (out,)

        except Exception as e:
            raise ValueError(
                f"SesquiLSR [{model_format}]: {e}  "
                f"(SDXL=4ch, Flux/Wan/Krea=16ch, Flux2/Ideogram4=128ch packed)"
            ) from e

# ── Registration ─────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SesquiLatentUpscale": SesquiLatentUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SesquiLatentUpscale": "Upscale Latent (SesquiLSR)",
}
