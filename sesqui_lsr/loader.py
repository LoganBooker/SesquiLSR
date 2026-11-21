"""Load a SesquiLSR checkpoint into the model class it was trained with."""

from typing import Mapping

from torch import Tensor, nn

from .upscaler import LatentUpscaler
from .upscaler_v1 import LatentUpscaler as V1LatentUpscaler


def load_upscaler(state_dict: Mapping[str, Tensor]) -> nn.Module:
    """Build the matching model, load ``state_dict`` strictly and return it in eval mode."""
    cls = V1LatentUpscaler if "upsample.base_taps" in state_dict else LatentUpscaler
    model = cls(in_channels=state_dict["head_conv.0.weight"].shape[1])
    model.load_state_dict(state_dict)
    return model.eval().requires_grad_(False)
