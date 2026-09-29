# theNoise Qwen Image 2.1 Transcoder Integration

This package replaces theNoise's Qwen Image 2.1 `VAEPixelUpscaler` with a fixed
2× decoder-feature latent bridge.

```text
normalized Qwen latent [B, 64, H, W]
→ existing Qwen VAE decoder prefix
→ decoder feature [B, 1152, 2H, 2W]
→ feature-to-latent bridge
→ normalized Qwen latent [B, 64, 2H, 2W]
```

No image decode/resize/re-encode round trip, latent adaptor, sampler change, or
pipeline change is required. The bundled model source and weights are a matched
release pair.

## Files

- `qwen21_transcode.py` — native theNoise `LatentUpscaler` implementation.
- `qwen21_transcode_2x.safetensors` — matching bridge weights.

## Integration

1. Copy `qwen21_transcode.py` to:

   ```text
   thenoise/upscale/qwen21_transcode.py
   ```

2. Copy `qwen21_transcode_2x.safetensors` to:

   ```text
   thenoise/upscale/weights/qwen21_transcode_2x.safetensors
   ```

3. In `thenoise/models/qwen_image21.py`, replace:

   ```python
   from thenoise.upscale import LatentUpscaler, VAEPixelUpscaler
   ```

   with:

   ```python
   from thenoise.upscale import LatentUpscaler
   from thenoise.upscale.qwen21_transcode import Qwen21TranscodeUpscaler
   ```

4. Add `from pathlib import Path`, then replace `_create_upscaler()` with:

   ```python
   def _create_upscaler(self) -> LatentUpscaler:
       return Qwen21TranscodeUpscaler(
           self.vae,
           Path(__file__).resolve().parent.parent
           / "upscale" / "weights" / "qwen21_transcode_2x.safetensors",
           device=self.device,
           dtype=self.dtype,
       )
   ```

The existing `thenoise.upscale` package-data rule includes `weights/**/*`, so no
packaging configuration change is required.

## Release contract

`Qwen21FeatureTranscoder` is authoritative. A future model revision can change
its private internals and ship matching weights without changing
`Qwen21TranscodeUpscaler`, its constructor, or the external fixed-2× latent
contract.

The current source expects a 30-tensor feature-only checkpoint. The previous
54-tensor checkpoint with an LR input trunk is intentionally incompatible.

## Smoke test

Run a normal Qwen Image 2.1 refined 2× upscale. The result should complete with
a normalized 64-channel latent at twice the input height and width, ready for
the existing refinement pass.
