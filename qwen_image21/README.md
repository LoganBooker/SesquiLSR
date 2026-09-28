# theNoise Qwen Image 2.1 Transcoder Integration

This package replaces theNoise's Qwen Image 2.1 `VAEPixelUpscaler` with a trained
latent-only 2× bridge.

The bridge accepts and returns theNoise's existing canonical normalized Qwen
latent. The bundled model code and weights are a matched release pair; changing
the internals only requires shipping updated code and matching weights.


```text
[B, 64, H, W] -> [B, 64, 2H, 2W]
```

No pipeline, sampler, latent adaptor, or VAE round trip is required.

## Files

- `qwen21_transcode.py` — native theNoise `LatentUpscaler` implementation.
- `export_checkpoint.py` — checks that a checkpoint matches the bundled preview model and copies it for handoff.

## Integration

1. Copy `qwen21_transcode.py` into:

   ```text
   thenoise/upscale/qwen21_transcode.py
   ```

2. Copy the final trained bridge file to:

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

4. Replace `_create_upscaler()` with:

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

   Add `from pathlib import Path` to that module.

The existing `thenoise.upscale` package-data rule already includes
`weights/**/*`, so no packaging configuration change is needed.

The deployment module does not reconstruct architecture from checkpoint metadata.
`Qwen21FeatureTranscoder` is authoritative. A later model revision can change that
private class and ship matching weights without changing `Qwen21TranscodeUpscaler`,
its constructor, or the pipeline's latent contract.

## Smoke test

Run a normal Qwen Image 2.1 refined 2× upscale. The output should complete
without conversion code or pipeline changes. The upscaler receives the same
normalized 64-channel latent the Qwen DiT uses and returns a normalized 64-channel
latent at double height and width, ready for the existing refine pass.
