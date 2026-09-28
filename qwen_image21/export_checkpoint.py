"""Copy a final Qwen 2.1 transcoder checkpoint into the theNoise handoff.

The deployment model code is authoritative. The checkpoint must contain weights
for the exact Qwen21FeatureTranscoder implementation shipped beside it.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Final trainer .safetensors checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thenoise_integration/qwen21_transcode_2x.safetensors"),
    )
    args = parser.parse_args()

    if args.checkpoint.suffix != ".safetensors":
        parser.error("checkpoint must be a trainer safetensors file")

    state = load_file(str(args.checkpoint), device="cpu")
    if not state:
        raise SystemExit(f"{args.checkpoint} has no bridge tensors")
    expected = {"head.weight", "head.bias", "out.weight", "out.bias", "skip.weight"}
    for group in ("in_body", "body"):
        for index in range(6):
            prefix = f"{group}.{index}"
            expected.update({
                f"{prefix}.norm.weight",
                f"{prefix}.input.weight",
                f"{prefix}.spatial.weight",
                f"{prefix}.output.weight",
            })
            if group == "in_body" and index == 0:
                expected.add(f"{prefix}.shortcut.weight")
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise SystemExit(f"checkpoint keys do not match bundled model; missing={missing}, extra={extra}")
    shapes = {
        "head.weight": (384, 1152, 1, 1),
        "out.weight": (64, 384, 3, 3),
        "skip.weight": (64, 64, 3, 3),
        "in_body.0.norm.weight": (64,),
        "body.0.norm.weight": (384,),
    }
    for key, shape in shapes.items():
        if tuple(state[key].shape) != shape:
            raise SystemExit(f"unexpected {key} shape: {tuple(state[key].shape)} != {shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, args.output)
    print(f"exported {args.output}")
    print(f"tensors: {len(state)}")


if __name__ == "__main__":
    main()
