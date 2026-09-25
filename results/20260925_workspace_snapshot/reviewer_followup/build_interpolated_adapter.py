#!/usr/bin/env python
"""Build a minimal run directory by linearly interpolating two adapter states."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--right-weight", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def blend_state(left: Path, right: Path, output: Path, right_weight: float) -> None:
    a = load_file(str(left), device="cpu")
    b = load_file(str(right), device="cpu")
    if a.keys() != b.keys():
        raise ValueError(f"State keys differ: {left} vs {right}")
    result: dict[str, torch.Tensor] = {}
    for name in a:
        if a[name].shape != b[name].shape:
            raise ValueError(f"Shape mismatch for {name}: {a[name].shape} vs {b[name].shape}")
        result[name] = (1.0 - right_weight) * a[name].float() + right_weight * b[name].float()
    save_file(result, str(output))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.right_weight <= 1.0:
        raise ValueError("--right-weight must be in [0, 1]")
    left, right, output = map(Path, (args.left_run, args.right_run, args.output_dir))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    for name in ("run_args.json", "affine_vocab_config.json", "adapter_config.json"):
        shutil.copy2(right / name, output / name)
    blend_state(left / "affine_vocab_adapter.safetensors", right / "affine_vocab_adapter.safetensors", output / "affine_vocab_adapter.safetensors", args.right_weight)
    blend_state(left / "adapter_model.safetensors", right / "adapter_model.safetensors", output / "adapter_model.safetensors", args.right_weight)
    print(f"wrote {output} (right_weight={args.right_weight:g})")


if __name__ == "__main__":
    main()
