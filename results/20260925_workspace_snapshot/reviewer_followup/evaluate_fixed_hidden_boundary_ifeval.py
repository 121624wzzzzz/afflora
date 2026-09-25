#!/usr/bin/env python
"""IFEval generation wrapper for fixed-hidden FP32-boundary control runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reviewer_followup"))

import evaluate_ifeval as evaluator  # noqa: E402
from train_corrected_sft_fixed_hidden_boundary_fp32 import (  # noqa: E402
    CONFIG_FILENAME,
    load_fixed_hidden_boundary_run,
)


ORIGINAL_LOAD_RUN = evaluator.load_run


def load_run(run_dir: Path, device: torch.device, affine_ablation: str):  # noqa: ANN201
    if not (run_dir / CONFIG_FILENAME).is_file():
        return ORIGINAL_LOAD_RUN(run_dir, device, affine_ablation)
    args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    model_path = args["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_fixed_hidden_boundary_run(model_path, run_dir, torch_dtype="auto")
    if affine_ablation != "none":
        if affine_ablation not in ("zero_update", "zero_all"):
            raise ValueError(
                "Fixed boundaries have no bias; supported ablations are "
                "none, zero_update, and zero_all"
            )
        boundary = next(
            module
            for module in model.modules()
            if module.__class__.__name__ == "FixedOutputBoundaryLMHead"
        )
        boundary.affine.up.weight.data.zero_()
        print("Applied fixed-boundary zero-update ablation.", flush=True)
    model.to(device).eval()
    return model, tokenizer, args


def main() -> None:
    evaluator.load_run = load_run
    evaluator.main()


if __name__ == "__main__":
    main()
