#!/usr/bin/env python
"""Held-out CE evaluator for fixed-hidden FP32-boundary control runs.

All scoring, tokenization, sharding, and output formatting are delegated to the
existing corrected evaluator.  This wrapper changes only model reconstruction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corrected_sft_experiment"))
sys.path.insert(0, str(ROOT / "reviewer_followup"))

import evaluate_corrected_sft as evaluator  # noqa: E402
from train_corrected_sft_fixed_hidden_boundary_fp32 import (  # noqa: E402
    CONFIG_FILENAME,
    load_fixed_hidden_boundary_run,
)


ORIGINAL_LOAD_MODEL = evaluator.load_model


def load_model(args: Any):  # noqa: ANN201
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is None or not (run_dir / CONFIG_FILENAME).is_file():
        return ORIGINAL_LOAD_MODEL(args)
    run_args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    model_path = args.model_path or run_args.get("model_path")
    if not model_path:
        raise ValueError("The fixed-boundary run does not record a model path")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_fixed_hidden_boundary_run(model_path, run_dir, torch_dtype="auto")
    if args.affine_ablation != "none":
        if args.affine_ablation not in ("zero_update", "zero_all"):
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
    model.to(torch.device(args.device)).eval()
    return model, tokenizer, str(model_path), run_args


def main() -> None:
    evaluator.load_model = load_model
    evaluator.main()


if __name__ == "__main__":
    main()
