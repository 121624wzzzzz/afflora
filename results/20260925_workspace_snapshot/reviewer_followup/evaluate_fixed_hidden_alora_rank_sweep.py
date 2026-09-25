#!/usr/bin/env python
"""Corrected-SFT CE evaluator for generalized fixed-hidden A-LoRA ranks."""

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
import train_corrected_sft_fixed_hidden_boundary_fp32 as base  # noqa: E402


ORIGINAL_LOAD_MODEL = evaluator.load_model
SWEEP_MARKER = "fixed_hidden_output_boundary_fp32_rank_sweep_v1"


def configure_rank_loader(config: dict[str, Any]) -> None:
    if config.get("marker") != SWEEP_MARKER:
        return
    rank = int(config["rank"])
    count = int(config["trainable_parameters"])
    base.CONTROL_MARKER = SWEEP_MARKER
    base.EXPECTED_COUNTS["alora"] = count

    def assert_sweep_budget(boundary) -> int:  # noqa: ANN001
        actual = sum(parameter.numel() for parameter in boundary.affine.parameters())
        expected = 2 * base.EXPECTED_HIDDEN_SIZE * rank
        if (
            boundary.kind != "alora"
            or boundary.rank != rank
            or boundary.hidden_size != base.EXPECTED_HIDDEN_SIZE
            or boundary.vocab_size != base.EXPECTED_VOCAB_SIZE
            or actual != count
            or count != expected
        ):
            raise RuntimeError("Saved sweep boundary failed rank/budget reconstruction")
        return actual

    base._assert_qwen_budget = assert_sweep_budget


def load_model(args: Any):  # noqa: ANN201
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is None or not (run_dir / base.CONFIG_FILENAME).is_file():
        return ORIGINAL_LOAD_MODEL(args)
    config = json.loads(
        (run_dir / base.CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    configure_rank_loader(config)
    run_args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    model_path = args.model_path or run_args.get("model_path")
    if not model_path:
        raise ValueError("The fixed-boundary run does not record a model path")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = base.load_fixed_hidden_boundary_run(
        model_path, run_dir, torch_dtype="auto"
    )
    if args.affine_ablation != "none":
        if args.affine_ablation not in ("zero_update", "zero_all"):
            raise ValueError("The sweep boundary has no bias")
        boundary = base.find_fixed_boundary(model)
        boundary.affine.up.weight.data.zero_()
    model.to(torch.device(args.device)).eval()
    return model, tokenizer, str(model_path), run_args


def main() -> None:
    evaluator.load_model = load_model
    evaluator.main()


if __name__ == "__main__":
    main()
