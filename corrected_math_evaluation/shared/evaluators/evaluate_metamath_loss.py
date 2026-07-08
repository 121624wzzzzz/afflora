#!/usr/bin/env python
"""Evaluate answer-token CE loss on MetaMathQA-style JSONL data.

This uses the same prompt/answer tokenization and label masking as
scripts/train_affine_vocab_lora.py, then reports loss over non-masked answer
tokens. It supports PEFT hidden LoRA and affine vocab adapters saved by the
training script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "data").is_dir() and (p / "scripts").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from affine_vocab_lora import load_affine_vocab_adapter  # noqa: E402
from train_affine_vocab_lora import tokenize_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--eval-data", default=str(PROJECT_ROOT / "data/metamathqa_40k/eval.jsonl"))
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_model(model_path: str, run_dir: Path | None, dtype: torch.dtype, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if run_dir and (run_dir / "affine_vocab_config.json").exists():
        model = load_affine_vocab_adapter(model, run_dir)
    if run_dir and (run_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, run_dir)
    model.eval()
    return model, tokenizer


def collate(features: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in features)
    input_ids = []
    attention_mask = []
    labels = []
    for row in features:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    run_args: dict[str, Any] = {}
    if run_dir:
        run_args_path = run_dir / "run_args.json"
        if run_args_path.exists():
            run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
    model_path = args.model_path or run_args.get("model_path")
    if not model_path:
        raise ValueError("Provide --model-path or --run-dir containing run_args.json.")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model, tokenizer = load_model(str(model_path), run_dir, dtype, args.trust_remote_code)
    rows = load_rows(Path(args.eval_data))
    features = [tokenize_row(row, tokenizer, args.max_seq_len) for row in rows]

    total_loss = 0.0
    total_tokens = 0
    total_sequences = 0
    started = time.time()
    for begin in range(0, len(features), args.batch_size):
        batch_features = features[begin : begin + args.batch_size]
        batch = collate(batch_features, tokenizer.pad_token_id)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        labels = batch.pop("labels")
        with torch.inference_mode():
            logits = model(**batch).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels.ne(-100)
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(loss_sum.item())
        total_tokens += int(mask.sum().item())
        total_sequences += len(batch_features)
        print(f"[{total_sequences}/{len(features)}] tokens={total_tokens}", flush=True)

    loss = total_loss / max(total_tokens, 1)
    report = {
        "schema_version": 1,
        "model_path": str(model_path),
        "run_dir": str(run_dir) if run_dir else None,
        "variant": run_args.get("variant", "frozen_base"),
        "seed": run_args.get("seed"),
        "eval_data": str(Path(args.eval_data).resolve()),
        "num_samples": len(rows),
        "answer_tokens": total_tokens,
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "elapsed_seconds": time.time() - started,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
