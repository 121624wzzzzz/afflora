#!/usr/bin/env python
"""Evaluate base or trained adapters with per-example assistant-only NLL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from affine_vocab_lora import load_affine_vocab_adapter  # noqa: E402
from data_pipeline import tokenize_conversation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_model(args: argparse.Namespace):  # noqa: ANN201
    run_args: dict[str, Any] = {}
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir:
        run_args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    model_path = args.model_path or run_args.get("model_path")
    if not model_path:
        raise ValueError("Provide --model-path for base evaluation or --run-dir for an adapter.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    if run_dir and (run_dir / "affine_vocab_config.json").exists():
        model = load_affine_vocab_adapter(model, run_dir)
    if run_dir and (run_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, run_dir)
    model.to(torch.device(args.device))
    model.eval()
    return model, tokenizer, str(model_path), run_args


def collate(items: list[dict[str, list[int]]], pad_token_id: int, device: torch.device):  # noqa: ANN201
    width = max(len(item["input_ids"]) for item in items)
    input_ids = []
    attention_mask = []
    labels = []
    for item in items:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def evaluate(model, tokenizer, rows, batch_size: int, max_seq_len: int, device: torch.device):  # noqa: ANN001, ANN201
    tokenized = [tokenize_conversation(row, tokenizer, max_seq_len) for row in rows]
    per_example: list[dict[str, Any]] = []
    total_nll = 0.0
    total_tokens = 0
    for begin in range(0, len(rows), batch_size):
        batch_rows = rows[begin : begin + batch_size]
        batch_items = tokenized[begin : begin + batch_size]
        input_ids, attention_mask, labels = collate(
            batch_items, tokenizer.pad_token_id, device
        )
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        losses = F.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
        )
        mask = shift_labels.ne(-100)
        row_sums = (losses * mask).sum(dim=1).cpu()
        row_counts = mask.sum(dim=1).cpu()
        for row, nll, count in zip(batch_rows, row_sums.tolist(), row_counts.tolist()):
            if count <= 0:
                raise RuntimeError("Encountered an evaluation row with zero supervised tokens.")
            per_example.append(
                {
                    "record_id": row.get("record_id"),
                    "source_file": row.get("source_file"),
                    "source_index": row.get("source_index"),
                    "nll_sum": nll,
                    "token_count": count,
                    "mean_ce": nll / count,
                }
            )
            total_nll += nll
            total_tokens += count
    avg_ce = total_nll / total_tokens
    return {
        "num_examples": len(rows),
        "supervised_tokens": total_tokens,
        "total_nll": total_nll,
        "avg_ce": avg_ce,
        "perplexity": math.exp(avg_ce),
        "per_example": per_example,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, tokenizer, model_path, run_args = load_model(args)
    rows = load_rows(Path(args.data))
    report = evaluate(
        model, tokenizer, rows, args.batch_size, args.max_seq_len, device
    )
    report.update(
        {
            "model_path": model_path,
            "run_dir": str(Path(args.run_dir).resolve()) if args.run_dir else None,
            "data": str(Path(args.data).resolve()),
            "variant": run_args.get("variant", "frozen_base"),
            "seed": run_args.get("seed"),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_example"}, indent=2))


if __name__ == "__main__":
    main()

