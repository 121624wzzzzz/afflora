#!/usr/bin/env python
"""Check Trainer evaluation against the independent corrected-SFT evaluator.

This is a checkpoint-integrity diagnostic. It loads an already saved run and
uses the exact corrected tokenization in both paths, then writes both losses
and a small label-collator equivalence check as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT), str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "corrected_sft_experiment")
]

import train_affine_vocab_lora as legacy  # noqa: E402
from data_pipeline import tokenize_conversation  # noqa: E402
from corrected_sft_experiment.evaluate_corrected_sft import evaluate, load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--energy-lambda", type=float, default=0.0)
    parser.add_argument("--energy-tau", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, _, _ = load_model(
        SimpleNamespace(
            run_dir=args.run_dir,
            model_path=None,
            affine_ablation="none",
            device="cuda",
        )
    )
    raw = load_dataset("json", data_files=args.data, split="train")
    tokenized = raw.map(
        lambda row: tokenize_conversation(row, tokenizer, args.max_seq_len),
        remove_columns=raw.column_names,
        desc="Tokenizing parity data",
    )
    training_args = TrainingArguments(
        output_dir="/tmp/affine_trainer_evaluator_parity",
        per_device_eval_batch_size=args.batch_size,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    trainer = legacy.AffineLearningRateTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized,
        data_collator=collator,
        affine_energy_lambda=args.energy_lambda,
        affine_energy_tau=args.energy_tau,
    )
    trainer_metrics = trainer.evaluate()
    rows = [dict(row) for row in raw]
    manual = evaluate(
        model, tokenizer, rows, args.batch_size, args.max_seq_len, torch.device("cuda")
    )
    labels_match = True
    for index in range(min(10, len(rows))):
        batch = collator([tokenized[index]])
        expected = tokenize_conversation(rows[index], tokenizer, args.max_seq_len)
        width = len(expected["labels"])
        labels_match &= torch.equal(
            batch["labels"][0, :width].cpu(), torch.tensor(expected["labels"])
        )
    result = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "data": str(Path(args.data).resolve()),
        "trainer_eval_loss": float(trainer_metrics["eval_loss"]),
        "manual_avg_ce": float(manual["avg_ce"]),
        "absolute_difference": abs(float(trainer_metrics["eval_loss"]) - float(manual["avg_ce"])),
        "first_10_collator_labels_match": bool(labels_match),
        "energy_lambda": args.energy_lambda,
        "energy_tau": args.energy_tau,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
