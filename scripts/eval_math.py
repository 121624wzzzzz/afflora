#!/usr/bin/env python
"""Evaluate a base model (with optional PEFT LoRA adapter) on GSM8K and MATH datasets
using generation-based accuracy.

Supports two tasks:
  - gsm8k: extracts the number after ``####`` (or the last number in the text).
  - math:  extracts the content inside ``\\boxed{...}`` (or the last number).

Usage:
    python scripts/eval_math.py \\
        --model-path /path/to/model \\
        --eval-data data/eval.jsonl \\
        --task gsm8k \\
        --adapter-dir /path/to/adapter \\
        --output-file results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from affine_vocab_lora import AffineVocabConfig, apply_affine_vocab_adapters, load_affine_vocab_adapter  # noqa: E402


PROMPT_TEMPLATE = """Question:
{question}

Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on GSM8K or MATH via generation accuracy."
    )
    parser.add_argument("--model-path", required=True, help="Path to HuggingFace model")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional path to PEFT LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--eval-data",
        required=True,
        help="Path to JSON/JSONL eval data file",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["gsm8k", "math"],
        help="Task: gsm8k or math — controls answer extraction strategy",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to save detailed per-sample results as JSON",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Which split of the dataset to use (default: train, works for JSONL files)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to this many samples",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--affine-adapter-dir",
        default=None,
        help="Optional path to saved affine vocab adapter (for AffLoRA variants)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation (default: 16)",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code in model loading")
    return parser.parse_args()


def first_present(row: dict[str, Any], keys: list[str]) -> str:
    """Return the first non-empty string value for any of *keys*."""
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def normalize_example(row: dict[str, Any]) -> tuple[str, str]:
    """Extract (question, answer) from a data row.

    Supports the same field conventions as ``train_affine_vocab_lora.py``.
    """
    question = first_present(row, ["query", "question", "problem", "instruction", "input", "prompt"])
    answer = first_present(row, ["answer", "response", "output", "solution", "target", "completion"])
    if not question or not answer:
        raise ValueError(
            "Each eval row must contain a question-like field and an answer-like field. "
            f"Found keys: {list(row.keys())}"
        )
    return question, answer


def extract_answer_gsm8k(text: str) -> str | None:
    """Extract the answer from a GSM8K generation.

    Priority:
      1. The number after ``####``.
      2. The last number found anywhere in the text.
    """
    # Look for #### marker
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fall back to the last number in the text
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1]

    return None


def extract_answer_math(text: str) -> str | None:
    """Extract the answer from a MATH generation.

    Priority:
      1. The content inside ``\\boxed{...}``.
      2. The last number found anywhere in the text.
    """
    # Look for \boxed{...}
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()

    # Fall back to the last number in the text
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, removes commas from numbers, and (for MATH) collapses
    LaTeX whitespace.
    """
    answer = answer.strip()
    # Remove commas in numbers
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)
    return answer


def normalize_math_answer(answer: str) -> str:
    """Normalize a MATH answer, removing LaTeX whitespace."""
    answer = normalize_answer(answer)
    # Collapse LaTeX whitespace
    answer = re.sub(r"\s+", "", answer)
    return answer


def main() -> None:
    args = parse_args()

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Model ----------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    # Load affine vocab adapter if provided (applies adapters + loads weights, before PEFT)
    if args.affine_adapter_dir:
        model = load_affine_vocab_adapter(model, args.affine_adapter_dir)

    # Load PEFT adapter if provided
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()

    # ---------- Data ----------
    ds = load_dataset("json", data_files=args.eval_data, split=args.dataset_split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # ---------- Answer extractor ----------
    if args.task == "gsm8k":
        extract_answer = extract_answer_gsm8k
        normalize_fn: Any = normalize_answer
    else:
        extract_answer = extract_answer_math
        normalize_fn = normalize_math_answer

    # Left-padding is required for correct batch generation with decoder-only models
    tokenizer.padding_side = "left"

    # ---------- Pre-extract questions and answers ----------
    samples: list[tuple[str, str]] = []
    for row in ds:
        question, answer = normalize_example(row)
        samples.append((question, answer))
    total = len(samples)
    batch_size = args.batch_size

    # ---------- Batch evaluation loop ----------
    correct = 0
    results: list[dict[str, Any]] = []

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_samples = samples[batch_start:batch_end]
        prompts = [PROMPT_TEMPLATE.format(question=q) for q, _ in batch_samples]
        ground_truths = [a for _, a in batch_samples]

        # Pre-compute per-prompt token lengths (without padding)
        prompt_lens = [
            tokenizer(p, return_tensors="pt")["input_ids"].shape[1]
            for p in prompts
        ]

        # Tokenize with left padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Process each sample in the batch
        for i, (question, ground_truth) in enumerate(batch_samples):
            prompt_len = prompt_lens[i]
            generated_ids = outputs[i][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            gt_extracted = extract_answer(ground_truth)
            prediction = extract_answer(generated_text)
            is_correct = False
            if prediction is not None and gt_extracted is not None:
                norm_pred = normalize_fn(prediction)
                norm_gt = normalize_fn(gt_extracted)
                is_correct = norm_pred == norm_gt

            if is_correct:
                correct += 1

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "generated_text": generated_text,
                "is_correct": is_correct,
            })

        processed = batch_end
        accuracy = correct / processed * 100
        print(f"[{processed}/{total}]  accuracy: {correct}/{processed} = {accuracy:.2f}%")

    # ---------- Final report ----------
    final_accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\nFinal accuracy: {correct}/{total} = {final_accuracy:.2f}%")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_path": args.model_path,
            "adapter_dir": args.adapter_dir,
            "task": args.task,
            "num_samples": total,
            "correct": correct,
            "accuracy_pct": round(final_accuracy, 2),
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
