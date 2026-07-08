#!/usr/bin/env python
"""Correct, shardable generation evaluation for the full MATH test set."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    p for p in HERE.parents if (p / "data").is_dir() and (p / "scripts").is_dir()
)
CME_ROOT = PROJECT_ROOT / "corrected_math_evaluation"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from affine_vocab_lora import load_affine_vocab_adapter  # noqa: E402


PROMPT_TEMPLATE = "Question:\n{question}\n\nAnswer:\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--eval-data", default=str(PROJECT_ROOT / "data/math/test.jsonl"))
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def balanced_command_content(text: str, command: str) -> str | None:
    starts = list(re.finditer(rf"\\{command}\s*\{{", text))
    for match in reversed(starts):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            return text[start : index - 1].strip()
    return None


def extract_math_answer(text: str) -> str | None:
    for command in ("boxed", "fbox"):
        value = balanced_command_content(text, command)
        if value is not None:
            return value
    marker = re.findall(r"(?:the\s+answer\s+is|final\s+answer)\s*:?\s*([^\n]+)", text, re.I)
    if marker:
        return marker[-1].strip().rstrip(".")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def normalize_math_answer(answer: str) -> str:
    value = answer.strip().replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "")
    # Standard MATH answers often differ only by optional units or by compact
    # single-token LaTeX arguments (e.g. \\frac78 vs \\frac{7}{8}).
    value = re.sub(r"\\(?:text|mathrm|mbox)\s*\{[^{}]*\}", "", value)
    value = re.sub(r"\^\s*\{?\\circ\}?", "", value)
    value = re.sub(r"\\%", "%", value)
    value = re.sub(r"\\frac\s*([^\s{}])\s*([^\s{}])", r"\\frac{\1}{\2}", value)
    value = re.sub(r"\\sqrt\s*([^\s{}])", r"\\sqrt{\1}", value)
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.strip("$.")
    value = re.sub(r"^[a-zA-Z]+\s*=", "", value)
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_model(model_path: str, run_dir: Path | None):  # noqa: ANN201
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if run_dir and (run_dir / "affine_vocab_config.json").exists():
        model = load_affine_vocab_adapter(model, run_dir)
    if run_dir and (run_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, run_dir)
    model.eval()
    return model, tokenizer


def metric(results: list[dict[str, Any]], excluded: set[int] | None = None) -> dict[str, Any]:
    selected = [row for row in results if excluded is None or row["dataset_index"] not in excluded]
    correct = sum(row["is_correct"] for row in selected)
    return {
        "num_samples": len(selected),
        "correct": correct,
        "accuracy_pct": 100.0 * correct / len(selected) if selected else 0.0,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards).")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    run_args: dict[str, Any] = {}
    if run_dir:
        run_args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    model_path = args.model_path or run_args.get("model_path")
    if not model_path:
        raise ValueError("Provide --model-path or a --run-dir containing run_args.json.")

    all_rows = load_rows(Path(args.eval_data))
    indexed = list(enumerate(all_rows))
    if args.max_samples is not None:
        indexed = indexed[: args.max_samples]
    indexed = [item for item in indexed if item[0] % args.num_shards == args.shard_index]

    known = json.loads(
        (CME_ROOT / "shared/data_quality/known_contaminated_math_indices.json").read_text(
            encoding="utf-8"
        )
    )
    contaminated = set(known["indices"])
    model, tokenizer = load_model(str(model_path), run_dir)
    results: list[dict[str, Any]] = []
    started = time.time()

    for begin in range(0, len(indexed), args.batch_size):
        batch = indexed[begin : begin + args.batch_size]
        prompts = [PROMPT_TEMPLATE.format(question=row["question"]) for _, row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for offset, (dataset_index, row) in enumerate(batch):
            generated_ids = outputs[offset, input_width:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            ground_truth = extract_math_answer(row["answer"])
            prediction = extract_math_answer(generated_text)
            norm_gt = normalize_math_answer(ground_truth) if ground_truth is not None else None
            norm_pred = normalize_math_answer(prediction) if prediction is not None else None
            results.append(
                {
                    "dataset_index": dataset_index,
                    "is_known_contaminated": dataset_index in contaminated,
                    "question": row["question"],
                    "ground_truth": row["answer"],
                    "ground_truth_extracted": ground_truth,
                    "prediction": prediction,
                    "normalized_ground_truth": norm_gt,
                    "normalized_prediction": norm_pred,
                    "generated_text": generated_text,
                    "is_correct": norm_gt is not None and norm_pred is not None and norm_gt == norm_pred,
                }
            )
        done = min(begin + args.batch_size, len(indexed))
        elapsed = time.time() - started
        print(
            f"[{done}/{len(indexed)}] shard={args.shard_index}/{args.num_shards} "
            f"samples_per_second={done / max(elapsed, 1e-9):.3f}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "model_path": str(model_path),
        "run_dir": str(run_dir) if run_dir else None,
        "variant": run_args.get("variant", "frozen_base"),
        "seed": run_args.get("seed"),
        "eval_data": str(Path(args.eval_data).resolve()),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "elapsed_seconds": time.time() - started,
        "full": metric(results),
        "clean": metric(results, contaminated),
        "results": results,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
