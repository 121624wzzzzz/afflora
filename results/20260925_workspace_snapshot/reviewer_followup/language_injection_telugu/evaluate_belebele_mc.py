#!/usr/bin/env python
"""Score Telugu Belebele by likelihood of its four exact option-number labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from affine_vocab_lora import load_affine_vocab_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_run(run_dir: Path):
    args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args["model_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args["model_path"], torch_dtype="auto")
    if (run_dir / "affine_vocab_config.json").exists():
        model = load_affine_vocab_adapter(model, run_dir)
    if (run_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, run_dir)
    return model.cuda().eval(), tokenizer


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    model, tokenizer = load_run(run_dir)
    label_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 5)]
    if not all(len(ids) == 1 for ids in label_ids):
        raise RuntimeError(f"Expected one token per option label, got {label_ids}")
    label_ids = torch.tensor([ids[0] for ids in label_ids], device="cuda")
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            messages = [row["conversations"][:-1] for row in batch]
            prompts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for m in messages
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            logits = model(**encoded).logits[:, -1, :]
            scores = torch.log_softmax(logits, dim=-1).index_select(1, label_ids).float().cpu().tolist()
            for row, score in zip(batch, scores):
                gold = int(row["conversations"][-1]["content"])
                pred = int(max(range(4), key=lambda i: score[i])) + 1
                outputs.append({"record_id": row["record_id"], "gold": gold, "prediction": pred, "correct": pred == gold, "logprobs": score})
    result = {
        "run_dir": str(run_dir),
        "n": len(outputs),
        "accuracy": sum(row["correct"] for row in outputs) / len(outputs),
        "rows": outputs,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("run_dir", "n", "accuracy")}))


if __name__ == "__main__":
    main()
