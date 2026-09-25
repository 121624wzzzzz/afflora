#!/usr/bin/env python
"""Generate greedy IFEval responses for an existing corrected-SFT adapter run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from affine_vocab_lora import load_affine_vocab_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Inclusive global IFEval prompt index; enables non-overlapping shards.",
    )
    parser.add_argument(
        "--end-index", type=int, default=None,
        help="Exclusive global IFEval prompt index; defaults to the dataset end.",
    )
    parser.add_argument(
        "--affine-ablation",
        choices=("none", "zero_update", "zero_bias", "zero_all"),
        default="none",
        help=(
            "Diagnostic-only ablation after adapter loading. zero_update clears affine "
            "low-rank up matrices; zero_bias clears affine beta; zero_all applies both."
        ),
    )
    return parser.parse_args()


def render_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def load_run(run_dir: Path, device: torch.device, affine_ablation: str):  # noqa: ANN201
    args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
    model_path = args["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    if (run_dir / "affine_vocab_config.json").exists():
        model = load_affine_vocab_adapter(model, run_dir)
    if (run_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, run_dir)
    if affine_ablation != "none":
        ablated = []
        zero_update = affine_ablation in ("zero_update", "zero_all")
        zero_bias = affine_ablation in ("zero_bias", "zero_all")
        for name, parameter in model.named_parameters():
            if (zero_update and name.endswith(".affine.up.weight")) or (
                zero_bias and name.endswith(".affine.bias")
            ):
                parameter.data.zero_()
                ablated.append(name)
        if not ablated:
            raise RuntimeError(
                f"Requested affine ablation {affine_ablation!r}, but found no affine parameters."
            )
        print(f"Applied {affine_ablation} to {len(ablated)} affine tensors.", flush=True)
    model.to(device).eval()
    return model, tokenizer, args


@torch.inference_mode()
def generate(model, tokenizer, prompts, batch_size: int, max_new_tokens: int, device):  # noqa: ANN001
    responses = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        rendered = [render_prompt(tokenizer, row["prompt"]) for row in batch]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True).to(device)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = encoded["input_ids"].shape[1]
        continuations = generated[:, prompt_width:]
        responses.extend(tokenizer.batch_decode(continuations, skip_special_tokens=True))
        print(f"generated {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)
    return responses


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    device = torch.device(args.device)
    dataset = load_dataset("google/IFEval", split="train")
    rows = [dict(row) for row in dataset]
    if args.limit is not None:
        rows = rows[: args.limit]
    end_index = len(rows) if args.end_index is None else args.end_index
    if not (0 <= args.start_index <= end_index <= len(rows)):
        raise ValueError(
            f"Invalid shard [{args.start_index}, {end_index}) for {len(rows)} prompts."
        )
    rows = rows[args.start_index:end_index]

    model, tokenizer, run_args = load_run(run_dir, device, args.affine_ablation)
    responses = generate(
        model, tokenizer, rows, args.batch_size, args.max_new_tokens, device
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row, response in zip(rows, responses):
            stream.write(
                json.dumps(
                    {
                        "key": row["key"],
                        "prompt": row["prompt"],
                        "response": response,
                        "instruction_id_list": row["instruction_id_list"],
                        "kwargs": row["kwargs"],
                        "run_dir": str(run_dir),
                        "variant": run_args["variant"],
                        "seed": run_args["seed"],
                        "affine_ablation": args.affine_ablation,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"wrote {len(rows)} responses for shard [{args.start_index}, {end_index}) to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
