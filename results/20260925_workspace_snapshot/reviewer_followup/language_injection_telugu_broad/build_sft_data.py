#!/usr/bin/env python
"""Merge translated shards and enforce the Qwen training-token budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-rows", type=int, default=22780)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = []
    for shard in sorted(Path(args.shards_dir).glob("*.jsonl")):
        rows.extend(json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line)
    rows.sort(key=lambda row: int(row["source_index"]))
    seen = set()
    accepted = []
    rejected = []
    for row in rows:
        key = row["source_index"]
        if key in seen:
            raise ValueError(f"duplicate source index {key}")
        seen.add(key)
        conversations = [
            {"role": "user", "content": row["telugu_user"]},
            {"role": "assistant", "content": row["telugu_assistant"]},
        ]
        text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        destination = accepted if len(encoded) <= args.max_seq_len and row["telugu_user"] and row["telugu_assistant"] else rejected
        destination.append({
            "record_id": f"ultrachat_telugu_{key}",
            "source": "HuggingFaceH4/ultrachat_200k translated by NLLB-200",
            "source_index": key,
            "conversations": conversations,
            "qwen_tokens": len(encoded),
        })
    if len(accepted) < args.target_rows:
        raise ValueError(f"Only {len(accepted)} translation rows fit, need {args.target_rows}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in (("train.jsonl", accepted[:args.target_rows]), ("train.maxseq_rejected.jsonl", rejected)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for row in contents:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "input_shards": len(list(Path(args.shards_dir).glob("*.jsonl"))),
        "translated_rows": len(rows),
        "accepted_before_cap": len(accepted),
        "target_rows": args.target_rows,
        "max_seq_len": args.max_seq_len,
        "selection_order": "ascending source_index after deterministic selection",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "written": args.target_rows, "rejected": len(rejected)}), flush=True)


if __name__ == "__main__":
    main()
