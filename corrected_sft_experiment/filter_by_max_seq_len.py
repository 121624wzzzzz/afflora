#!/usr/bin/env python
"""Drop rows that would have no supervised assistant token after truncation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_pipeline import messages_from_row, tokenize_conversation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_split(split: str, input_dir: Path, output_dir: Path, tokenizer: Any, max_seq_len: int) -> dict[str, Any]:
    input_path = input_dir / f"{split}.jsonl"
    output_path = output_dir / f"{split}.jsonl"
    rejected_path = output_dir / f"{split}.maxseq_rejected.jsonl"
    kept = 0
    rejected = 0
    truncated = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as out, rejected_path.open(
        "w", encoding="utf-8"
    ) as rej:
        for line in source:
            row = json.loads(line)
            full_ids = tokenizer.apply_chat_template(
                messages_from_row(row),
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
                enable_thinking=False,
            )
            try:
                item = tokenize_conversation(row, tokenizer, max_seq_len)
            except RuntimeError as exc:
                rejected += 1
                rej.write(
                    json.dumps(
                        {
                            "record_id": row.get("record_id"),
                            "reason": str(exc),
                            "source_file": row.get("source_file"),
                            "source_index": row.get("source_index"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            kept += 1
            truncated += len(full_ids) > max_seq_len
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "maxseq_rejected": str(rejected_path),
        "maxseq_rejected_sha256": sha256_file(rejected_path),
        "kept": kept,
        "rejected_zero_supervised_after_truncation": rejected,
        "truncated_kept_rows": truncated,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    manifest = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "max_seq_len": args.max_seq_len,
        "filter_policy": "drop rows with zero supervised assistant tokens after truncation",
        "splits": {},
    }
    for split in args.splits:
        manifest["splits"][split] = process_split(split, input_dir, output_dir, tokenizer, args.max_seq_len)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
