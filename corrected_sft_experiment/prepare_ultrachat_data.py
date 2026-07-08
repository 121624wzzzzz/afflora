#!/usr/bin/env python
"""Filter UltraChat 100K for the corrected chat-template SFT pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_pipeline import ConversationFormatError, assistant_turn_count, messages_from_row


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_IN_DIR = ROOT / "data/ultrachat_100k"
DEFAULT_OUT_DIR = HERE / "data_ultrachat_100k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_record_id(split: str, source_index: int, row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"ultrachat_100k:{split}:{source_index}:{payload}".encode()).hexdigest()[:24]


def process_split(in_path: Path, out_path: Path, rejected_path: Path, split: str) -> dict[str, Any]:
    accepted = 0
    rejected = 0
    reasons: Counter[str] = Counter()
    turn_histogram: Counter[int] = Counter()
    with in_path.open(encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as out, rejected_path.open(
        "a", encoding="utf-8"
    ) as rej:
        for source_index, line in enumerate(source):
            row = json.loads(line)
            try:
                messages_from_row(row)
            except ConversationFormatError as exc:
                rejected += 1
                reasons[str(exc)] += 1
                rej.write(
                    json.dumps(
                        {
                            "split": split,
                            "source_file": str(in_path),
                            "source_index": source_index,
                            "reason": str(exc),
                            "row": row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            clean = {
                "record_id": stable_record_id(split, source_index, row),
                "source_file": str(in_path),
                "source_index": source_index,
                "prompt_id": row.get("prompt_id"),
                "prompt": row.get("prompt"),
                "conversations": row["conversations"],
            }
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")
            accepted += 1
            turn_histogram[assistant_turn_count(clean)] += 1
    return {
        "source": str(in_path),
        "source_sha256": sha256_file(in_path),
        "output": str(out_path),
        "accepted": accepted,
        "rejected": rejected,
        "rejection_reasons": dict(sorted(reasons.items())),
        "assistant_turn_histogram": dict(sorted(turn_histogram.items())),
    }


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = out_dir / "rejected.jsonl"
    rejected_path.write_text("", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "source_dataset": "ultrachat_100k",
        "filter_policy": "strict user/assistant alternating conversations; preserve existing train/eval split",
        "splits": {},
    }
    for split in ("train", "eval"):
        manifest["splits"][split] = process_split(
            in_dir / f"{split}.jsonl",
            out_dir / f"{split}.jsonl",
            rejected_path,
            split,
        )
        manifest["splits"][split]["output_sha256"] = sha256_file(out_dir / f"{split}.jsonl")
    manifest["rejected"] = {
        "path": str(rejected_path),
        "sha256": sha256_file(rejected_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
