#!/usr/bin/env python
"""Build deterministic, leakage-audited corrected SFT splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data_pipeline import (
    ConversationFormatError,
    assistant_turn_count,
    canonical_text,
    first_user_key,
    messages_from_row,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SOURCES = [
    ROOT / "data/sft_t2t_mini_25k/train.jsonl",
    ROOT / "data/sft_t2t_mini_25k/eval.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=[str(path) for path in DEFAULT_SOURCES])
    parser.add_argument("--out-dir", default=str(HERE / "data"))
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_record_id(source: Path, source_index: int, row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{source}:{source_index}:{payload}".encode()).hexdigest()[:24]


def load_and_validate(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for source_index, line in enumerate(stream):
                row = json.loads(line)
                try:
                    messages = messages_from_row(row)
                except ConversationFormatError as exc:
                    reason = str(exc)
                    reasons[reason] += 1
                    rejected.append(
                        {
                            "source_file": str(path),
                            "source_index": source_index,
                            "reason": reason,
                            "row": row,
                        }
                    )
                    continue
                accepted.append(
                    {
                        "record_id": stable_record_id(path, source_index, row),
                        "source_file": str(path),
                        "source_index": source_index,
                        "conversations": messages,
                    }
                )
    return accepted, rejected, reasons


def proportional_targets(rows: list[dict[str, Any]], size: int) -> dict[int, int]:
    histogram = Counter(assistant_turn_count(row) for row in rows)
    total = sum(histogram.values())
    raw = {turns: count * size / total for turns, count in histogram.items()}
    targets = {turns: math.floor(value) for turns, value in raw.items()}
    remaining = size - sum(targets.values())
    order = sorted(raw, key=lambda turns: (raw[turns] - targets[turns], -turns), reverse=True)
    for turns in order[:remaining]:
        targets[turns] += 1
    return targets


def build_splits(
    rows: list[dict[str, Any]], dev_size: int, test_size: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[first_user_key(row)].append(row)

    singleton_by_turns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups.values():
        if len(group) == 1:
            row = group[0]
            singleton_by_turns[assistant_turn_count(row)].append(row)

    dev_targets = proportional_targets(rows, dev_size)
    test_targets = proportional_targets(rows, test_size)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for turns in sorted(set(dev_targets) | set(test_targets)):
        candidates = singleton_by_turns[turns]
        random.Random(seed * 1000 + turns).shuffle(candidates)
        need_dev = dev_targets.get(turns, 0)
        need_test = test_targets.get(turns, 0)
        if len(candidates) < need_dev + need_test:
            raise RuntimeError(
                f"Not enough singleton prompt groups for {turns} turns: "
                f"available={len(candidates)}, required={need_dev + need_test}."
            )
        dev.extend(candidates[:need_dev])
        test.extend(candidates[need_dev : need_dev + need_test])
        selected_ids.update(row["record_id"] for row in candidates[: need_dev + need_test])

    train = [row for row in rows if row["record_id"] not in selected_ids]
    for offset, split in enumerate((train, dev, test)):
        random.Random(seed + 10_000 + offset).shuffle(split)
    return {"train": train, "dev": dev, "test": test}


def conversation_key(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], canonical_text(message["content"]))
        for message in messages_from_row(row)
    )


def assistant_contexts(rows: list[dict[str, Any]]) -> set[tuple[tuple[str, str], ...]]:
    contexts: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        conversation = conversation_key(row)
        for index, (role, _) in enumerate(conversation):
            if role == "assistant":
                contexts.add(conversation[:index])
    return contexts


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = Counter(first_user_key(row) for row in rows)
    sources = Counter(Path(row["source_file"]).name for row in rows)
    return {
        "rows": len(rows),
        "unique_first_user_prompts": len(prompts),
        "max_first_user_group_size": max(prompts.values(), default=0),
        "assistant_turn_histogram": dict(sorted(Counter(assistant_turn_count(row) for row in rows).items())),
        "source_file_counts": dict(sorted(sources.items())),
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    sources = [Path(value).resolve() for value in args.sources]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted, rejected, reasons = load_and_validate(sources)
    splits = build_splits(accepted, args.dev_size, args.test_size, args.seed)

    for name, rows in splits.items():
        write_jsonl(rows, out_dir / f"{name}.jsonl")
    write_jsonl(rejected, out_dir / "rejected.jsonl")

    prompt_sets = {
        name: {first_user_key(row) for row in rows} for name, rows in splits.items()
    }
    context_sets = {name: assistant_contexts(rows) for name, rows in splits.items()}
    overlap: dict[str, Any] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap[f"{left}_vs_{right}"] = {
            "first_user_prompt": len(prompt_sets[left] & prompt_sets[right]),
            "assistant_context": len(context_sets[left] & context_sets[right]),
            "full_conversation": len(
                {conversation_key(row) for row in splits[left]}
                & {conversation_key(row) for row in splits[right]}
            ),
        }

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "split_policy": "singleton normalized first-user prompt holdout, stratified by assistant-turn count",
        "source_files": [
            {"path": str(path), "rows_sha256": sha256_file(path)} for path in sources
        ],
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(reasons.items())),
        "splits": {name: split_summary(rows) for name, rows in splits.items()},
        "cross_split_overlap": overlap,
        "generated_files": {
            name: {
                "path": str(out_dir / f"{name}.jsonl"),
                "sha256": sha256_file(out_dir / f"{name}.jsonl"),
            }
            for name in (*splits.keys(), "rejected")
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

