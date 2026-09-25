#!/usr/bin/env python
"""Validate and merge contiguous IFEval generation shards in dataset order."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SHARD_RE = re.compile(r".+_sd(?P<seed>\d+)_(?P<start>\d+)_(?P<end>\d+)\.jsonl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--name-prefix", default="", help="Optional filename prefix selecting one configuration."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", default=541, type=int)
    return parser.parse_args()


def metadata(path: Path) -> tuple[int, int]:
    match = SHARD_RE.fullmatch(path.name)
    if not match or int(match["seed"]) < 0:
        raise ValueError(f"Unrecognized shard name: {path.name}")
    return int(match["start"]), int(match["end"])


def main() -> None:
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    shards = []
    for path in shard_dir.glob(f"{args.name_prefix}*.jsonl"):
        start, end = metadata(path)
        match = SHARD_RE.fullmatch(path.name)
        assert match is not None
        if int(match["seed"]) != args.seed:
            continue
        if end <= start:
            raise ValueError(f"Invalid interval [{start}, {end}) in {path}")
        shards.append((start, end, path))
    shards.sort()
    if not shards:
        raise ValueError(f"No shards for seed {args.seed} in {shard_dir}")

    cursor = 0
    rows: list[dict] = []
    keys: set[object] = set()
    for start, end, path in shards:
        if start != cursor:
            raise ValueError(f"Expected shard beginning at {cursor}, got [{start}, {end}) in {path}")
        # JSONL is delimited by the byte/newline character ``\n``.  Do not use
        # ``str.splitlines()`` here: model text may legitimately contain a
        # Unicode line-separator (for example U+2028), which is valid inside a
        # JSON string but ``splitlines()`` would incorrectly turn into a record
        # boundary.
        part = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line
        ]
        if len(part) != end - start:
            raise ValueError(f"{path} has {len(part)} rows but interval is [{start}, {end})")
        for row in part:
            if row["key"] in keys:
                raise ValueError(f"Duplicate IFEval key {row['key']} in {path}")
            keys.add(row["key"])
        rows.extend(part)
        cursor = end
    if cursor != args.expected_count or len(rows) != args.expected_count:
        raise ValueError(f"Merged {len(rows)} rows over [0, {cursor}); expected {args.expected_count}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged {len(rows)} unique rows for seed {args.seed} into {output}")


if __name__ == "__main__":
    main()
