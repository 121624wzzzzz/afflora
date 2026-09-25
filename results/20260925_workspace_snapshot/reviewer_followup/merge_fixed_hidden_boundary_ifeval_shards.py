#!/usr/bin/env python
"""Merge the eight exact fixed-hidden IFEval shards in canonical order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_fixed_hidden_boundary_ifeval as validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def merge(
    *,
    shard_dir: Path,
    run_dir: Path,
    protocol_manifest: Path,
    output: Path,
) -> list[dict]:
    run_dir = run_dir.resolve()
    identity = validation.validate_checkpoint(run_dir)
    validation.validate_protocol_manifest(protocol_manifest, run_dir=run_dir)
    run_name = identity["run_name"]
    expected_names = {
        f"{run_name}_{start}_{end}.jsonl"
        for start, end in zip(
            validation.SHARD_BOUNDS[:-1],
            validation.SHARD_BOUNDS[1:],
            strict=True,
        )
    }
    if not shard_dir.is_dir():
        raise ValueError(f"Shard directory is missing: {shard_dir}")
    actual_names = {
        path.name for path in shard_dir.glob(f"{run_name}_*.jsonl")
    }
    if actual_names != expected_names:
        raise ValueError(
            f"{run_name}: exact shard set mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    canonical = validation.canonical_rows()
    rows: list[dict] = []
    seen_keys: set[object] = set()
    for start, end in zip(
        validation.SHARD_BOUNDS[:-1],
        validation.SHARD_BOUNDS[1:],
        strict=True,
    ):
        path = shard_dir / f"{run_name}_{start}_{end}.jsonl"
        part = validation.validate_responses(
            path=path,
            run_dir=run_dir,
            protocol_manifest=protocol_manifest,
            start_index=start,
            end_index=end,
            expected=canonical,
        )
        for row in part:
            if row["key"] in seen_keys:
                raise ValueError(f"{path}: duplicate key across shards")
            seen_keys.add(row["key"])
        rows.extend(part)
    if len(rows) != validation.EXPECTED_COUNT:
        raise ValueError(
            f"{run_name}: merged {len(rows)} rows, "
            f"expected {validation.EXPECTED_COUNT}"
        )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return rows


def main() -> None:
    args = parse_args()
    rows = merge(
        shard_dir=args.shard_dir.resolve(),
        run_dir=args.run_dir.resolve(),
        protocol_manifest=args.protocol_manifest.resolve(),
        output=args.output.resolve(),
    )
    print(
        f"merged {len(rows)} canonical rows for {args.run_dir.resolve().name} "
        f"into {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
