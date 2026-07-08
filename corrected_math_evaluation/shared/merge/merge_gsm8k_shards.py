#!/usr/bin/env python
"""Merge deterministic GSM8K evaluation shards and verify complete coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    p for p in HERE.parents if (p / "data").is_dir() and (p / "scripts").is_dir()
)
CME_ROOT = PROJECT_ROOT / "corrected_math_evaluation"
sys.path.insert(0, str(CME_ROOT / "shared/evaluators"))

from evaluate_gsm8k_full import metric  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=1319)
    args = parser.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    identity = [(row["model_path"], row["run_dir"], row["variant"], row["seed"]) for row in reports]
    if len(set(identity)) != 1:
        raise ValueError(f"Shard model identities differ: {identity}")
    results = sorted(
        [result for report in reports for result in report["results"]],
        key=lambda row: row["dataset_index"],
    )
    indices = [row["dataset_index"] for row in results]
    if indices != list(range(args.expected_samples)):
        raise ValueError("Merged shards do not cover each expected dataset index exactly once.")
    first = reports[0]
    merged = {
        "schema_version": 1,
        "dataset": "gsm8k",
        "model_path": first["model_path"],
        "run_dir": first["run_dir"],
        "variant": first["variant"],
        "seed": first["seed"],
        "eval_data": first["eval_data"],
        "batch_size_per_shard": first["batch_size"],
        "max_new_tokens": first["max_new_tokens"],
        "num_shards": len(reports),
        "elapsed_seconds_max_shard": max(row["elapsed_seconds"] for row in reports),
        "full": metric(results),
        "clean": metric(results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in merged.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
