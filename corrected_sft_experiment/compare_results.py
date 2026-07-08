#!/usr/bin/env python
"""Paired bootstrap comparison of two per-example NLL reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    treatment = json.loads(Path(args.treatment).read_text(encoding="utf-8"))
    base_rows = {row["record_id"]: row for row in baseline["per_example"]}
    treat_rows = {row["record_id"]: row for row in treatment["per_example"]}
    if base_rows.keys() != treat_rows.keys():
        raise ValueError("Reports do not contain the same record IDs.")
    ids = sorted(base_rows)
    base_nll = np.array([base_rows[key]["nll_sum"] for key in ids], dtype=np.float64)
    treat_nll = np.array([treat_rows[key]["nll_sum"] for key in ids], dtype=np.float64)
    counts = np.array([base_rows[key]["token_count"] for key in ids], dtype=np.float64)
    if not np.array_equal(
        counts, np.array([treat_rows[key]["token_count"] for key in ids], dtype=np.float64)
    ):
        raise ValueError("Reports disagree on supervised token counts.")

    observed = (treat_nll.sum() - base_nll.sum()) / counts.sum()
    rng = np.random.default_rng(args.seed)
    deltas = np.empty(args.samples, dtype=np.float64)
    for index in range(args.samples):
        sample = rng.integers(0, len(ids), size=len(ids))
        deltas[index] = (
            treat_nll[sample].sum() - base_nll[sample].sum()
        ) / counts[sample].sum()
    report = {
        "baseline": str(Path(args.baseline).resolve()),
        "treatment": str(Path(args.treatment).resolve()),
        "examples": len(ids),
        "delta_ce_treatment_minus_baseline": observed,
        "bootstrap_samples": args.samples,
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_treatment_better": float(np.mean(deltas < 0)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

