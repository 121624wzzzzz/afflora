#!/usr/bin/env python
"""Post-hoc paired-item analysis for the completed rank sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RANKS = (2, 4, 8, 16, 32, 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--rank50-dir", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment_dir).resolve()
    rank50 = Path(args.rank50_dir).resolve()
    summary = json.loads((experiment / "rank_summary.json").read_text())
    summary_rows = {int(row["rank"]): row for row in summary["rows"]}

    reports = {}
    for rank in RANKS:
        directory = (
            rank50
            if rank == 50
            else experiment
            / "checkpoints"
            / f"qwen25_15b_fhfp32_hsd42_alora_r{rank}_s16_bsd42"
        )
        reports[rank] = json.loads((directory / "test_report.json").read_text())

    reference = reports[50]["per_example"]
    reference_ids = [row.get("record_id") for row in reference]
    counts = np.asarray(
        [int(row["token_count"]) for row in reference], dtype=np.float64
    )
    reference_nll = np.asarray(
        [float(row["nll_sum"]) for row in reference], dtype=np.float64
    )
    comparisons = []
    for rank in RANKS:
        candidate = reports[rank]["per_example"]
        if [row.get("record_id") for row in candidate] != reference_ids:
            raise RuntimeError(f"rank {rank}: test record order differs from r50")
        candidate_counts = np.asarray(
            [int(row["token_count"]) for row in candidate], dtype=np.float64
        )
        if not np.array_equal(candidate_counts, counts):
            raise RuntimeError(f"rank {rank}: token counts differ from r50")
        candidate_nll = np.asarray(
            [float(row["nll_sum"]) for row in candidate], dtype=np.float64
        )
        difference = candidate_nll - reference_nll
        rng = np.random.default_rng(2026072900 + rank)
        chunks = []
        for _ in range(10):
            indices = rng.integers(0, len(reference), size=(1000, len(reference)))
            chunks.append(
                difference[indices].sum(axis=1) / counts[indices].sum(axis=1)
            )
        bootstrap = np.concatenate(chunks)
        comparisons.append(
            {
                "rank": rank,
                "test_ce": float(summary_rows[rank]["test_ce"]),
                "test_ce_minus_r50": float(summary_rows[rank]["test_ce"])
                - float(summary_rows[50]["test_ce"]),
                "paired_item_bootstrap_samples": 10000,
                "paired_item_bootstrap_ci95": [
                    float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5)),
                ],
            }
        )

    train_gain = (
        float(summary_rows[2]["train1000_ce"])
        - float(summary_rows[50]["train1000_ce"])
    )
    dev_gain = float(summary_rows[2]["dev_ce"]) - float(summary_rows[50]["dev_ce"])
    test_gain = (
        float(summary_rows[2]["test_ce"]) - float(summary_rows[50]["test_ce"])
    )
    payload = {
        "marker": "fixed_hidden_alora_rank_sweep_paired_posthoc_v1",
        "comparison": "rank CE minus r50 CE; positive favors r50",
        "comparisons": comparisons,
        "r2_to_r50_ce_reduction": {
            "train1000": train_gain,
            "dev": dev_gain,
            "test": test_gain,
            "dev_fraction_of_train1000_gain": dev_gain / train_gain,
            "test_fraction_of_train1000_gain": test_gain / train_gain,
        },
        "diagnosis": (
            "Train1000, dev, and test CE improve monotonically through r50. "
            "The growing train-to-held-out gap shows a partial overfitting "
            "component, but no held-out reversal or lower-rank sweet spot "
            "appears within ranks 2..50."
        ),
    }
    (experiment / "paired_rank_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        "# Paired test analysis against r50",
        "",
        "Positive deltas favor r50. Bootstrap unit: paired corrected-SFT test example.",
        "",
        "| rank | test CE | rank−r50 CE | paired bootstrap 95% CI |",
        "|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        low, high = row["paired_item_bootstrap_ci95"]
        lines.append(
            f"| {row['rank']} | {row['test_ce']:.9f} | "
            f"{row['test_ce_minus_r50']:+.9f} | [{low:+.9f}, {high:+.9f}] |"
        )
    lines.extend(
        [
            "",
            f"- r2→r50 train1000 CE reduction: `{train_gain:.9f}`.",
            f"- r2→r50 dev CE reduction: `{dev_gain:.9f}` "
            f"({dev_gain / train_gain:.1%} of train1000 gain).",
            f"- r2→r50 test CE reduction: `{test_gain:.9f}` "
            f"({test_gain / train_gain:.1%} of train1000 gain).",
            "- Diagnosis: all three curves improve monotonically through r50. "
            "The generalization gap widens, but there is no held-out reversal "
            "or lower-rank sweet spot within r2–r50.",
        ]
    )
    (experiment / "paired_rank_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
