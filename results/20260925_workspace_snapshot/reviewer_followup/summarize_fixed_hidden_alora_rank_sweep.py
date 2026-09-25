#!/usr/bin/env python
"""Summarize the predeclared seed-42 fixed-hidden A-LoRA rank sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RANKS = (2, 4, 8, 16, 32, 50)


def load_report(path: Path, expected_rows: int = 1000) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_example")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise RuntimeError(f"{path}: expected {expected_rows} per-example rows")
    total_nll = sum(float(row["nll_sum"]) for row in rows)
    total_tokens = sum(int(row["token_count"]) for row in rows)
    recomputed = total_nll / total_tokens
    if not math.isclose(
        recomputed, float(payload["avg_ce"]), rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(f"{path}: report arithmetic mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--rank50-dir", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment_dir).resolve()
    rank50 = Path(args.rank50_dir).resolve()

    rows = []
    for rank in RANKS:
        directory = (
            rank50
            if rank == 50
            else experiment
            / "checkpoints"
            / f"qwen25_15b_fhfp32_hsd42_alora_r{rank}_s16_bsd42"
        )
        config = json.loads(
            (directory / "fixed_boundary_config.json").read_text(encoding="utf-8")
        )
        report_paths = {
            "train1000": (
                experiment / "rank50_train1000_report.json"
                if rank == 50
                else directory / "train1000_report.json"
            ),
            "dev": directory / "dev_report.json",
            "test": directory / "test_report.json",
        }
        reports = {
            split: load_report(path) for split, path in report_paths.items()
        }
        rows.append(
            {
                "rank": rank,
                "alpha": float(config["alpha"]),
                "scale": float(config["scale"]),
                "boundary_parameters": int(config["trainable_parameters"]),
                "run_dir": str(directory),
                "train1000_ce": float(reports["train1000"]["avg_ce"]),
                "dev_ce": float(reports["dev"]["avg_ce"]),
                "test_ce": float(reports["test"]["avg_ce"]),
                "dev_minus_train1000": float(reports["dev"]["avg_ce"])
                - float(reports["train1000"]["avg_ce"]),
                "test_minus_train1000": float(reports["test"]["avg_ce"])
                - float(reports["train1000"]["avg_ce"]),
            }
        )

    best_dev = min(rows, key=lambda row: row["dev_ce"])
    best_test = min(rows, key=lambda row: row["test_ce"])
    rank50_row = next(row for row in rows if row["rank"] == 50)
    payload = {
        "marker": "fixed_hidden_alora_rank_sweep_summary_v1",
        "seed": 42,
        "functional_scale": 16.0,
        "ranks": list(RANKS),
        "rows": rows,
        "best_dev_rank": best_dev["rank"],
        "best_test_rank_descriptive": best_test["rank"],
        "rank50_reference": rank50_row,
    }
    (experiment / "rank_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Fixed-hidden A-LoRA rank sweep",
        "",
        "All endpoints use hidden seed 42 and functional scale 16. "
        "`train1000` is the predeclared first 1,000 training records.",
        "",
        "| rank | alpha | boundary params | train1000 CE | dev CE | test CE | dev−train |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['alpha']:.0f} | "
            f"{row['boundary_parameters']:,} | {row['train1000_ce']:.9f} | "
            f"{row['dev_ce']:.9f} | {row['test_ce']:.9f} | "
            f"{row['dev_minus_train1000']:+.9f} |"
        )
    lines.extend(
        [
            "",
            f"- Best dev rank: `{best_dev['rank']}` "
            f"(CE `{best_dev['dev_ce']:.9f}`).",
            f"- Descriptive best test rank: `{best_test['rank']}` "
            f"(CE `{best_test['test_ce']:.9f}`).",
            "",
            "This single-seed diagnostic distinguishes rank behavior within the "
            "fixed-hidden output-only protocol; it is not a cross-seed claim.",
        ]
    )
    (experiment / "rank_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
