#!/usr/bin/env python
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "HIDDEN_RANK_ANALYSIS.md"
SEEDS = (42, 43, 44)
HIDDEN_RANKS = (1, 2, 4, 8, 16)


def read_score(rank: int, seed: int, task: str) -> float | None:
    path = OUTPUTS / task / f"qwen25_05b_hidden_hr{rank}_seed{seed}_full.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = 5000 if task == "math" else 1319
    rows = report["results"]
    if report["full"]["num_samples"] != expected or [row["dataset_index"] for row in rows] != list(range(expected)):
        raise ValueError(f"Incomplete or duplicate coverage: {path}")
    metric = report["clean"] if task == "math" else report["full"]
    return float(metric["accuracy_pct"])


def fmt(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f}%"
    return f"{statistics.mean(values):.4f}% ± {statistics.stdev(values):.4f}"


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def main() -> None:
    scores: dict[tuple[int, int, str], float] = {}
    for rank in HIDDEN_RANKS:
        for seed in SEEDS:
            for task in ("math", "gsm8k"):
                value = read_score(rank, seed, task)
                if value is not None:
                    scores[rank, seed, task] = value

    rows = []
    for rank in HIDDEN_RANKS:
        ms = [scores[rank, seed, "math"] for seed in SEEDS if (rank, seed, "math") in scores]
        gs = [scores[rank, seed, "gsm8k"] for seed in SEEDS if (rank, seed, "gsm8k") in scores]
        rows.append((rank, ms, gs))

    complete = [row for row in rows if len(row[1]) == 3 and len(row[2]) == 3]
    best_math = max(complete, key=lambda row: mean(row[1]), default=None)
    best_gsm = max(complete, key=lambda row: mean(row[2]), default=None)
    best_avg = max(complete, key=lambda row: (mean(row[1]) + mean(row[2])) / 2, default=None)

    lines = [
        "# Qwen2.5-0.5B hidden LoRA rank analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "This is the first stage before checking mergeable AffLoRA on top of hidden LoRA.",
        "MATH reports the clean-4,995 score.",
        "",
        "## Aggregate accuracy",
        "",
        "| hidden rank | completed | MATH mean ± sd | GSM8K mean ± sd |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for rank, ms, gs in rows:
        lines.append(f"| {rank} | {min(len(ms), len(gs))}/3 | {fmt(ms)} | {fmt(gs)} |")

    lines += ["", "## Ranking", ""]
    if complete:
        lines += ["| rank by MATH | hidden rank | MATH mean | GSM8K mean | mean(MATH,GSM8K) |", "| ---: | ---: | ---: | ---: | ---: |"]
        for idx, (rank, ms, gs) in enumerate(sorted(complete, key=lambda row: mean(row[1]), reverse=True), 1):
            lines.append(f"| {idx} | {rank} | {mean(ms):.4f}% | {mean(gs):.4f}% | {(mean(ms)+mean(gs))/2:.4f}% |")
    else:
        lines.append("Pending.")

    lines += ["", "## Suggested second-stage hidden ranks", ""]
    if best_math and best_gsm and best_avg:
        chosen = []
        for row in (best_math, best_gsm, best_avg):
            if row[0] not in chosen:
                chosen.append(row[0])
        lines.append("Use these hidden ranks for the mergeable AffLoRA follow-up unless manual inspection overrides:")
        lines.append("")
        for rank in chosen:
            lines.append(f"- hidden hr{rank}")
        lines.append("")
        lines.append("Rationale:")
        lines.append("")
        lines.append(f"- best MATH: hr{best_math[0]}")
        lines.append(f"- best GSM8K: hr{best_gsm[0]}")
        lines.append(f"- best average of MATH and GSM8K: hr{best_avg[0]}")
    else:
        lines.append("Pending completion.")

    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
