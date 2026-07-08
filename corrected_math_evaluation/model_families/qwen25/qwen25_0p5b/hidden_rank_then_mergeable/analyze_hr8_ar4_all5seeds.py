#!/usr/bin/env python
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "HR8_AR4_ALL5SEEDS_ANALYSIS.md"
SEEDS = (42, 43, 44, 123, 2026)
HIDDEN_RANK = 8
AFFINE_RANK = 4
AFFINE_SCALE = 8


T_CRITICAL = {
    2: 4.3026527297,
    4: 2.7764451052,
}


def read_score(name: str, task: str) -> float | None:
    path = OUTPUTS / task / f"{name}_full.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = 5000 if task == "math" else 1319
    rows = report["results"]
    if report["full"]["num_samples"] != expected or [row["dataset_index"] for row in rows] != list(range(expected)):
        raise ValueError(f"Incomplete or duplicate coverage: {path}")
    metric = report["clean"] if task == "math" else report["full"]
    return float(metric["accuracy_pct"])


def paired_summary(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) < 3:
        return f"{mean:+.4f} pp (n={len(values)})"
    sd = statistics.stdev(values)
    t = T_CRITICAL.get(len(values) - 1, 1.96)
    half = t * sd / math.sqrt(len(values))
    return f"{mean:+.4f} ± {sd:.4f} pp; 95% CI [{mean-half:+.4f}, {mean+half:+.4f}]"


def main() -> None:
    rows = []
    for seed in SEEDS:
        base = f"qwen25_05b_hidden_hr{HIDDEN_RANK}_seed{seed}"
        run = f"qwen25_05b_mergeable_ar{AFFINE_RANK}_s{AFFINE_SCALE}_hr{HIDDEN_RANK}_seed{seed}"
        bm, bg = read_score(base, "math"), read_score(base, "gsm8k")
        m, g = read_score(run, "math"), read_score(run, "gsm8k")
        if None not in (bm, bg, m, g):
            rows.append((seed, bm, m, m - bm, bg, g, g - bg))

    md = [row[3] for row in rows]
    gd = [row[6] for row in rows]
    lines = [
        "# Qwen2.5-0.5B hidden hr8 + mergeable ar4 all-seed analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "| seed | hidden hr8 MATH | hr8+ar4 MATH | Δ MATH | hidden hr8 GSM8K | hr8+ar4 GSM8K | Δ GSM8K |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, bm, m, dm, bg, g, dg in rows:
        lines.append(f"| {seed} | {bm:.4f}% | {m:.4f}% | {dm:+.4f} pp | {bg:.4f}% | {g:.4f}% | {dg:+.4f} pp |")
    lines += [
        "",
        "## Paired summary",
        "",
        f"- completed paired seeds: {len(rows)}/5",
        f"- Δ MATH: {paired_summary(md)}; positive seeds {sum(x > 0 for x in md)}/{len(md)}",
        f"- Δ GSM8K: {paired_summary(gd)}; positive seeds {sum(x > 0 for x in gd)}/{len(gd)}",
    ]
    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
