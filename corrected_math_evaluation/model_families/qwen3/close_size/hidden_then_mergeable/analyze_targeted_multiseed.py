#!/usr/bin/env python
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "TARGETED_MULTISEED_ANALYSIS.md"
SEEDS = (42, 43, 44)
CONFIGS = [
    ("qwen3_06b", 8, 8, (1, 2)),
    ("qwen3_17b", 4, 1, (2, 8)),
]
T_975_DF2 = 4.3026527297


def read_score(name: str, task: str) -> float | None:
    path = OUTPUTS / task / f"{name}_full.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = 5000 if task == "math" else 1319
    rows = report["results"]
    if report["full"]["num_samples"] != expected or [row["dataset_index"] for row in rows] != list(range(expected)):
        raise ValueError(f"Incomplete or duplicate coverage: {path}")
    return float(report["clean" if task == "math" else "full"]["accuracy_pct"])


def paired_summary(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) < 3:
        return f"{mean:+.4f} pp (n={len(values)})"
    sd = statistics.stdev(values)
    half = T_975_DF2 * sd / math.sqrt(len(values))
    return f"{mean:+.4f} ± {sd:.4f} pp; 95% CI [{mean-half:+.4f}, {mean+half:+.4f}]"


def main() -> None:
    lines = [
        "# Qwen3 targeted hidden+mergeable multi-seed analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Deltas are paired against the matching hidden-only baseline for the same model, hidden rank, and seed.",
        "",
    ]
    for model, hr, scale, ranks in CONFIGS:
        lines += [f"## {model} hidden hr{hr}", ""]
        for rank in ranks:
            rows = []
            for seed in SEEDS:
                base = f"{model}_hidden_hr{hr}_seed{seed}"
                run = f"{model}_mergeable_ar{rank}_s{scale}_hr{hr}_seed{seed}"
                bm, bg = read_score(base, "math"), read_score(base, "gsm8k")
                m, g = read_score(run, "math"), read_score(run, "gsm8k")
                if None not in (bm, bg, m, g):
                    rows.append((seed, bm, m, m - bm, bg, g, g - bg))
            md = [row[3] for row in rows]
            gd = [row[6] for row in rows]
            lines += [
                f"### mergeable ar{rank}_s{scale}",
                "",
                "| seed | hidden MATH | mergeable MATH | Δ MATH | hidden GSM8K | mergeable GSM8K | Δ GSM8K |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for seed, bm, m, dm, bg, g, dg in rows:
                lines.append(f"| {seed} | {bm:.4f}% | {m:.4f}% | {dm:+.4f} pp | {bg:.4f}% | {g:.4f}% | {dg:+.4f} pp |")
            lines += [
                "",
                f"- completed paired seeds: {len(rows)}/3",
                f"- Δ MATH: {paired_summary(md)}; positive seeds {sum(x > 0 for x in md)}/{len(md)}",
                f"- Δ GSM8K: {paired_summary(gd)}; positive seeds {sum(x > 0 for x in gd)}/{len(gd)}",
                "",
            ]
    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
