#!/usr/bin/env python
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "ANALYSIS.md"
SEEDS = (42, 43, 44)
T_975_DF2 = 4.3026527297


def configs() -> list[tuple[str, int, int | None]]:
    values = [("hidden_hr4", 4, None)]
    values += [(f"lmhead_ar1_s{scale}_hr4", 4, scale) for scale in (1, 2, 4, 8, 16)]
    values += [("hidden_hr8", 8, None), ("lmhead_ar1_s8_hr8", 8, 8)]
    return values


def run_name(label: str, seed: int) -> str:
    return f"{label}_seed{seed}"


def read_score(label: str, seed: int, task: str) -> float | None:
    path = OUTPUTS / task / f"{run_name(label, seed)}_full.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = 5000 if task == "math" else 1319
    rows = report["results"]
    indices = [row["dataset_index"] for row in rows]
    if report["full"]["num_samples"] != expected or indices != list(range(expected)):
        raise ValueError(f"Incomplete or duplicate coverage: {path}")
    metric = report["clean"] if task == "math" else report["full"]
    return float(metric["accuracy_pct"])


def mean_sd(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f}%"
    return f"{statistics.mean(values):.4f}% ± {statistics.stdev(values):.4f}"


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
    score: dict[tuple[str, int, str], float] = {}
    for label, _, _ in configs():
        for seed in SEEDS:
            for task in ("math", "gsm8k"):
                value = read_score(label, seed, task)
                if value is not None:
                    score[label, seed, task] = value

    lines = [
        "# Statistical analysis", "", f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "MATH reports the clean-4,995 score. Intervals below are paired across the three initialization/training seeds; with n=3 they are intentionally conservative.", "",
        "## Aggregate accuracy", "", "| config | completed | MATH mean ± sd | GSM8K mean ± sd |", "| --- | ---: | ---: | ---: |",
    ]
    for label, _, _ in configs():
        ms = [score[label, seed, "math"] for seed in SEEDS if (label, seed, "math") in score]
        gs = [score[label, seed, "gsm8k"] for seed in SEEDS if (label, seed, "gsm8k") in score]
        lines.append(f"| {label} | {min(len(ms), len(gs))}/3 | {mean_sd(ms)} | {mean_sd(gs)} |")

    lines += [
        "", "## Paired improvement over matching hidden-LoRA baseline", "",
        "| treatment | paired seeds | Δ MATH | MATH positive seeds | Δ GSM8K | GSM8K positive seeds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, hidden_rank, scale in configs():
        if scale is None:
            continue
        baseline = f"hidden_hr{hidden_rank}"
        md = [score[label, seed, "math"] - score[baseline, seed, "math"] for seed in SEEDS if (label, seed, "math") in score and (baseline, seed, "math") in score]
        gd = [score[label, seed, "gsm8k"] - score[baseline, seed, "gsm8k"] for seed in SEEDS if (label, seed, "gsm8k") in score and (baseline, seed, "gsm8k") in score]
        lines.append(f"| {label} | {min(len(md), len(gd))}/3 | {paired_summary(md)} | {sum(x > 0 for x in md)}/{len(md) if md else 0} | {paired_summary(gd)} | {sum(x > 0 for x in gd)}/{len(gd) if gd else 0} |")

    hr4_complete: list[tuple[int, float, float]] = []
    for scale in (1, 2, 4, 8, 16):
        label = f"lmhead_ar1_s{scale}_hr4"
        ms = [score[label, seed, "math"] for seed in SEEDS if (label, seed, "math") in score]
        gs = [score[label, seed, "gsm8k"] for seed in SEEDS if (label, seed, "gsm8k") in score]
        if len(ms) == len(gs) == 3:
            hr4_complete.append((scale, statistics.mean(ms), statistics.mean(gs)))
    lines += ["", "## Scale sweep ranking", ""]
    if len(hr4_complete) == 5:
        lines += ["| rank by MATH | scale | MATH mean | GSM8K mean |", "| ---: | ---: | ---: | ---: |"]
        for rank, (scale, m, g) in enumerate(sorted(hr4_complete, key=lambda x: x[1], reverse=True), 1):
            lines.append(f"| {rank} | {scale} | {m:.4f}% | {g:.4f}% |")
    else:
        lines.append(f"Scale ranking pending: {len(hr4_complete)}/5 configurations have all three seeds and both evaluations.")

    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
