#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "HR8_MERGEABLE_ANALYSIS.md"
SEEDS = (42,)
HIDDEN_RANK = 8
AFFINE_RANKS = (1, 2, 4, 8, 16)
AFFINE_SCALE = 8


def read_score(name: str, task: str) -> float | None:
    path = OUTPUTS / task / f"{name}_full.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    metric = report["clean"] if task == "math" else report["full"]
    return float(metric["accuracy_pct"])


def main() -> None:
    lines = [
        "# Qwen2.5-0.5B hidden hr8 + mergeable AffLoRA rank analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "This is a main-trend sweep before adding more seeds.",
        "Baseline is the matching hidden-only `hr8` result from the hidden-rank stage.",
        "",
        "| affine rank | seed | MATH | Δ MATH vs hr8 hidden | GSM8K | Δ GSM8K vs hr8 hidden |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    complete_rows = []
    for seed in SEEDS:
        base = f"qwen25_05b_hidden_hr{HIDDEN_RANK}_seed{seed}"
        bm = read_score(base, "math")
        bg = read_score(base, "gsm8k")
        for rank in AFFINE_RANKS:
            name = f"qwen25_05b_mergeable_ar{rank}_s{AFFINE_SCALE}_hr{HIDDEN_RANK}_seed{seed}"
            m = read_score(name, "math")
            g = read_score(name, "gsm8k")
            if None not in (m, g, bm, bg):
                dm = m - bm
                dg = g - bg
                complete_rows.append((rank, seed, m, dm, g, dg))
                lines.append(f"| {rank} | {seed} | {m:.4f}% | {dm:+.4f} pp | {g:.4f}% | {dg:+.4f} pp |")
            else:
                lines.append(f"| {rank} | {seed} | - | - | - | - |")
    lines += ["", "## Ranking by MATH delta", ""]
    if complete_rows:
        lines += ["| rank | seed | Δ MATH | Δ GSM8K |", "| ---: | ---: | ---: | ---: |"]
        for rank, seed, _, dm, _, dg in sorted(complete_rows, key=lambda row: row[3], reverse=True):
            lines.append(f"| {rank} | {seed} | {dm:+.4f} pp | {dg:+.4f} pp |")
    else:
        lines.append("Pending.")
    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
