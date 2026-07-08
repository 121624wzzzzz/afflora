#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "MERGEABLE_RANK_ANALYSIS.md"
MODELS = ("qwen3_06b", "qwen3_17b")
MODEL_HIDDEN_RANK = {"qwen3_06b": 8, "qwen3_17b": 4}
MODEL_AFFINE_SCALE = {"qwen3_06b": 8, "qwen3_17b": 1}
AFFINE_RANKS = (1, 2, 4, 8, 16)
SEED = 42


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


def main() -> None:
    lines = [
        "# Qwen3 close-size hidden + mergeable AffLoRA rank analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "This is a seed42 main-trend sweep. Deltas are paired against the matching hidden-only baseline.",
        "",
    ]
    for model in MODELS:
        hr = MODEL_HIDDEN_RANK[model]
        scale = MODEL_AFFINE_SCALE[model]
        baseline = f"{model}_hidden_hr{hr}_seed{SEED}"
        bm = read_score(baseline, "math")
        bg = read_score(baseline, "gsm8k")
        rows = []
        for rank in AFFINE_RANKS:
            name = f"{model}_mergeable_ar{rank}_s{scale}_hr{hr}_seed{SEED}"
            m = read_score(name, "math")
            g = read_score(name, "gsm8k")
            rows.append((rank, m, None if m is None or bm is None else m - bm, g, None if g is None or bg is None else g - bg))
        complete = [row for row in rows if row[1] is not None and row[3] is not None]
        lines += [
            f"## {model}",
            "",
            f"Baseline hidden hr{hr}: MATH {bm:.4f}% / GSM8K {bg:.4f}%" if bm is not None and bg is not None else f"Baseline hidden hr{hr}: pending",
            "",
            "| affine rank | MATH | Δ MATH | GSM8K | Δ GSM8K |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for rank, m, dm, g, dg in rows:
            if None in (m, dm, g, dg):
                lines.append(f"| {rank} | - | - | - | - |")
            else:
                lines.append(f"| {rank} | {m:.4f}% | {dm:+.4f} pp | {g:.4f}% | {dg:+.4f} pp |")
        lines += ["", "### Ranking by Δ MATH", ""]
        if complete:
            lines += ["| rank by Δ MATH | affine rank | Δ MATH | Δ GSM8K |", "| ---: | ---: | ---: | ---: |"]
            for idx, (rank, _, dm, _, dg) in enumerate(sorted(complete, key=lambda row: row[2], reverse=True), 1):
                lines.append(f"| {idx} | {rank} | {dm:+.4f} pp | {dg:+.4f} pp |")
            lines.append("")
        else:
            lines += ["Pending.", ""]
    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
