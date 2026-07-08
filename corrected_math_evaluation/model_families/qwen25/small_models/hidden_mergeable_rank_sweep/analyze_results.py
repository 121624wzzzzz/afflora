#!/usr/bin/env python
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "ANALYSIS.md"
SEEDS = (42, 43, 44)
MODELS = ("qwen25_05b", "qwen25_15b")
AFFINE_RANKS = (1, 2, 4, 8, 16)
MODEL_SCALE = {
    "qwen25_05b": 8.0,
    "qwen25_15b": 1.0,
}
T_975_DF2 = 4.3026527297


def scale_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def configs() -> list[tuple[str, str, int | None, float | None]]:
    values: list[tuple[str, str, int | None, float | None]] = []
    for model in MODELS:
        values.append((f"{model}_hidden_hr4", model, None, None))
        scale = MODEL_SCALE[model]
        for rank in AFFINE_RANKS:
            values.append((f"{model}_mergeable_ar{rank}_s{scale_tag(scale)}_hr4", model, rank, scale))
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
    for label, _, _, _ in configs():
        for seed in SEEDS:
            for task in ("math", "gsm8k"):
                value = read_score(label, seed, task)
                if value is not None:
                    score[label, seed, task] = value

    lines = [
        "# Small-model mergeable AffLoRA math analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "MATH reports the clean-4,995 score. Treatments use mergeable tied input/lm_head AffLoRA on top of hidden LoRA hr4. This sweep fixes scale per model and varies affine rank.",
        "",
        "## Aggregate accuracy",
        "",
        "| config | completed | MATH mean ± sd | GSM8K mean ± sd |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, _, _, _ in configs():
        ms = [score[label, seed, "math"] for seed in SEEDS if (label, seed, "math") in score]
        gs = [score[label, seed, "gsm8k"] for seed in SEEDS if (label, seed, "gsm8k") in score]
        lines.append(f"| {label} | {min(len(ms), len(gs))}/3 | {mean_sd(ms)} | {mean_sd(gs)} |")

    lines += [
        "",
        "## Paired improvement over matching hidden-LoRA baseline",
        "",
        "| treatment | paired seeds | Δ MATH | MATH positive seeds | Δ GSM8K | GSM8K positive seeds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranking: list[tuple[str, int, float, float, float]] = []
    for label, model, rank, scale in configs():
        if rank is None:
            continue
        baseline = f"{model}_hidden_hr4"
        md = [
            score[label, seed, "math"] - score[baseline, seed, "math"]
            for seed in SEEDS
            if (label, seed, "math") in score and (baseline, seed, "math") in score
        ]
        gd = [
            score[label, seed, "gsm8k"] - score[baseline, seed, "gsm8k"]
            for seed in SEEDS
            if (label, seed, "gsm8k") in score and (baseline, seed, "gsm8k") in score
        ]
        lines.append(
            f"| {label} | {min(len(md), len(gd))}/3 | {paired_summary(md)} | "
            f"{sum(x > 0 for x in md)}/{len(md) if md else 0} | {paired_summary(gd)} | "
            f"{sum(x > 0 for x in gd)}/{len(gd) if gd else 0} |"
        )
        ms = [score[label, seed, "math"] for seed in SEEDS if (label, seed, "math") in score]
        gs = [score[label, seed, "gsm8k"] for seed in SEEDS if (label, seed, "gsm8k") in score]
        if len(ms) == len(gs) == 3:
            ranking.append((model, rank, statistics.mean(ms), statistics.mean(gs), statistics.mean(md) if len(md) == 3 else float("nan")))

    lines += ["", "## Ranking by model", ""]
    for model in MODELS:
        rows = [row for row in ranking if row[0] == model]
        lines += [f"### {model}", ""]
        if not rows:
            lines.append("Pending.")
            lines.append("")
            continue
        lines += ["| rank by MATH | affine rank | scale | MATH mean | GSM8K mean | Δ MATH vs hidden |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for idx, (_, rank, m, g, dm) in enumerate(sorted(rows, key=lambda x: x[2], reverse=True), 1):
            lines.append(f"| {idx} | {rank} | {MODEL_SCALE[model]:g} | {m:.4f}% | {g:.4f}% | {dm:+.4f} pp |")
        lines.append("")

    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
