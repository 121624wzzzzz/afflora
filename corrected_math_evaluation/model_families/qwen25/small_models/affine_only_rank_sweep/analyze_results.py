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
SEEDS = (42,)
MODELS = ("qwen25_05b", "qwen25_15b")
FAMILIES = ("emb", "mergeable")
AFFINE_RANKS = (1, 2, 4, 8, 16)
MODEL_SCALE = {
    "qwen25_05b": 8.0,
    "qwen25_15b": 1.0,
}
T_975_DF2 = 4.3026527297


def scale_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def configs() -> list[tuple[str, str, str, int | None, float | None]]:
    values: list[tuple[str, str, str, int | None, float | None]] = []
    for model in MODELS:
        values.append((f"{model}_base", model, "base", None, None))
        scale = MODEL_SCALE[model]
        for family in FAMILIES:
            for rank in AFFINE_RANKS:
                values.append((f"{model}_{family}_ar{rank}_s{scale_tag(scale)}", model, family, rank, scale))
    return values


def run_name(label: str, seed: int | None) -> str:
    return label if seed is None else f"{label}_seed{seed}"


def seeds_for(family: str) -> tuple[int | None, ...]:
    return (None,) if family == "base" else SEEDS


def read_score(label: str, seed: int | None, task: str) -> float | None:
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


def delta_summary(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:+.4f} pp"
    if len(values) == 2:
        return f"{mean:+.4f} ± {statistics.stdev(values):.4f} pp (n=2)"
    sd = statistics.stdev(values)
    half = T_975_DF2 * sd / math.sqrt(len(values))
    return f"{mean:+.4f} ± {sd:.4f} pp; 95% CI [{mean-half:+.4f}, {mean+half:+.4f}]"


def main() -> None:
    score: dict[tuple[str, int | None, str], float] = {}
    for label, _, family, _, _ in configs():
        for seed in seeds_for(family):
            for task in ("math", "gsm8k"):
                value = read_score(label, seed, task)
                if value is not None:
                    score[label, seed, task] = value

    lines = [
        "# Small-model affine-only AffLoRA math analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "MATH reports the clean-4,995 score. This sweep has no hidden LoRA. It compares input-embedding-only AffLoRA (`emb`) and mergeable tied input/lm_head AffLoRA (`mergeable`) against the frozen base model.",
        "",
        "## Aggregate accuracy",
        "",
        "| config | completed | MATH mean ± sd | GSM8K mean ± sd |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, _, family, _, _ in configs():
        seeds = seeds_for(family)
        ms = [score[label, seed, "math"] for seed in seeds if (label, seed, "math") in score]
        gs = [score[label, seed, "gsm8k"] for seed in seeds if (label, seed, "gsm8k") in score]
        lines.append(f"| {label} | {min(len(ms), len(gs))}/{len(seeds)} | {mean_sd(ms)} | {mean_sd(gs)} |")

    lines += [
        "",
        "## Improvement over frozen base",
        "",
        "| treatment | completed seeds | Δ MATH vs base | MATH positive seeds | Δ GSM8K vs base | GSM8K positive seeds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, model, family, _, _ in configs():
        if family == "base":
            continue
        base = f"{model}_base"
        md = [
            score[label, seed, "math"] - score[base, None, "math"]
            for seed in SEEDS
            if (label, seed, "math") in score and (base, None, "math") in score
        ]
        gd = [
            score[label, seed, "gsm8k"] - score[base, None, "gsm8k"]
            for seed in SEEDS
            if (label, seed, "gsm8k") in score and (base, None, "gsm8k") in score
        ]
        lines.append(
            f"| {label} | {min(len(md), len(gd))}/{len(SEEDS)} | {delta_summary(md)} | "
            f"{sum(x > 0 for x in md)}/{len(md) if md else 0} | {delta_summary(gd)} | "
            f"{sum(x > 0 for x in gd)}/{len(gd) if gd else 0} |"
        )

    lines += [
        "",
        "## Mergeable minus emb-only at same rank",
        "",
        "| model | rank | scale | Δ MATH mergeable-emb | Δ GSM8K mergeable-emb |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        scale = MODEL_SCALE[model]
        for rank in AFFINE_RANKS:
            emb = f"{model}_emb_ar{rank}_s{scale_tag(scale)}"
            mergeable = f"{model}_mergeable_ar{rank}_s{scale_tag(scale)}"
            md = [
                score[mergeable, seed, "math"] - score[emb, seed, "math"]
                for seed in SEEDS
                if (mergeable, seed, "math") in score and (emb, seed, "math") in score
            ]
            gd = [
                score[mergeable, seed, "gsm8k"] - score[emb, seed, "gsm8k"]
                for seed in SEEDS
                if (mergeable, seed, "gsm8k") in score and (emb, seed, "gsm8k") in score
            ]
            lines.append(f"| {model} | {rank} | {scale:g} | {delta_summary(md)} | {delta_summary(gd)} |")

    lines += ["", "## Ranking by model and family", ""]
    for model in MODELS:
        scale = MODEL_SCALE[model]
        for family in FAMILIES:
            rows = []
            for rank in AFFINE_RANKS:
                label = f"{model}_{family}_ar{rank}_s{scale_tag(scale)}"
                ms = [score[label, seed, "math"] for seed in SEEDS if (label, seed, "math") in score]
                gs = [score[label, seed, "gsm8k"] for seed in SEEDS if (label, seed, "gsm8k") in score]
                if ms and gs:
                    rows.append((rank, statistics.mean(ms), statistics.mean(gs)))
            lines += [f"### {model} {family}", ""]
            if not rows:
                lines += ["Pending.", ""]
                continue
            lines += ["| rank by MATH | affine rank | scale | MATH mean | GSM8K mean |", "| ---: | ---: | ---: | ---: | ---: |"]
            for idx, (rank, m, g) in enumerate(sorted(rows, key=lambda x: x[1], reverse=True), 1):
                lines.append(f"| {idx} | {rank} | {scale:g} | {m:.4f}% | {g:.4f}% |")
            lines.append("")

    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
