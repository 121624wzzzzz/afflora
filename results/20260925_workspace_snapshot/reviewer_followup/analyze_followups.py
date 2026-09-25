#!/usr/bin/env python
"""Summarize reviewer-followup experiments without silently dropping missing runs."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviewer_followup/SUMMARY.md"
T975 = {2: 4.3026527297, 4: 2.7764451052}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def mean_sd_ci(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:+.6f} (n=1)"
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    critical = T975.get(len(values) - 1, 1.96)
    half = critical * sd / math.sqrt(len(values))
    return f"{mean:+.6f} ± {sd:.6f}; 95% CI [{mean-half:+.6f}, {mean+half:+.6f}]"


def formal_sft(lines: list[str]) -> None:
    lines += ["## Corrected SFT: Qwen3-0.6B extra seeds", ""]
    rows = []
    deltas = []
    formal = ROOT / "corrected_sft_experiment/outputs/formal"
    for seed in (42, 43, 44):
        base = read(formal / f"qwen3_06b_hidden_sd{seed}/test_report.json")
        treat = read(formal / f"qwen3_06b_afflora_sd{seed}/test_report.json")
        comparison = read(formal / f"qwen3_06b_sd{seed}_test_comparison.json")
        if base and treat:
            delta = treat["avg_ce"] - base["avg_ce"]
            deltas.append(delta)
            ci = comparison["ci95"] if comparison else None
            rows.append(
                (seed, base["avg_ce"], treat["avg_ce"], delta, base["perplexity"], treat["perplexity"], ci)
            )
    lines += [
        "| seed | hidden CE | +A-LoRA CE | ΔCE | hidden PPL | +A-LoRA PPL | item-bootstrap CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for seed, b, t, d, bp, tp, ci in rows:
        ci_text = "-" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        lines.append(f"| {seed} | {b:.6f} | {t:.6f} | {d:+.6f} | {bp:.6f} | {tp:.6f} | {ci_text} |")
    lines += ["", f"Seed-paired ΔCE: {mean_sd_ci(deltas)}; positive improvements: {sum(x < 0 for x in deltas)}/{len(deltas)}.", ""]


def equal_budget(lines: list[str]) -> None:
    lines += ["## Corrected SFT: near-equal boundary budget", ""]
    base_root = ROOT / "corrected_sft_experiment/outputs/formal"
    exp = ROOT / "reviewer_followup/equal_budget_qwen25/checkpoints"
    methods = {
        "A-LoRA r48": "afflora_r48",
        "Vocab LoRA r1": "vocab_lora_r1",
    }
    lines += ["| method | completed | test CE mean | ΔCE vs hidden | trainable params |", "| --- | ---: | ---: | --- | ---: |"]
    for label, tag in methods.items():
        values, deltas, params = [], [], []
        for seed in (42, 43, 44):
            base = read(base_root / f"qwen25_15b_hidden_sd{seed}/test_report.json")
            report = read(exp / f"qwen25_15b_{tag}_sd{seed}/test_report.json")
            summary = read(exp / f"qwen25_15b_{tag}_sd{seed}/trainable_summary.json")
            if base and report:
                values.append(report["avg_ce"])
                deltas.append(report["avg_ce"] - base["avg_ce"])
            if summary:
                params.append(summary["trainable"])
        mean_value = "-" if not values else f"{statistics.mean(values):.6f}"
        param_value = "-" if not params else f"{round(statistics.mean(params)):,}"
        lines.append(f"| {label} | {len(values)}/3 | {mean_value} | {mean_sd_ci(deltas)} | {param_value} |")
    lines.append("")


def accuracy_contrast(
    lines: list[str], title: str, directory: Path,
    configs: list[tuple[str, str, str, int, float]],
) -> None:
    lines += [f"## {title}", ""]
    lines += [
        "| model/config | paired seeds | Δ MATH pp | MATH positive | Δ GSM8K pp | GSM8K positive |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    for label, alias, placement, rank, scale in configs:
        math_delta, gsm_delta = [], []
        for seed in (42, 43, 44, 45, 46, 47):
            base_name = f"{alias}_hidden_hr4_seed{seed}"
            treat_name = f"{alias}_{placement}_ar{rank}_s{scale:g}_hr4_seed{seed}"
            bm = read(directory / "outputs/math" / f"{base_name}_full.json")
            tm = read(directory / "outputs/math" / f"{treat_name}_full.json")
            bg = read(directory / "outputs/gsm8k" / f"{base_name}_full.json")
            tg = read(directory / "outputs/gsm8k" / f"{treat_name}_full.json")
            if bm and tm:
                math_delta.append(tm["clean"]["accuracy_pct"] - bm["clean"]["accuracy_pct"])
            if bg and tg:
                gsm_delta.append(tg["full"]["accuracy_pct"] - bg["full"]["accuracy_pct"])
        lines.append(
            f"| {label} | {min(len(math_delta), len(gsm_delta))} | "
            f"{mean_sd_ci(math_delta)} | {sum(x > 0 for x in math_delta)}/{len(math_delta)} | "
            f"{mean_sd_ci(gsm_delta)} | {sum(x > 0 for x in gsm_delta)}/{len(gsm_delta)} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Reviewer follow-up experiment summary", "", "Missing runs are shown as incomplete rather than omitted.", ""]
    formal_sft(lines)
    equal_budget(lines)
    accuracy_contrast(
        lines,
        "Qwen2.5-1.5B untouched-seed Math confirmation",
        ROOT / "reviewer_followup/qwen25_math_confirmation",
        [
            ("qwen25_15b + ar8 s1", "qwen25_15b", "mergeable", 8, 1.0),
            ("qwen25_15b + ar16 s1", "qwen25_15b", "mergeable", 16, 1.0),
        ],
    )
    accuracy_contrast(
        lines,
        "Llama-3.1/3.2 cross-version contrast",
        ROOT / "reviewer_followup/llama_cross_version",
        [
            ("Llama-3.1-8B output (R²=0.9976)", "llama31_8b", "lmhead", 16, 1.0),
            ("Llama-3.2-3B tied (R²=0.9795)", "llama32_3b", "mergeable", 16, 1.0),
        ],
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
