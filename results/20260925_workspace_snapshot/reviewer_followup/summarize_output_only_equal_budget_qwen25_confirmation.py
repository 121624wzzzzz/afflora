#!/usr/bin/env python
"""Fail-closed final confirmation summary for output-only A-LoRA vs Vocab-LoRA.

The selected hyperparameters are read from ``final_selection.json``.  All
seed-42--45 dev/test reports and paired-bootstrap comparison artifacts are
validated before any summary is written.  The primary inferential unit is the
independently trained seed, so the primary test estimate uses only seeds
43--45.  Seed 42, which selected the hyperparameters on dev, is kept separate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "reviewer_followup/output_only_equal_budget_qwen25"
EXPECTED_EXAMPLES = 1000
BOOTSTRAP_SAMPLES = 10_000
INDEPENDENT_SEEDS = (43, 44, 45)
ALL_SEEDS = (42, 43, 44, 45)
SPLITS = ("dev", "test")
T_CRITICAL_975_DF2 = 4.302652729696142

METHODS: dict[str, dict[str, Any]] = {
    "aff_r50": {
        "label": "output-only A-LoRA r50",
        "variant": "affine_lm_head_plus_hidden_lora",
        "trainable_parameters": 9_385_984,
    },
    "vocab_r1": {
        "label": "output-only Vocab-LoRA r1",
        "variant": "hidden_lora",
        "trainable_parameters": 9_385_856,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="Output-only equal-budget experiment directory.",
    )
    parser.add_argument(
        "--final-selection",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/final_selection.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/confirmation_summary.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/confirmation_summary.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_equal(source: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{source}={actual!r}, expected {expected!r}")


def require_close(
    source: str,
    actual: Any,
    expected: float,
    *,
    abs_tol: float = 1e-11,
) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=abs_tol
    ):
        raise ValueError(f"{source}={actual!r}, expected {expected!r}")


def resolve_recorded_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected a non-empty recorded path, got {value!r}")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def lr_suffix(boundary_lr_scale: float) -> str:
    if boundary_lr_scale == 1.0:
        return ""
    if boundary_lr_scale == 0.5:
        return "_blr0p5"
    if boundary_lr_scale == 2.0:
        return "_blr2"
    raise ValueError(
        f"Unsupported selected boundary-LR multiplier: {boundary_lr_scale}"
    )


def run_name(method: str, scale: int, boundary_lr_scale: float, seed: int) -> str:
    return (
        f"qwen25_15b_out_{method}_s{scale}"
        f"{lr_suffix(boundary_lr_scale)}_sd{seed}"
    )


def config_tag(scale: int, boundary_lr_scale: float) -> str:
    return f"s{scale}{lr_suffix(boundary_lr_scale)}"


def validate_selection(
    selection: dict[str, Any], experiment_dir: Path
) -> dict[str, dict[str, float | int | str]]:
    require_equal(
        "final_selection.experiment",
        resolve_recorded_path(selection.get("experiment")),
        experiment_dir.resolve(),
    )
    protocol = selection.get("selection_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("final_selection.selection_protocol is missing")
    require_equal("selection split", protocol.get("split"), "corrected dev")
    require_equal("selection seed", protocol.get("seed"), 42)
    require_equal(
        "test_or_ifeval_used_for_selection",
        protocol.get("test_or_ifeval_used_for_selection"),
        False,
    )
    require_equal(
        "final selection validation status",
        selection.get("validation", {}).get("status"),
        "passed",
    )
    require_equal(
        "search_stopped_after_round2",
        selection.get("search_stopped_after_round2"),
        True,
    )

    selected = selection.get("selected_hyperparameters")
    methods = selection.get("methods")
    if not isinstance(selected, dict) or not isinstance(methods, dict):
        raise ValueError("final_selection selected method metadata is missing")

    normalized: dict[str, dict[str, float | int | str]] = {}
    for method in METHODS:
        item = selected.get(method)
        selected_run = methods.get(method, {}).get("selected")
        if not isinstance(item, dict) or not isinstance(selected_run, dict):
            raise ValueError(f"Missing selected metadata for {method}")
        scale = item.get("scale")
        boundary_lr_scale = item.get("boundary_lr_scale")
        if not isinstance(scale, int) or scale not in {1, 2, 4, 8, 16, 32}:
            raise ValueError(f"{method}: invalid selected scale {scale!r}")
        if not isinstance(boundary_lr_scale, (int, float)):
            raise ValueError(f"{method}: invalid boundary-LR multiplier")
        boundary_lr_scale = float(boundary_lr_scale)
        lr_suffix(boundary_lr_scale)
        expected_run = run_name(method, scale, boundary_lr_scale, 42)
        require_equal(f"{method} selected run_name", item.get("run_name"), expected_run)
        require_equal(
            f"{method} methods.selected.run_name",
            selected_run.get("run_name"),
            expected_run,
        )
        require_equal(
            f"{method} methods.selected.scale", selected_run.get("scale"), scale
        )
        require_close(
            f"{method} methods.selected.boundary_lr_scale",
            selected_run.get("boundary_lr_scale"),
            boundary_lr_scale,
        )
        require_close(
            f"{method} selected boundary_learning_rate",
            item.get("boundary_learning_rate"),
            2e-4 * boundary_lr_scale,
        )
        normalized[method] = {
            "scale": scale,
            "boundary_lr_scale": boundary_lr_scale,
            "boundary_learning_rate": 2e-4 * boundary_lr_scale,
            "seed42_run_name": expected_run,
            "dev_avg_ce_used_for_selection": float(item["dev_avg_ce"]),
        }
    return normalized


def validate_run_metadata(
    method: str,
    selected: dict[str, float | int | str],
    seed: int,
    run_dir: Path,
) -> None:
    args = load_json(run_dir / "run_args.json")
    params = load_json(run_dir / "trainable_summary.json")
    common = {
        "seed": seed,
        "hidden_lora_rank": 8,
        "hidden_lora_alpha": 16,
        "hidden_lora_dropout": 0.05,
        "hidden_lora_target_modules": (
            "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
        ),
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1,
        "save_strategy": "no",
        "master_dtype": "fp32",
        "variant": METHODS[method]["variant"],
    }
    for key, expected in common.items():
        require_equal(f"{run_dir.name} run_args.{key}", args.get(key), expected)
    require_equal(
        f"{run_dir.name} run_args.output_dir",
        resolve_recorded_path(args.get("output_dir")),
        run_dir.resolve(),
    )
    require_equal(
        f"{run_dir.name} trainable parameters",
        params.get("trainable"),
        METHODS[method]["trainable_parameters"],
    )

    scale = int(selected["scale"])
    boundary_lr_scale = float(selected["boundary_lr_scale"])
    if method == "aff_r50":
        require_equal(f"{run_dir.name} affine_rank", args.get("affine_rank"), 50)
        require_close(
            f"{run_dir.name} affine_alpha", args.get("affine_alpha"), 50 * scale
        )
        require_close(
            f"{run_dir.name} affine_dropout", args.get("affine_dropout"), 0
        )
        require_equal(
            f"{run_dir.name} no_affine_input_bias",
            args.get("no_affine_input_bias"),
            True,
        )
        require_close(
            f"{run_dir.name} affine_learning_rate_scale",
            args.get("affine_learning_rate_scale"),
            boundary_lr_scale,
        )
    else:
        require_equal(
            f"{run_dir.name} include_emb_lmh_lora_rank",
            args.get("include_emb_lmh_lora_rank"),
            1,
        )
        require_equal(
            f"{run_dir.name} emb_lmh_lora_alpha",
            args.get("emb_lmh_lora_alpha"),
            scale,
        )
        if boundary_lr_scale != 1.0:
            audit = args.get("output_vocab_lr")
            if not isinstance(audit, dict):
                raise ValueError(f"{run_dir.name}: output_vocab_lr audit is missing")
            require_close(
                f"{run_dir.name} hidden optimizer LR",
                audit.get("hidden_learning_rate"),
                2e-4,
            )
            require_close(
                f"{run_dir.name} boundary optimizer LR",
                audit.get("output_vocab_learning_rate"),
                2e-4 * boundary_lr_scale,
            )
            require_close(
                f"{run_dir.name} boundary optimizer LR scale",
                audit.get("output_vocab_lr_scale"),
                boundary_lr_scale,
            )
            require_equal(
                f"{run_dir.name} optimizer coverage",
                audit.get("coverage_assertion"),
                "passed",
            )


def validate_report(
    path: Path,
    method: str,
    seed: int,
    split: str,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, tuple[float, int]]]:
    report = load_json(path)
    expected_data = (
        ROOT / f"corrected_sft_experiment/data/{split}.jsonl"
    ).resolve()
    expected = {
        "num_examples": EXPECTED_EXAMPLES,
        "seed": seed,
        "source_start_index": 0,
        "source_end_index": EXPECTED_EXAMPLES,
        "variant": METHODS[method]["variant"],
        "affine_ablation": "none",
    }
    for key, value in expected.items():
        require_equal(f"{path} {key}", report.get(key), value)
    require_equal(
        f"{path} run_dir",
        resolve_recorded_path(report.get("run_dir")),
        run_dir.resolve(),
    )
    require_equal(
        f"{path} data",
        resolve_recorded_path(report.get("data")),
        expected_data,
    )

    rows = report.get("per_example")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EXAMPLES:
        raise ValueError(f"{path}: expected {EXPECTED_EXAMPLES} per-example rows")
    by_id: dict[str, tuple[float, int]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: per_example[{index}] is not an object")
        record_id = row.get("record_id")
        nll = row.get("nll_sum")
        count = row.get("token_count")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}: invalid record_id at row {index}")
        if record_id in by_id:
            raise ValueError(f"{path}: duplicate record_id {record_id}")
        if (
            not isinstance(nll, (int, float))
            or not math.isfinite(float(nll))
            or float(nll) < 0
        ):
            raise ValueError(f"{path}: invalid NLL for {record_id}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{path}: invalid token count for {record_id}")
        require_close(
            f"{path} mean_ce[{record_id}]",
            row.get("mean_ce"),
            float(nll) / count,
        )
        by_id[record_id] = (float(nll), count)

    total_nll = math.fsum(nll for nll, _ in by_id.values())
    total_tokens = sum(count for _, count in by_id.values())
    avg_ce = total_nll / total_tokens
    require_equal(
        f"{path} supervised_tokens",
        report.get("supervised_tokens"),
        total_tokens,
    )
    require_close(f"{path} total_nll", report.get("total_nll"), total_nll, abs_tol=1e-5)
    require_close(f"{path} avg_ce", report.get("avg_ce"), avg_ce)
    require_close(f"{path} perplexity", report.get("perplexity"), math.exp(avg_ce))
    return report, by_id


def comparison_path(
    experiment_dir: Path,
    selected: dict[str, dict[str, float | int | str]],
    seed: int,
    split: str,
) -> Path:
    aff = selected["aff_r50"]
    vocab = selected["vocab_r1"]
    name = (
        f"vocab_{config_tag(int(vocab['scale']), float(vocab['boundary_lr_scale']))}"
        f"_vs_aff_{config_tag(int(aff['scale']), float(aff['boundary_lr_scale']))}"
        f"_sd{seed}_{split}.json"
    )
    return experiment_dir / "phase2_v2_comparisons" / name


def validate_comparison(
    path: Path,
    baseline_path: Path,
    treatment_path: Path,
    baseline_rows: dict[str, tuple[float, int]],
    treatment_rows: dict[str, tuple[float, int]],
    seed: int,
) -> dict[str, Any]:
    comparison = load_json(path)
    require_equal(
        f"{path} baseline",
        resolve_recorded_path(comparison.get("baseline")),
        baseline_path.resolve(),
    )
    require_equal(
        f"{path} treatment",
        resolve_recorded_path(comparison.get("treatment")),
        treatment_path.resolve(),
    )
    require_equal(f"{path} examples", comparison.get("examples"), EXPECTED_EXAMPLES)
    require_equal(
        f"{path} bootstrap_samples",
        comparison.get("bootstrap_samples"),
        BOOTSTRAP_SAMPLES,
    )
    if baseline_rows.keys() != treatment_rows.keys():
        raise ValueError(f"{path}: method reports do not contain paired record IDs")

    ids = sorted(baseline_rows)
    baseline_nll = np.array([baseline_rows[key][0] for key in ids], dtype=np.float64)
    treatment_nll = np.array(
        [treatment_rows[key][0] for key in ids], dtype=np.float64
    )
    baseline_counts = np.array(
        [baseline_rows[key][1] for key in ids], dtype=np.float64
    )
    treatment_counts = np.array(
        [treatment_rows[key][1] for key in ids], dtype=np.float64
    )
    if not np.array_equal(baseline_counts, treatment_counts):
        raise ValueError(f"{path}: supervised token counts are not paired")

    observed = float(
        (treatment_nll.sum() - baseline_nll.sum()) / baseline_counts.sum()
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(ids), size=len(ids))
        deltas[index] = (
            treatment_nll[sample].sum() - baseline_nll[sample].sum()
        ) / baseline_counts[sample].sum()
    expected_ci = [
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    ]
    expected_probability = float(np.mean(deltas < 0))

    require_close(
        f"{path} observed delta",
        comparison.get("delta_ce_treatment_minus_baseline"),
        observed,
    )
    ci = comparison.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{path}: ci95 must contain two values")
    require_close(f"{path} ci95 lower", ci[0], expected_ci[0])
    require_close(f"{path} ci95 upper", ci[1], expected_ci[1])
    require_close(
        f"{path} probability_treatment_better",
        comparison.get("probability_treatment_better"),
        expected_probability,
    )
    return {
        "path": str(path.resolve()),
        "delta_ce_aff_minus_vocab": observed,
        "per_example_bootstrap_ci95": expected_ci,
        "per_example_probability_aff_better": expected_probability,
    }


def direction_counts(deltas: list[float]) -> dict[str, int]:
    return {
        "aff_better_delta_lt_0": sum(delta < 0 for delta in deltas),
        "vocab_better_delta_gt_0": sum(delta > 0 for delta in deltas),
        "ties_delta_eq_0": sum(delta == 0 for delta in deltas),
    }


def descriptive_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aff = [float(row["aff_ce"]) for row in rows]
    vocab = [float(row["vocab_ce"]) for row in rows]
    deltas = [float(row["delta_ce_aff_minus_vocab"]) for row in rows]
    return {
        "n_seeds": len(rows),
        "aff_ce_mean": statistics.fmean(aff),
        "aff_ce_sample_sd": statistics.stdev(aff),
        "vocab_ce_mean": statistics.fmean(vocab),
        "vocab_ce_sample_sd": statistics.stdev(vocab),
        "delta_ce_aff_minus_vocab_mean": statistics.fmean(deltas),
        "delta_ce_aff_minus_vocab_sample_sd": statistics.stdev(deltas),
        "directions": direction_counts(deltas),
    }


def primary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["seed"] for row in rows] != list(INDEPENDENT_SEEDS):
        raise ValueError("Primary rows must be exactly seeds 43, 44, and 45")
    summary = descriptive_summary(rows)
    delta_mean = float(summary["delta_ce_aff_minus_vocab_mean"])
    delta_sd = float(summary["delta_ce_aff_minus_vocab_sample_sd"])
    margin = T_CRITICAL_975_DF2 * delta_sd / math.sqrt(len(rows))
    ci = [delta_mean - margin, delta_mean + margin]
    summary.update(
        {
            "paired_t_df": 2,
            "paired_t_critical_975": T_CRITICAL_975_DF2,
            "paired_t_ci95": ci,
            "paired_t_ci95_contains_zero": ci[0] <= 0 <= ci[1],
        }
    )
    return summary


def collect(
    experiment_dir: Path,
    selection_path: Path,
) -> dict[str, Any]:
    selection = load_json(selection_path)
    selected = validate_selection(selection, experiment_dir)
    checkpoints = experiment_dir / "checkpoints"

    reports: dict[tuple[str, int, str], dict[str, Any]] = {}
    report_rows: dict[tuple[str, int, str], dict[str, tuple[float, int]]] = {}
    report_paths: dict[tuple[str, int, str], Path] = {}
    reference_by_split: dict[str, dict[str, int]] = {}

    for method in METHODS:
        config = selected[method]
        for seed in ALL_SEEDS:
            name = run_name(
                method,
                int(config["scale"]),
                float(config["boundary_lr_scale"]),
                seed,
            )
            run_dir = checkpoints / name
            validate_run_metadata(method, config, seed, run_dir)
            for split in SPLITS:
                path = run_dir / f"{split}_report.json"
                report, rows = validate_report(
                    path, method, seed, split, run_dir
                )
                reports[(method, seed, split)] = report
                report_rows[(method, seed, split)] = rows
                report_paths[(method, seed, split)] = path
                counts = {key: value[1] for key, value in rows.items()}
                if split not in reference_by_split:
                    reference_by_split[split] = counts
                elif counts != reference_by_split[split]:
                    raise ValueError(
                        f"{name} {split}: IDs/token counts differ across runs"
                    )
        require_close(
            f"{method} selected dev CE vs seed42 report",
            config["dev_avg_ce_used_for_selection"],
            reports[(method, 42, "dev")]["avg_ce"],
        )

    comparisons: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in ALL_SEEDS:
        for split in SPLITS:
            vocab_key = ("vocab_r1", seed, split)
            aff_key = ("aff_r50", seed, split)
            path = comparison_path(experiment_dir, selected, seed, split)
            comparisons[(seed, split)] = validate_comparison(
                path,
                report_paths[vocab_key],
                report_paths[aff_key],
                report_rows[vocab_key],
                report_rows[aff_key],
                seed,
            )

    test_rows: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        aff_report = reports[("aff_r50", seed, "test")]
        vocab_report = reports[("vocab_r1", seed, "test")]
        comparison = comparisons[(seed, "test")]
        delta = float(aff_report["avg_ce"]) - float(vocab_report["avg_ce"])
        require_close(
            f"seed {seed} test report-vs-comparison delta",
            comparison["delta_ce_aff_minus_vocab"],
            delta,
        )
        test_rows.append(
            {
                "seed": seed,
                "aff_ce": float(aff_report["avg_ce"]),
                "vocab_ce": float(vocab_report["avg_ce"]),
                "delta_ce_aff_minus_vocab": delta,
                "direction": (
                    "aff_better"
                    if delta < 0
                    else "vocab_better"
                    if delta > 0
                    else "tie"
                ),
                "per_example_bootstrap_ci95": comparison[
                    "per_example_bootstrap_ci95"
                ],
                "per_example_probability_aff_better": comparison[
                    "per_example_probability_aff_better"
                ],
            }
        )

    primary_rows = [row for row in test_rows if row["seed"] in INDEPENDENT_SEEDS]
    seed42_row = next(row for row in test_rows if row["seed"] == 42)
    return {
        "experiment": str(experiment_dir.resolve()),
        "final_selection": str(selection_path.resolve()),
        "selected_hyperparameters": selected,
        "estimand": {
            "metric": "token-weighted assistant-only cross-entropy",
            "difference": "A-LoRA minus Vocab-LoRA",
            "better_direction": "negative",
            "primary_split": "corrected test",
            "primary_inferential_unit": "independently trained seed",
        },
        "validation": {
            "status": "passed",
            "reports_validated": 16,
            "comparisons_validated": 8,
            "seeds": list(ALL_SEEDS),
            "splits": list(SPLITS),
            "examples_per_report": EXPECTED_EXAMPLES,
            "bootstrap_recomputed": True,
            "test_used_for_selection": False,
        },
        "primary_test_independent_seeds": {
            "role": (
                "primary confirmation; only independently trained seeds 43-45 "
                "enter the seed-level paired-t interval"
            ),
            "per_seed": primary_rows,
            "summary": primary_summary(primary_rows),
        },
        "selected_seed42_test_descriptive": {
            "role": (
                "reported separately because seed42 dev selected the "
                "hyperparameters; excluded from the primary interval"
            ),
            **seed42_row,
        },
        "four_seed_test_descriptive": {
            "role": "descriptive only; no confirmatory interval",
            "per_seed": test_rows,
            "summary": descriptive_summary(test_rows),
        },
        "validated_comparisons": {
            f"seed{seed}_{split}": value
            for (seed, split), value in comparisons.items()
        },
    }


def format_number(value: float) -> str:
    return f"{value:.9f}"


def render_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selected_hyperparameters"]
    primary = summary["primary_test_independent_seeds"]
    seed42 = summary["selected_seed42_test_descriptive"]
    four = summary["four_seed_test_descriptive"]
    stats = primary["summary"]
    four_stats = four["summary"]

    lines = [
        "# Output-only equal-budget final confirmation",
        "",
        "The sign convention throughout is **A-LoRA CE − Vocab-LoRA CE**; "
        "negative values favor A-LoRA.",
        "",
        "## Frozen configurations",
        "",
        "| Method | Scale | Boundary-LR multiplier | Boundary LR |",
        "|---|---:|---:|---:|",
    ]
    for method in ("aff_r50", "vocab_r1"):
        item = selected[method]
        lines.append(
            f"| {METHODS[method]['label']} | {item['scale']} | "
            f"{item['boundary_lr_scale']:g} | "
            f"{item['boundary_learning_rate']:.7f} |"
        )

    lines.extend(
        [
            "",
            "## Primary test confirmation: independent seeds 43–45",
            "",
            "| Seed | A-LoRA CE | Vocab-LoRA CE | A−V | Direction |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in primary["per_seed"]:
        lines.append(
            f"| {row['seed']} | {format_number(row['aff_ce'])} | "
            f"{format_number(row['vocab_ce'])} | "
            f"{row['delta_ce_aff_minus_vocab']:+.9f} | "
            f"{row['direction']} |"
        )
    ci = stats["paired_t_ci95"]
    directions = stats["directions"]
    lines.extend(
        [
            "",
            f"- Mean A-LoRA CE: `{format_number(stats['aff_ce_mean'])}` "
            f"(sample SD `{format_number(stats['aff_ce_sample_sd'])}`)",
            f"- Mean Vocab-LoRA CE: `{format_number(stats['vocab_ce_mean'])}` "
            f"(sample SD `{format_number(stats['vocab_ce_sample_sd'])}`)",
            f"- Mean A−V: `{stats['delta_ce_aff_minus_vocab_mean']:+.9f}` "
            f"(sample SD `{format_number(stats['delta_ce_aff_minus_vocab_sample_sd'])}`)",
            f"- Seed-level paired-t 95% CI (df=2): "
            f"`[{ci[0]:+.9f}, {ci[1]:+.9f}]`",
            f"- Directions: A-LoRA better `{directions['aff_better_delta_lt_0']}/3`; "
            f"Vocab-LoRA better `{directions['vocab_better_delta_gt_0']}/3`; "
            f"ties `{directions['ties_delta_eq_0']}/3`.",
            "",
            "## Seed 42 test, reported separately",
            "",
            "| Seed | A-LoRA CE | Vocab-LoRA CE | A−V | Direction |",
            "|---:|---:|---:|---:|---|",
            f"| 42 | {format_number(seed42['aff_ce'])} | "
            f"{format_number(seed42['vocab_ce'])} | "
            f"{seed42['delta_ce_aff_minus_vocab']:+.9f} | "
            f"{seed42['direction']} |",
            "",
            "Seed 42 is excluded from the primary interval because its dev result "
            "was used to choose the frozen hyperparameters.",
            "",
            "## Four-seed descriptive view",
            "",
            f"- Mean A-LoRA CE: `{format_number(four_stats['aff_ce_mean'])}`",
            f"- Mean Vocab-LoRA CE: `{format_number(four_stats['vocab_ce_mean'])}`",
            f"- Mean A−V: `{four_stats['delta_ce_aff_minus_vocab_mean']:+.9f}` "
            f"(sample SD `{format_number(four_stats['delta_ce_aff_minus_vocab_sample_sd'])}`)",
            "",
            "This four-seed mean is descriptive only and is not the confirmatory estimate.",
            "",
            "## Artifact validation",
            "",
            "Validated 16 reports (dev/test × two methods × four seeds) and "
            "recomputed all eight 10,000-sample paired-bootstrap comparisons. "
            "All configuration, path, seed, record-ID, supervised-token, CE, "
            "and comparison checks passed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    selection_path = (
        args.final_selection.resolve()
        if args.final_selection
        else experiment_dir / "final_selection.json"
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json
        else experiment_dir / "confirmation_summary.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md
        else experiment_dir / "confirmation_summary.md"
    )
    summary = collect(experiment_dir, selection_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "primary_summary": summary[
                    "primary_test_independent_seeds"
                ]["summary"],
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
