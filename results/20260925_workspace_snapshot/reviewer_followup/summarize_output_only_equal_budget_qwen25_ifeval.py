#!/usr/bin/env python
"""Fail-closed final IFEval summary for the output-only equal-budget study.

The two frozen configurations are read from ``final_selection.json``.  The
summary is written only after all eight runs (two methods x seeds 42--45)
have one canonical 541-row response artifact and one valid strict and loose
official-score artifact.

Prompt-level paired analyses and seed-level confirmation are deliberately
kept separate:

* within each trained seed, the 541 paired prompts provide a contingency
  table, exact discordant-binomial (exact McNemar) p-value, and paired
  nonparametric-bootstrap interval;
* the primary across-training-run estimate uses only independent seeds
  43--45 and a paired-t interval over the three seed-level accuracy
  differences;
* selected seed 42 is reported separately, and all four seeds receive a
  descriptive-only summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import summarize_output_only_equal_budget_qwen25_confirmation as confirmation  # noqa: E402
import validate_output_only_ifeval as artifact_validation  # noqa: E402


DEFAULT_EXPERIMENT = ROOT / "reviewer_followup/output_only_equal_budget_qwen25"
EXPECTED_PROMPTS = 541
INDEPENDENT_SEEDS = (43, 44, 45)
ALL_SEEDS = (42, 43, 44, 45)
MODES = ("strict", "loose")
METHODS = ("aff_r50", "vocab_r1")
BOOTSTRAP_SAMPLES = 10_000
T_CRITICAL_975_DF2 = 4.302652729696142
SCORE_FIELDS = {
    "follow_all_instructions",
    "follow_instruction_list",
    "instruction_id_list",
    "prompt",
    "response",
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
        "--ifeval-dir",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ifeval_final.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ifeval_summary.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ifeval_summary.md.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(source: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{source}={actual!r}, expected {expected!r}")


def require_exact_artifact_set(
    response_dir: Path, score_root: Path, expected_names: set[str]
) -> None:
    if not response_dir.is_dir():
        raise FileNotFoundError(f"Required response directory is missing: {response_dir}")
    if not score_root.is_dir():
        raise FileNotFoundError(f"Required score directory is missing: {score_root}")

    actual_responses = {path.stem for path in response_dir.glob("*.jsonl")}
    if actual_responses != expected_names:
        raise ValueError(
            "Final response artifact set does not exactly match the eight frozen runs: "
            f"missing={sorted(expected_names - actual_responses)}, "
            f"unexpected={sorted(actual_responses - expected_names)}"
        )

    actual_score_dirs = {path.name for path in score_root.iterdir() if path.is_dir()}
    if actual_score_dirs != expected_names:
        raise ValueError(
            "Final score-directory set does not exactly match the eight frozen runs: "
            f"missing={sorted(expected_names - actual_score_dirs)}, "
            f"unexpected={sorted(actual_score_dirs - expected_names)}"
        )
    expected_score_files = {
        "eval_results_strict.jsonl",
        "eval_results_loose.jsonl",
    }
    for name in sorted(expected_names):
        score_dir = score_root / name
        actual_score_files = {
            path.name for path in score_dir.iterdir() if path.is_file()
        }
        if actual_score_files != expected_score_files:
            raise ValueError(
                f"{score_dir}: expected exactly {sorted(expected_score_files)}, "
                f"found {sorted(actual_score_files)}"
            )


def validate_score_rows(
    path: Path,
    response_rows: list[dict[str, Any]],
) -> list[bool]:
    rows = artifact_validation.read_jsonl(path)
    if len(rows) != EXPECTED_PROMPTS:
        raise ValueError(
            f"{path}: found {len(rows)} score rows; expected {EXPECTED_PROMPTS}"
        )
    outcomes: list[bool] = []
    for index, (row, response) in enumerate(
        zip(rows, response_rows, strict=True)
    ):
        if set(row) != SCORE_FIELDS:
            raise ValueError(
                f"{path}: row {index} fields={sorted(row)}, "
                f"expected {sorted(SCORE_FIELDS)}"
            )
        for field in ("prompt", "response", "instruction_id_list"):
            if row[field] != response[field]:
                raise ValueError(f"{path}: row {index} mismatched {field}")
        follow_all = row["follow_all_instructions"]
        follow_list = row["follow_instruction_list"]
        if type(follow_all) is not bool:
            raise ValueError(f"{path}: row {index} aggregate result is not boolean")
        if not isinstance(follow_list, list) or not all(
            type(value) is bool for value in follow_list
        ):
            raise ValueError(
                f"{path}: row {index} instruction results are not booleans"
            )
        if len(follow_list) != len(response["instruction_id_list"]):
            raise ValueError(
                f"{path}: row {index} instruction-result count mismatch"
            )
        if not follow_list:
            raise ValueError(f"{path}: row {index} has no instruction checks")
        if follow_all != all(follow_list):
            raise ValueError(
                f"{path}: row {index} aggregate result disagrees with "
                "per-instruction results"
            )
        outcomes.append(follow_all)
    return outcomes


def exact_mcnemar_p(aff_only: int, vocab_only: int) -> float:
    """Two-sided exact binomial p-value over discordant prompt pairs."""

    discordant = aff_only + vocab_only
    if discordant == 0:
        return 1.0
    lower = min(aff_only, vocab_only)
    lower_tail = math.fsum(
        math.comb(discordant, value) / (2**discordant)
        for value in range(lower + 1)
    )
    return min(1.0, 2.0 * lower_tail)


def bootstrap_seed(seed: int, mode: str) -> int:
    return 1_000_000 + 10 * seed + (0 if mode == "strict" else 1)


def prompt_pair_statistics(
    aff_outcomes: list[bool],
    vocab_outcomes: list[bool],
    *,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    if len(aff_outcomes) != EXPECTED_PROMPTS or len(vocab_outcomes) != (
        EXPECTED_PROMPTS
    ):
        raise ValueError("Prompt-pair statistic received a non-541 outcome vector")

    both_correct = sum(a and v for a, v in zip(aff_outcomes, vocab_outcomes))
    aff_only = sum(a and not v for a, v in zip(aff_outcomes, vocab_outcomes))
    vocab_only = sum(not a and v for a, v in zip(aff_outcomes, vocab_outcomes))
    both_incorrect = EXPECTED_PROMPTS - both_correct - aff_only - vocab_only
    contingency = {
        "both_correct": both_correct,
        "aff_only_correct": aff_only,
        "vocab_only_correct": vocab_only,
        "both_incorrect": both_incorrect,
    }
    if sum(contingency.values()) != EXPECTED_PROMPTS:
        raise AssertionError("Prompt contingency does not sum to 541")

    aff_correct = both_correct + aff_only
    vocab_correct = both_correct + vocab_only
    aff_accuracy = aff_correct / EXPECTED_PROMPTS
    vocab_accuracy = vocab_correct / EXPECTED_PROMPTS
    delta = aff_accuracy - vocab_accuracy
    require_equal(
        "paired prompt delta numerator",
        aff_only - vocab_only,
        aff_correct - vocab_correct,
    )

    # Resampling the four paired contingency cells is exactly equivalent to
    # resampling the 541 observed prompt pairs with replacement.
    rng_seed = bootstrap_seed(seed, mode)
    rng = np.random.default_rng(rng_seed)
    probabilities = np.array(
        [both_correct, aff_only, vocab_only, both_incorrect], dtype=np.float64
    ) / EXPECTED_PROMPTS
    bootstrap_counts = rng.multinomial(
        EXPECTED_PROMPTS, probabilities, size=BOOTSTRAP_SAMPLES
    )
    bootstrap_delta = (
        bootstrap_counts[:, 1] - bootstrap_counts[:, 2]
    ) / EXPECTED_PROMPTS
    ci = [
        float(np.quantile(bootstrap_delta, 0.025)),
        float(np.quantile(bootstrap_delta, 0.975)),
    ]

    return {
        "seed": seed,
        "mode": mode,
        "n_prompts": EXPECTED_PROMPTS,
        "aff_correct": aff_correct,
        "vocab_correct": vocab_correct,
        "aff_prompt_accuracy": aff_accuracy,
        "vocab_prompt_accuracy": vocab_accuracy,
        "delta_accuracy_aff_minus_vocab": delta,
        "delta_percentage_points_aff_minus_vocab": 100.0 * delta,
        "direction": (
            "aff_better"
            if delta > 0
            else "vocab_better"
            if delta < 0
            else "tie"
        ),
        "paired_prompt_contingency": contingency,
        "mcnemar_exact": {
            "test": "two-sided exact binomial test on discordant pairs",
            "null_discordant_probability_aff_only": 0.5,
            "discordant_prompts": aff_only + vocab_only,
            "p_value": exact_mcnemar_p(aff_only, vocab_only),
        },
        "paired_prompt_bootstrap": {
            "unit": "prompt pair within this trained seed",
            "method": "paired nonparametric percentile bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "rng_seed": rng_seed,
            "ci95_delta_accuracy_aff_minus_vocab": ci,
            "ci95_delta_percentage_points_aff_minus_vocab": [
                100.0 * ci[0],
                100.0 * ci[1],
            ],
        },
    }


def direction_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    deltas = [float(row["delta_accuracy_aff_minus_vocab"]) for row in rows]
    return {
        "aff_better_delta_gt_0": sum(value > 0 for value in deltas),
        "vocab_better_delta_lt_0": sum(value < 0 for value in deltas),
        "ties_delta_eq_0": sum(value == 0 for value in deltas),
    }


def descriptive_seed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("Seed-level descriptive summary requires at least two seeds")
    aff = [float(row["aff_prompt_accuracy"]) for row in rows]
    vocab = [float(row["vocab_prompt_accuracy"]) for row in rows]
    deltas = [float(row["delta_accuracy_aff_minus_vocab"]) for row in rows]
    return {
        "n_seeds": len(rows),
        "aff_accuracy_mean": statistics.fmean(aff),
        "aff_accuracy_sample_sd": statistics.stdev(aff),
        "vocab_accuracy_mean": statistics.fmean(vocab),
        "vocab_accuracy_sample_sd": statistics.stdev(vocab),
        "delta_accuracy_aff_minus_vocab_mean": statistics.fmean(deltas),
        "delta_accuracy_aff_minus_vocab_sample_sd": statistics.stdev(deltas),
        "delta_percentage_points_aff_minus_vocab_mean": (
            100.0 * statistics.fmean(deltas)
        ),
        "directions": direction_counts(rows),
    }


def primary_seed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(row["seed"]) for row in rows] != list(INDEPENDENT_SEEDS):
        raise ValueError("Primary IFEval rows must be exactly seeds 43, 44, 45")
    summary = descriptive_seed_summary(rows)
    delta_mean = float(summary["delta_accuracy_aff_minus_vocab_mean"])
    delta_sd = float(summary["delta_accuracy_aff_minus_vocab_sample_sd"])
    margin = T_CRITICAL_975_DF2 * delta_sd / math.sqrt(len(rows))
    ci = [delta_mean - margin, delta_mean + margin]
    summary.update(
        {
            "inferential_unit": "independently trained seed",
            "paired_t_df": 2,
            "paired_t_critical_975": T_CRITICAL_975_DF2,
            "paired_t_ci95_delta_accuracy_aff_minus_vocab": ci,
            "paired_t_ci95_delta_percentage_points_aff_minus_vocab": [
                100.0 * ci[0],
                100.0 * ci[1],
            ],
            "paired_t_ci95_contains_zero": ci[0] <= 0 <= ci[1],
        }
    )
    return summary


def validate_one_run(
    *,
    method: str,
    seed: int,
    selected: dict[str, float | int | str],
    experiment_dir: Path,
    ifeval_dir: Path,
    canonical_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[bool]], dict[str, Any]]:
    name = confirmation.run_name(
        method,
        int(selected["scale"]),
        float(selected["boundary_lr_scale"]),
        seed,
    )
    run_dir = experiment_dir / "checkpoints" / name
    confirmation.validate_run_metadata(method, selected, seed, run_dir)
    _, parsed_method, parsed_scale, parsed_boundary_lr, parsed_seed = (
        artifact_validation.validate_checkpoint(run_dir)
    )
    require_equal(f"{name} parsed method", parsed_method, method)
    require_equal(f"{name} parsed scale", parsed_scale, int(selected["scale"]))
    confirmation.require_close(
        f"{name} parsed boundary LR",
        parsed_boundary_lr,
        float(selected["boundary_lr_scale"]),
    )
    require_equal(f"{name} parsed seed", parsed_seed, seed)

    response_path = ifeval_dir / "responses" / f"{name}.jsonl"
    response_rows = artifact_validation.validate_responses(
        path=response_path,
        run_dir=run_dir,
        start_index=0,
        end_index=EXPECTED_PROMPTS,
        expected=canonical_rows,
    )
    outcomes: dict[str, list[bool]] = {}
    score_paths: dict[str, Path] = {}
    for mode in MODES:
        score_path = ifeval_dir / "scores" / name / f"eval_results_{mode}.jsonl"
        outcomes[mode] = validate_score_rows(score_path, response_rows)
        score_paths[mode] = score_path

    return outcomes, {
        "method": method,
        "seed": seed,
        "run_name": name,
        "run_dir": str(run_dir.resolve()),
        "response_path": str(response_path.resolve()),
        "response_sha256": sha256_file(response_path),
        "strict_score_path": str(score_paths["strict"].resolve()),
        "strict_score_sha256": sha256_file(score_paths["strict"]),
        "loose_score_path": str(score_paths["loose"].resolve()),
        "loose_score_sha256": sha256_file(score_paths["loose"]),
    }


def collect(
    experiment_dir: Path,
    selection_path: Path,
    ifeval_dir: Path,
) -> dict[str, Any]:
    selection = confirmation.load_json(selection_path)
    selected = confirmation.validate_selection(selection, experiment_dir)

    expected_names = {
        confirmation.run_name(
            method,
            int(selected[method]["scale"]),
            float(selected[method]["boundary_lr_scale"]),
            seed,
        )
        for method in METHODS
        for seed in ALL_SEEDS
    }
    require_equal("number of frozen IFEval runs", len(expected_names), 8)
    require_exact_artifact_set(
        ifeval_dir / "responses",
        ifeval_dir / "scores",
        expected_names,
    )

    canonical_rows = artifact_validation.canonical_rows()
    if len(canonical_rows) != EXPECTED_PROMPTS:
        raise ValueError(
            f"Canonical IFEval has {len(canonical_rows)} rows, "
            f"expected {EXPECTED_PROMPTS}"
        )
    outcomes: dict[tuple[str, int, str], list[bool]] = {}
    validated_artifacts: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        for method in METHODS:
            result, record = validate_one_run(
                method=method,
                seed=seed,
                selected=selected[method],
                experiment_dir=experiment_dir,
                ifeval_dir=ifeval_dir,
                canonical_rows=canonical_rows,
            )
            for mode in MODES:
                outcomes[(method, seed, mode)] = result[mode]
            validated_artifacts.append(record)

    per_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in ALL_SEEDS:
        per_seed[str(seed)] = {}
        for mode in MODES:
            per_seed[str(seed)][mode] = prompt_pair_statistics(
                outcomes[("aff_r50", seed, mode)],
                outcomes[("vocab_r1", seed, mode)],
                seed=seed,
                mode=mode,
            )

    primary: dict[str, Any] = {}
    four_seed: dict[str, Any] = {}
    seed42: dict[str, Any] = {}
    for mode in MODES:
        primary_rows = [
            per_seed[str(seed)][mode] for seed in INDEPENDENT_SEEDS
        ]
        four_rows = [per_seed[str(seed)][mode] for seed in ALL_SEEDS]
        primary[mode] = {
            "per_seed": primary_rows,
            "seed_level_summary": primary_seed_summary(primary_rows),
        }
        seed42[mode] = per_seed["42"][mode]
        four_seed[mode] = {
            "per_seed": four_rows,
            "seed_level_descriptive_summary": descriptive_seed_summary(four_rows),
        }

    canonical_digest = hashlib.sha256()
    for row in canonical_rows:
        canonical_digest.update(
            json.dumps(
                {
                    "key": row["key"],
                    "prompt": row["prompt"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        canonical_digest.update(b"\n")

    return {
        "experiment": str(experiment_dir.resolve()),
        "final_selection": str(selection_path.resolve()),
        "final_selection_sha256": sha256_file(selection_path),
        "ifeval_dir": str(ifeval_dir.resolve()),
        "selected_hyperparameters": selected,
        "metric": {
            "name": "IFEval prompt-level instruction-following accuracy",
            "score_field": "follow_all_instructions",
            "modes": list(MODES),
            "difference": "A-LoRA minus Vocab-LoRA",
            "better_direction": "positive",
        },
        "inferential_units": {
            "item_level": (
                "Within each trained seed, the 541 paired prompts support the "
                "contingency table, exact McNemar test, and paired-prompt "
                "bootstrap interval. These do not estimate training-seed variation."
            ),
            "seed_level_primary": (
                "Only independently trained seeds 43-45 enter the primary "
                "paired-t interval over seed-level A-LoRA-minus-Vocab-LoRA "
                "accuracy differences."
            ),
            "selected_seed42": (
                "Seed 42 selected the frozen hyperparameters on corrected dev "
                "and is excluded from the primary seed-level interval."
            ),
        },
        "validation": {
            "status": "passed",
            "runs_validated": 8,
            "response_artifacts_validated": 8,
            "strict_score_artifacts_validated": 8,
            "loose_score_artifacts_validated": 8,
            "prompts_per_artifact": EXPECTED_PROMPTS,
            "canonical_prompt_alignment": "passed",
            "canonical_prompt_metadata_sha256": canonical_digest.hexdigest(),
            "exact_artifact_set_required": True,
            "test_or_ifeval_used_for_selection": False,
        },
        "per_seed_paired_prompt_analysis": per_seed,
        "primary_independent_seeds_43_45": {
            "role": "primary seed-level confirmation",
            **primary,
        },
        "selected_seed42_descriptive": {
            "role": "reported separately; excluded from primary seed-level CI",
            **seed42,
        },
        "four_seed_descriptive": {
            "role": "descriptive only; no four-seed confirmatory interval",
            **four_seed,
        },
        "validated_artifacts": validated_artifacts,
    }


def format_accuracy(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def format_pp(value: float) -> str:
    return f"{100.0 * value:+.3f} pp"


def render_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selected_hyperparameters"]
    per_seed = summary["per_seed_paired_prompt_analysis"]
    primary = summary["primary_independent_seeds_43_45"]
    seed42 = summary["selected_seed42_descriptive"]
    four = summary["four_seed_descriptive"]

    lines = [
        "# Output-only equal-budget final IFEval summary",
        "",
        "The metric is official IFEval prompt-level accuracy "
        "(`follow_all_instructions`). The sign convention is **A-LoRA − "
        "Vocab-LoRA**; positive differences favor A-LoRA.",
        "",
        "## Frozen configurations",
        "",
        "| Method | Scale | Boundary-LR multiplier | Boundary LR |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        item = selected[method]
        lines.append(
            f"| {confirmation.METHODS[method]['label']} | {item['scale']} | "
            f"{item['boundary_lr_scale']:g} | "
            f"{item['boundary_learning_rate']:.7f} |"
        )

    lines.extend(
        [
            "",
            "## Inferential units",
            "",
            "- **Prompt/item level:** Within each trained seed, 541 paired "
            "prompts provide the contingency table, exact McNemar p-value, "
            "and paired-prompt bootstrap CI.",
            "- **Training-seed level:** The primary paired-t CI uses only "
            "independently trained seeds 43–45. It does not pool prompts "
            "across seeds.",
            "- Seed 42 selected the hyperparameters on corrected dev and is "
            "reported separately.",
        ]
    )

    for mode in MODES:
        title = "Strict" if mode == "strict" else "Loose"
        lines.extend(
            [
                "",
                f"## {title}: per-seed paired prompt analysis",
                "",
                "| Seed | Role | A acc. | V acc. | A−V | Direction | "
                "Both ✓ | A-only ✓ | V-only ✓ | Both ✗ | Bootstrap 95% CI | "
                "Exact p |",
                "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for seed in ALL_SEEDS:
            row = per_seed[str(seed)][mode]
            contingency = row["paired_prompt_contingency"]
            ci = row["paired_prompt_bootstrap"][
                "ci95_delta_accuracy_aff_minus_vocab"
            ]
            role = "selected" if seed == 42 else "primary"
            lines.append(
                f"| {seed} | {role} | "
                f"{format_accuracy(row['aff_prompt_accuracy'])} | "
                f"{format_accuracy(row['vocab_prompt_accuracy'])} | "
                f"{format_pp(row['delta_accuracy_aff_minus_vocab'])} | "
                f"{row['direction']} | "
                f"{contingency['both_correct']} | "
                f"{contingency['aff_only_correct']} | "
                f"{contingency['vocab_only_correct']} | "
                f"{contingency['both_incorrect']} | "
                f"[{format_pp(ci[0])}, {format_pp(ci[1])}] | "
                f"{row['mcnemar_exact']['p_value']:.6g} |"
            )

    lines.extend(["", "## Primary seed-level summary: seeds 43–45", ""])
    for mode in MODES:
        title = "Strict" if mode == "strict" else "Loose"
        stats = primary[mode]["seed_level_summary"]
        ci = stats["paired_t_ci95_delta_accuracy_aff_minus_vocab"]
        directions = stats["directions"]
        lines.extend(
            [
                f"### {title}",
                "",
                f"- A-LoRA mean `{format_accuracy(stats['aff_accuracy_mean'])}`, "
                f"sample SD `{format_accuracy(stats['aff_accuracy_sample_sd'])}`.",
                f"- Vocab-LoRA mean `{format_accuracy(stats['vocab_accuracy_mean'])}`, "
                f"sample SD `{format_accuracy(stats['vocab_accuracy_sample_sd'])}`.",
                f"- Mean A−V `{format_pp(stats['delta_accuracy_aff_minus_vocab_mean'])}`, "
                f"sample SD `{format_pp(stats['delta_accuracy_aff_minus_vocab_sample_sd'])}`.",
                f"- Seed-level paired-t 95% CI (df=2): "
                f"`[{format_pp(ci[0])}, {format_pp(ci[1])}]`.",
                f"- Directions: A-LoRA better "
                f"`{directions['aff_better_delta_gt_0']}/3`; Vocab-LoRA better "
                f"`{directions['vocab_better_delta_lt_0']}/3`; ties "
                f"`{directions['ties_delta_eq_0']}/3`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Seed 42, reported separately",
            "",
            "| Mode | A acc. | V acc. | A−V | Direction |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for mode in MODES:
        row = seed42[mode]
        lines.append(
            f"| {mode} | {format_accuracy(row['aff_prompt_accuracy'])} | "
            f"{format_accuracy(row['vocab_prompt_accuracy'])} | "
            f"{format_pp(row['delta_accuracy_aff_minus_vocab'])} | "
            f"{row['direction']} |"
        )
    lines.extend(
        [
            "",
            "Seed 42 is excluded from the primary paired-t interval because "
            "its corrected-dev results selected the frozen configurations.",
            "",
            "## Four-seed descriptive view",
            "",
            "| Mode | Mean A acc. | Mean V acc. | Mean A−V | Delta sample SD | "
            "Directions A/V/tie |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        stats = four[mode]["seed_level_descriptive_summary"]
        directions = stats["directions"]
        lines.append(
            f"| {mode} | {format_accuracy(stats['aff_accuracy_mean'])} | "
            f"{format_accuracy(stats['vocab_accuracy_mean'])} | "
            f"{format_pp(stats['delta_accuracy_aff_minus_vocab_mean'])} | "
            f"{format_pp(stats['delta_accuracy_aff_minus_vocab_sample_sd'])} | "
            f"{directions['aff_better_delta_gt_0']}/"
            f"{directions['vocab_better_delta_lt_0']}/"
            f"{directions['ties_delta_eq_0']} |"
        )
    lines.extend(
        [
            "",
            "The four-seed view is descriptive only.",
            "",
            "## Artifact validation",
            "",
            "Validated the exact frozen set of 8 response artifacts, 8 strict "
            "score artifacts, and 8 loose score artifacts. Every artifact "
            "contains exactly 541 rows aligned in canonical Google IFEval "
            "prompt order; response, instruction-list, checkpoint, method, "
            "scale, boundary-LR, and seed metadata checks all passed.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    selection_path = (
        args.final_selection.resolve()
        if args.final_selection
        else experiment_dir / "final_selection.json"
    )
    ifeval_dir = (
        args.ifeval_dir.resolve()
        if args.ifeval_dir
        else experiment_dir / "ifeval_final"
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json
        else experiment_dir / "ifeval_summary.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md
        else experiment_dir / "ifeval_summary.md"
    )

    # Fail closed: perform every validation and statistic before touching
    # either final summary path.
    summary = collect(experiment_dir, selection_path, ifeval_dir)
    json_contents = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    markdown_contents = render_markdown(summary)
    atomic_write_text(output_json, json_contents)
    atomic_write_text(output_md, markdown_contents)
    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "primary": {
                    mode: summary["primary_independent_seeds_43_45"][mode][
                        "seed_level_summary"
                    ]
                    for mode in MODES
                },
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
