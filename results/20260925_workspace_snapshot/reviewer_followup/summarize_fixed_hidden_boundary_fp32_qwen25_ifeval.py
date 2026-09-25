#!/usr/bin/env python
"""Summarize the six frozen fixed-hidden boundary IFEval endpoints.

All validation completes before either summary file is touched.  Prompt-pair
uncertainty (bootstrap/McNemar within a trained seed) is kept separate from
the primary paired-t interval over the three independently trained hidden
seeds.  IFEval is not used for endpoint selection.
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

import validate_fixed_hidden_boundary_ifeval as validation  # noqa: E402


DEFAULT_EXPERIMENT = (
    ROOT / "reviewer_followup/fixed_hidden_boundary_fp32_qwen25"
)
SEEDS = (42, 43, 44)
KINDS = ("alora", "vocab_lora")
MODES = ("strict", "loose")
EXPECTED_PROMPTS = 541
BOOTSTRAP_SAMPLES = 10_000
T_CRITICAL_975_DF2 = 4.302652729696142


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT
    )
    parser.add_argument(
        "--ifeval-dir",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ifeval_main.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar_p(alora_only: int, vocab_only: int) -> float:
    discordant = alora_only + vocab_only
    if discordant == 0:
        return 1.0
    lower = min(alora_only, vocab_only)
    lower_tail = math.fsum(
        math.comb(discordant, value) / (2**discordant)
        for value in range(lower + 1)
    )
    return min(1.0, 2.0 * lower_tail)


def bootstrap_rng_seed(seed: int, mode: str) -> int:
    return 2_000_000 + 10 * seed + (0 if mode == "strict" else 1)


def prompt_pair_statistics(
    alora: list[bool],
    vocab: list[bool],
    *,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    if len(alora) != EXPECTED_PROMPTS or len(vocab) != EXPECTED_PROMPTS:
        raise ValueError("Paired prompt outcomes must each contain exactly 541 rows")
    both_correct = sum(a and v for a, v in zip(alora, vocab, strict=True))
    alora_only = sum(a and not v for a, v in zip(alora, vocab, strict=True))
    vocab_only = sum(not a and v for a, v in zip(alora, vocab, strict=True))
    both_incorrect = EXPECTED_PROMPTS - both_correct - alora_only - vocab_only
    contingency = {
        "both_correct": both_correct,
        "alora_only_correct": alora_only,
        "vocab_lora_only_correct": vocab_only,
        "both_incorrect": both_incorrect,
    }
    if sum(contingency.values()) != EXPECTED_PROMPTS:
        raise AssertionError("Prompt contingency does not sum to 541")

    alora_correct = both_correct + alora_only
    vocab_correct = both_correct + vocab_only
    alora_accuracy = alora_correct / EXPECTED_PROMPTS
    vocab_accuracy = vocab_correct / EXPECTED_PROMPTS
    delta = alora_accuracy - vocab_accuracy

    rng_seed = bootstrap_rng_seed(seed, mode)
    rng = np.random.default_rng(rng_seed)
    probabilities = np.asarray(
        [both_correct, alora_only, vocab_only, both_incorrect],
        dtype=np.float64,
    ) / EXPECTED_PROMPTS
    draws = rng.multinomial(
        EXPECTED_PROMPTS, probabilities, size=BOOTSTRAP_SAMPLES
    )
    bootstrap_delta = (draws[:, 1] - draws[:, 2]) / EXPECTED_PROMPTS
    ci = [
        float(np.quantile(bootstrap_delta, 0.025)),
        float(np.quantile(bootstrap_delta, 0.975)),
    ]

    return {
        "hidden_seed": seed,
        "boundary_seed": seed,
        "mode": mode,
        "n_paired_prompts": EXPECTED_PROMPTS,
        "alora_correct": alora_correct,
        "vocab_lora_correct": vocab_correct,
        "alora_accuracy": alora_accuracy,
        "vocab_lora_accuracy": vocab_accuracy,
        "delta_accuracy_alora_minus_vocab_lora": delta,
        "delta_percentage_points_alora_minus_vocab_lora": 100.0 * delta,
        "direction": (
            "alora_better"
            if delta > 0
            else "vocab_lora_better"
            if delta < 0
            else "tie"
        ),
        "paired_prompt_contingency": contingency,
        "mcnemar_exact": {
            "test": "two-sided exact binomial test on discordant prompt pairs",
            "discordant_prompts": alora_only + vocab_only,
            "null_probability_alora_only": 0.5,
            "p_value": exact_mcnemar_p(alora_only, vocab_only),
        },
        "paired_prompt_bootstrap": {
            "unit": "paired IFEval prompt within this trained hidden seed",
            "method": "paired nonparametric percentile bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "rng_seed": rng_seed,
            "ci95_delta_accuracy_alora_minus_vocab_lora": ci,
            "ci95_delta_percentage_points_alora_minus_vocab_lora": [
                100.0 * ci[0],
                100.0 * ci[1],
            ],
        },
    }


def seed_level_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["hidden_seed"] for row in rows] != list(SEEDS):
        raise ValueError("Seed-level rows must be ordered as hidden seeds 42,43,44")
    alora = [float(row["alora_accuracy"]) for row in rows]
    vocab = [float(row["vocab_lora_accuracy"]) for row in rows]
    deltas = [
        float(row["delta_accuracy_alora_minus_vocab_lora"]) for row in rows
    ]
    delta_mean = statistics.fmean(deltas)
    delta_sd = statistics.stdev(deltas)
    margin = T_CRITICAL_975_DF2 * delta_sd / math.sqrt(len(deltas))
    ci = [delta_mean - margin, delta_mean + margin]
    return {
        "inferential_unit": "independently trained frozen-hidden seed pair",
        "n_seed_pairs": len(rows),
        "hidden_seeds": list(SEEDS),
        "alora_accuracy_mean": statistics.fmean(alora),
        "alora_accuracy_sample_sd": statistics.stdev(alora),
        "vocab_lora_accuracy_mean": statistics.fmean(vocab),
        "vocab_lora_accuracy_sample_sd": statistics.stdev(vocab),
        "delta_accuracy_alora_minus_vocab_lora_mean": delta_mean,
        "delta_accuracy_alora_minus_vocab_lora_sample_sd": delta_sd,
        "delta_percentage_points_alora_minus_vocab_lora_mean": 100.0
        * delta_mean,
        "directions": {
            "alora_better": sum(value > 0 for value in deltas),
            "vocab_lora_better": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
        },
        "paired_t_df": 2,
        "paired_t_critical_975": T_CRITICAL_975_DF2,
        "paired_t_ci95_delta_accuracy_alora_minus_vocab_lora": ci,
        "paired_t_ci95_delta_percentage_points_alora_minus_vocab_lora": [
            100.0 * ci[0],
            100.0 * ci[1],
        ],
        "paired_t_ci95_contains_zero": ci[0] <= 0.0 <= ci[1],
    }


def require_exact_artifact_set(
    response_dir: Path, score_root: Path, expected_names: set[str]
) -> None:
    if not response_dir.is_dir() or not score_root.is_dir():
        raise FileNotFoundError("IFEval response or score directory is missing")
    actual_responses = {path.stem for path in response_dir.glob("*.jsonl")}
    if actual_responses != expected_names:
        raise ValueError(
            "Response artifact set mismatch: "
            f"missing={sorted(expected_names - actual_responses)}, "
            f"unexpected={sorted(actual_responses - expected_names)}"
        )
    actual_score_dirs = {
        path.name for path in score_root.iterdir() if path.is_dir()
    }
    if actual_score_dirs != expected_names:
        raise ValueError(
            "Score-directory set mismatch: "
            f"missing={sorted(expected_names - actual_score_dirs)}, "
            f"unexpected={sorted(actual_score_dirs - expected_names)}"
        )
    expected_score_files = {
        "eval_results_strict.jsonl",
        "eval_results_loose.jsonl",
    }
    for name in sorted(expected_names):
        actual_score_files = {
            path.name
            for path in (score_root / name).iterdir()
            if path.is_file()
        }
        if actual_score_files != expected_score_files:
            raise ValueError(
                f"{score_root / name}: score-file set mismatch: "
                f"missing={sorted(expected_score_files - actual_score_files)}, "
                f"unexpected={sorted(actual_score_files - expected_score_files)}"
            )


def collect(experiment_dir: Path, ifeval_dir: Path) -> dict[str, Any]:
    expected_experiment = DEFAULT_EXPERIMENT.resolve()
    if experiment_dir.resolve() != expected_experiment:
        raise ValueError(
            "This summary is pinned to the fixed-hidden Qwen2.5 experiment: "
            f"{expected_experiment}"
        )
    protocol_path = ifeval_dir / "protocol_manifest.json"
    protocol = validation.validate_protocol_manifest(protocol_path)
    expected_names = set(validation.EXPECTED_RUN_NAMES)
    require_exact_artifact_set(
        ifeval_dir / "responses", ifeval_dir / "scores", expected_names
    )
    canonical = validation.canonical_rows()

    outcomes: dict[tuple[str, int, str], list[bool]] = {}
    artifacts: list[dict[str, Any]] = []
    checkpoint_identities: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for kind in KINDS:
            name = validation.run_name(kind, seed)
            run_dir = experiment_dir / "checkpoints" / name
            identity = validation.validate_checkpoint(run_dir)
            checkpoint_identities[name] = identity
            if protocol["runs"].get(name) != identity:
                raise ValueError(f"{name}: protocol/checkpoint identity mismatch")
            response_path = ifeval_dir / "responses" / f"{name}.jsonl"
            response_rows = validation.validate_responses(
                path=response_path,
                run_dir=run_dir,
                protocol_manifest=protocol_path,
                start_index=0,
                end_index=EXPECTED_PROMPTS,
                expected=canonical,
            )
            score_dir = ifeval_dir / "scores" / name
            result = {
                mode: validation.validate_score_rows(
                    score_dir / f"eval_results_{mode}.jsonl", response_rows
                )
                for mode in MODES
            }
            for mode in MODES:
                outcomes[(kind, seed, mode)] = result[mode]
            artifacts.append(
                {
                    "run_name": name,
                    "kind": kind,
                    "hidden_seed": seed,
                    "checkpoint_identity": identity,
                    "response_path": str(response_path.resolve()),
                    "response_sha256": sha256_file(response_path),
                    "strict_score_path": str(
                        (score_dir / "eval_results_strict.jsonl").resolve()
                    ),
                    "strict_score_sha256": sha256_file(
                        score_dir / "eval_results_strict.jsonl"
                    ),
                    "loose_score_path": str(
                        (score_dir / "eval_results_loose.jsonl").resolve()
                    ),
                    "loose_score_sha256": sha256_file(
                        score_dir / "eval_results_loose.jsonl"
                    ),
                }
            )

    per_seed: dict[str, dict[str, Any]] = {}
    primary: dict[str, Any] = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {
            mode: prompt_pair_statistics(
                outcomes[("alora", seed, mode)],
                outcomes[("vocab_lora", seed, mode)],
                seed=seed,
                mode=mode,
            )
            for mode in MODES
        }
    for mode in MODES:
        rows = [per_seed[str(seed)][mode] for seed in SEEDS]
        primary[mode] = {
            "per_seed": rows,
            "seed_level_summary": seed_level_summary(rows),
        }

    return {
        "marker": "fixed_hidden_boundary_fp32_qwen25_ifeval_summary_v1",
        "experiment_dir": str(experiment_dir.resolve()),
        "ifeval_dir": str(ifeval_dir.resolve()),
        "protocol_manifest": str(protocol_path.resolve()),
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "endpoint_policy": {
            "alora": "output A-LoRA rank 50, scale 16",
            "vocab_lora": "direct output Vocab-LoRA rank 1, scale 32",
            "hidden_seed_pairs": list(SEEDS),
            "ifeval_used_for_endpoint_selection": False,
        },
        "metric": {
            "name": "IFEval prompt-level all-instructions-followed accuracy",
            "modes": list(MODES),
            "difference": "A-LoRA minus Vocab-LoRA",
            "better_direction": "positive",
        },
        "uncertainty_interpretation": {
            "prompt_level": (
                "Within each trained seed, paired bootstrap and exact McNemar "
                "use the 541 prompt pairs; they do not estimate training-seed "
                "variation."
            ),
            "seed_level_primary": (
                "The primary interval is a paired-t interval over the three "
                "independently trained frozen-hidden seed pairs 42,43,44."
            ),
        },
        "primary_all_predeclared_seeds": primary,
        "per_seed": per_seed,
        "validated_artifacts": artifacts,
        "validation": {
            "status": "passed",
            "canonical_prompts": EXPECTED_PROMPTS,
            "canonical_sha256": validation.EXPECTED_CANONICAL_SHA256,
            "run_count": len(artifacts),
            "checkpoint_protocol_bindings": len(checkpoint_identities),
            "strict_score_rows": len(artifacts) * EXPECTED_PROMPTS,
            "loose_score_rows": len(artifacts) * EXPECTED_PROMPTS,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fixed-hidden FP32-boundary IFEval",
        "",
        "IFEval was evaluation-only: the two endpoints and hidden seeds "
        "42/43/44 were predeclared, and no IFEval result selected a configuration.",
        "",
        "## Primary seed-level result",
        "",
        "| mode | A-LoRA mean | Vocab-LoRA mean | A−V (pp) | "
        "paired-t 95% CI (pp) | directions (A/V/tie) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        result = summary["primary_all_predeclared_seeds"][mode][
            "seed_level_summary"
        ]
        ci = result[
            "paired_t_ci95_delta_percentage_points_alora_minus_vocab_lora"
        ]
        directions = result["directions"]
        lines.append(
            f"| {mode} | {100 * result['alora_accuracy_mean']:.3f}% | "
            f"{100 * result['vocab_lora_accuracy_mean']:.3f}% | "
            f"{result['delta_percentage_points_alora_minus_vocab_lora_mean']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{directions['alora_better']}/"
            f"{directions['vocab_lora_better']}/{directions['ties']} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed paired prompt results",
            "",
            "| seed | mode | A-LoRA | Vocab-LoRA | A−V (pp) | "
            "prompt bootstrap 95% CI (pp) | discordant A/V | McNemar p |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        for mode in MODES:
            row = summary["per_seed"][str(seed)][mode]
            ci = row["paired_prompt_bootstrap"][
                "ci95_delta_percentage_points_alora_minus_vocab_lora"
            ]
            contingency = row["paired_prompt_contingency"]
            lines.append(
                f"| {seed} | {mode} | {100 * row['alora_accuracy']:.3f}% | "
                f"{100 * row['vocab_lora_accuracy']:.3f}% | "
                f"{row['delta_percentage_points_alora_minus_vocab_lora']:+.3f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{contingency['alora_only_correct']}/"
                f"{contingency['vocab_lora_only_correct']} | "
                f"{row['mcnemar_exact']['p_value']:.6g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Prompt-level bootstrap intervals and McNemar tests describe "
            "paired prompt variation within one trained seed.",
            "- The paired-t interval uses trained hidden-seed pairs as the "
            "inferential unit; with three pairs it has 2 degrees of freedom.",
            "- Positive deltas favor A-LoRA. Strict and loose are reported "
            "separately.",
            "",
            "## Validation",
            "",
            f"- Six checkpoint-bound 541-row response files validated.",
            f"- Six strict and six loose official score files validated.",
            f"- Canonical digest: "
            f"`{summary['validation']['canonical_sha256']}`.",
            f"- Protocol manifest: `{summary['protocol_manifest']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    ifeval_dir = (
        args.ifeval_dir.resolve()
        if args.ifeval_dir is not None
        else experiment_dir / "ifeval_main"
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else experiment_dir / "ifeval_main_summary.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else experiment_dir / "ifeval_main_summary.md"
    )

    summary = collect(experiment_dir, ifeval_dir)
    atomic_write(
        output_json, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    atomic_write(output_md, render_markdown(summary))
    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "primary": {
                    mode: summary["primary_all_predeclared_seeds"][mode][
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
