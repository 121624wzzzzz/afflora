#!/usr/bin/env python
"""Jointly select output-only scale and boundary LR using corrected dev only.

The candidate set is the original seed-42 scale sweep at boundary-LR
multiplier 1 plus the focused scale={2,8}, boundary-LR={0.5,2} extension.
The script fails closed unless all 16 runs and their paired dev examples pass
the original phase-1 checks and the additional optimizer-group checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import summarize_output_only_equal_budget_qwen25_phase1 as phase1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "reviewer_followup/output_only_equal_budget_qwen25"
FOCUSED_SCALES = (2, 8)
BOUNDARY_LR_SCALES = (0.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="Output-only equal-budget experiment directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/joint_scale_boundary_lr_summary.json.",
    )
    return parser.parse_args()


def lr_tag(value: float) -> str:
    if value == 0.5:
        return "0p5"
    if value == 2.0:
        return "2"
    raise ValueError(f"No preregistered run-name tag for LR multiplier {value}")


def require_optimizer_log(
    run_name: str,
    experiment_dir: Path,
    method: str,
    boundary_lr_scale: float,
) -> None:
    path = experiment_dir / "logs" / f"{run_name}.train.log"
    if not path.is_file():
        raise FileNotFoundError(f"Required training log is missing: {path}")
    contents = path.read_text(encoding="utf-8", errors="replace")
    if method == "aff_r50":
        required = (
            "[optimizer] separate_affine_lr",
            f"base_lr={2e-4:g}",
            f"affine_lr={2e-4 * boundary_lr_scale:g}",
            f"affine_scale={boundary_lr_scale:g}",
            "'hidden_or_other': 9232384",
            "'affine_weight': 153600",
            "'affine_bias': 0",
        )
    else:
        required = (
            "[optimizer] output_vocab_boundary_lr",
            '"coverage_assertion": "passed"',
            '"hidden_lora_params": 9232384',
            '"output_vocab_lora_params": 153472',
            f'"output_vocab_lr_scale": {boundary_lr_scale}',
        )
    missing = [marker for marker in required if marker not in contents]
    if missing:
        raise ValueError(
            f"{run_name}: optimizer log is missing expected markers {missing}"
        )


def validate_boundary_optimizer(
    run_name: str,
    run_dir: Path,
    method: str,
    boundary_lr_scale: float,
    run_args: dict[str, Any],
) -> None:
    expected_boundary_lr = 2e-4 * boundary_lr_scale
    phase1.require_close(
        run_name,
        "run_args.affine_learning_rate_scale",
        run_args.get("affine_learning_rate_scale"),
        boundary_lr_scale if method == "aff_r50" else 1.0,
    )
    phase1.require_close(
        run_name,
        "run_args.affine_bias_learning_rate_scale",
        run_args.get("affine_bias_learning_rate_scale"),
        1.0,
    )
    if method == "aff_r50":
        if "output_vocab_lr" in run_args:
            raise ValueError(
                f"{run_name}: A-LoRA unexpectedly contains Vocab-LR metadata"
            )
        return

    audit = phase1.load_json(run_dir / "optimizer_groups.json")
    run_args_audit = run_args.get("output_vocab_lr")
    if not isinstance(run_args_audit, dict):
        raise ValueError(f"{run_name}: run_args.output_vocab_lr is missing")
    expected = {
        "hidden_learning_rate": 2e-4,
        "output_vocab_learning_rate": expected_boundary_lr,
        "output_vocab_lr_scale": boundary_lr_scale,
        "hidden_lora_params": 9_232_384,
        "output_vocab_lora_params": 153_472,
        "total_trainable_params": 9_385_856,
        "output_lora_rank": 1,
        "output_lora_a_shape": [1, 1536],
        "output_lora_b_shape": [151936, 1],
        "coverage_assertion": "passed",
    }
    for key, expected_value in expected.items():
        actual = audit.get(key)
        recorded = run_args_audit.get(key)
        if isinstance(expected_value, float):
            phase1.require_close(
                run_name, f"optimizer_groups.{key}", actual, expected_value
            )
            phase1.require_close(
                run_name, f"run_args.output_vocab_lr.{key}", recorded, expected_value
            )
        else:
            phase1.require_equal(
                run_name, f"optimizer_groups.{key}", actual, expected_value
            )
            phase1.require_equal(
                run_name,
                f"run_args.output_vocab_lr.{key}",
                recorded,
                expected_value,
            )

    groups = audit.get("optimizer_group_params")
    if not isinstance(groups, dict):
        raise ValueError(f"{run_name}: optimizer_group_params is missing")
    hidden_group_count = sum(
        int(value)
        for key, value in groups.items()
        if key.startswith("hidden_lora_")
    )
    boundary_group_count = sum(
        int(value)
        for key, value in groups.items()
        if key.startswith("output_vocab_lora_")
    )
    phase1.require_equal(
        run_name, "optimizer hidden group count", hidden_group_count, 9_232_384
    )
    phase1.require_equal(
        run_name, "optimizer boundary group count", boundary_group_count, 153_472
    )


def reference_dev_pairing(
    original_summary: dict[str, Any],
) -> tuple[list[str], list[int]]:
    first = original_summary["methods"]["aff_r50"]["runs"][0]
    report = phase1.load_json(Path(first["dev_report"]))
    rows = report["per_example"]
    return (
        [row["record_id"] for row in rows],
        [int(row["token_count"]) for row in rows],
    )


def collect(experiment_dir: Path) -> dict[str, Any]:
    original = phase1.collect(experiment_dir)
    reference_ids, reference_counts = reference_dev_pairing(original)
    checkpoints = experiment_dir / "checkpoints"

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in phase1.METHODS:
        rows_by_method[method] = [
            {
                **row,
                "boundary_lr_scale": 1.0,
                "boundary_learning_rate": 2e-4,
                "candidate_source": "original_scale_sweep",
            }
            for row in original["methods"][method]["runs"]
        ]

    extension_runs = 0
    for method in phase1.METHODS:
        for scale in FOCUSED_SCALES:
            for boundary_lr_scale in BOUNDARY_LR_SCALES:
                run_name = (
                    f"qwen25_15b_out_{method}_s{scale}_"
                    f"blr{lr_tag(boundary_lr_scale)}_sd42"
                )
                run_dir = checkpoints / run_name
                run_args = phase1.load_json(run_dir / "run_args.json")
                params = phase1.load_json(run_dir / "trainable_summary.json")
                report_path = run_dir / "dev_report.json"
                report = phase1.load_json(report_path)

                phase1.validate_common_run_args(run_name, run_dir, run_args)
                phase1.validate_method_run_args(
                    run_name, method, scale, run_args
                )
                phase1.validate_adapter_config(
                    run_name, run_dir, method, scale
                )
                phase1.validate_train_log(
                    run_name, experiment_dir, method
                )
                validate_boundary_optimizer(
                    run_name,
                    run_dir,
                    method,
                    boundary_lr_scale,
                    run_args,
                )
                require_optimizer_log(
                    run_name,
                    experiment_dir,
                    method,
                    boundary_lr_scale,
                )
                phase1.require_equal(
                    run_name,
                    "trainable_summary.trainable",
                    params.get("trainable"),
                    phase1.METHODS[method]["expected_trainable"],
                )
                ids, counts = phase1.validate_dev_report(
                    run_name, run_dir, report
                )
                if ids != reference_ids or counts != reference_counts:
                    raise ValueError(
                        f"{run_name}: dev examples or token counts are not paired "
                        "with the original LR=1 sweep"
                    )

                rows_by_method[method].append(
                    {
                        "scale": scale,
                        "alpha": (
                            50 * scale if method == "aff_r50" else scale
                        ),
                        "rank": 50 if method == "aff_r50" else 1,
                        "boundary_lr_scale": boundary_lr_scale,
                        "boundary_learning_rate": (
                            2e-4 * boundary_lr_scale
                        ),
                        "candidate_source": "focused_boundary_lr_extension",
                        "run_name": run_name,
                        "run_dir": str(run_dir.resolve()),
                        "dev_report": str(report_path.resolve()),
                        "dev_avg_ce": float(report["avg_ce"]),
                        "dev_perplexity": float(report["perplexity"]),
                        "dev_supervised_tokens": int(
                            report["supervised_tokens"]
                        ),
                        "trainable_parameters": int(params["trainable"]),
                        "boundary_parameters": phase1.METHODS[method][
                            "expected_boundary"
                        ],
                    }
                )
                extension_runs += 1

    methods: dict[str, dict[str, Any]] = {}
    selected_hyperparameters: dict[str, dict[str, float | int]] = {}
    for method, rows in rows_by_method.items():
        rows.sort(key=lambda row: (row["scale"], row["boundary_lr_scale"]))
        selected = min(
            rows,
            key=lambda row: (
                row["dev_avg_ce"],
                row["scale"],
                row["boundary_lr_scale"],
            ),
        )
        methods[method] = {
            "label": phase1.METHODS[method]["label"],
            "runs": rows,
            "selected": selected,
        }
        selected_hyperparameters[method] = {
            "scale": int(selected["scale"]),
            "boundary_lr_scale": float(selected["boundary_lr_scale"]),
            "boundary_learning_rate": float(
                selected["boundary_learning_rate"]
            ),
        }

    aff_selected = methods["aff_r50"]["selected"]
    vocab_selected = methods["vocab_r1"]["selected"]
    return {
        "experiment": str(experiment_dir.resolve()),
        "selection_protocol": {
            "split": "corrected dev",
            "seed": 42,
            "original_candidates": {
                "scales": list(phase1.SCALES),
                "boundary_lr_scales": [1.0],
            },
            "focused_extension_candidates": {
                "scales": list(FOCUSED_SCALES),
                "boundary_lr_scales": list(BOUNDARY_LR_SCALES),
            },
            "criterion": (
                "minimum token-weighted dev cross-entropy per method"
            ),
            "tie_break": "smaller functional scale, then smaller boundary LR",
            "test_used_for_selection": False,
        },
        "validation": {
            "status": "passed",
            "original_runs_validated": 8,
            "extension_runs_validated": extension_runs,
            "total_runs_validated": 8 + extension_runs,
            "paired_dev_examples": len(reference_ids),
            "paired_supervised_tokens": sum(reference_counts),
        },
        "original_phase1_selected_scales": original["selected_scales"],
        "methods": methods,
        "selected_hyperparameters": selected_hyperparameters,
        "selected_dev_delta_ce_aff_minus_vocab": (
            aff_selected["dev_avg_ce"] - vocab_selected["dev_avg_ce"]
        ),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_json = (
        args.output_json.resolve()
        if args.output_json
        else experiment_dir / "joint_scale_boundary_lr_summary.json"
    )
    summary = collect(experiment_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "selected_hyperparameters": summary[
                    "selected_hyperparameters"
                ],
                "selected_dev_delta_ce_aff_minus_vocab": summary[
                    "selected_dev_delta_ce_aff_minus_vocab"
                ],
                "output_json": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
