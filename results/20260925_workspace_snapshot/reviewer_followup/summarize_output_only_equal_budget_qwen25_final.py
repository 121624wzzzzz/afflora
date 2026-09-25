#!/usr/bin/env python
"""Final strict 18-run selection for output-only equal-budget Qwen2.5.

This reuses the adaptive summarizer's fail-closed artifact, safetensor,
initialization-hash, optimizer-group, and paired-dev validators.  Round 2 adds
``scale=32, boundary_lr_scale=1`` and
``scale=16, boundary_lr_scale=2`` for each method.  The search is frozen after
this round; the scale-32 versus scale-16 comparison is descriptive only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import summarize_output_only_equal_budget_qwen25_adaptive as strict
import summarize_output_only_equal_budget_qwen25_phase1 as phase1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "reviewer_followup/output_only_equal_budget_qwen25"
FINAL_CANDIDATES = strict.CANDIDATES + (
    (32, 1.0),
    (16, 2.0),
)


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
        help="Defaults to <experiment-dir>/final_selection.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/final_selection.md.",
    )
    return parser.parse_args()


def validate_candidate(
    experiment_dir: Path,
    method: str,
    scale: int,
    boundary_lr_scale: float,
) -> tuple[list[tuple[Any, ...]], dict[str, Any], str, str]:
    """Validate one run and return its dev signature, row, hash marker, pipeline SHA."""

    name = strict.run_name(method, scale, boundary_lr_scale)
    run_dir = experiment_dir / "checkpoints" / name
    strict.require_artifacts(
        name, run_dir, method, boundary_lr_scale
    )
    args = phase1.load_json(run_dir / "run_args.json")
    params = phase1.load_json(run_dir / "trainable_summary.json")
    report_path = run_dir / "dev_report.json"
    report = phase1.load_json(report_path)

    phase1.validate_common_run_args(name, run_dir, args)
    phase1.validate_method_run_args(name, method, scale, args)
    phase1.validate_adapter_config(name, run_dir, method, scale)
    phase1.validate_train_log(name, experiment_dir, method)
    strict.validate_trainable_summary(name, method, params)
    tensor_audit = strict.validate_tensor_artifacts(name, run_dir, method)
    pipeline_hash = strict.validate_pipeline_metadata(name, args)
    train_log, marker_kind = strict.validate_hidden_hash_and_completion(
        name, experiment_dir, method
    )
    optimizer_audit = strict.validate_optimizer(
        name,
        run_dir,
        method,
        boundary_lr_scale,
        args,
        train_log,
    )
    signatures, dev = strict.validate_dev(name, run_dir, args, report)
    row = {
        "run_name": name,
        "run_dir": str(run_dir.resolve()),
        "dev_report": str(report_path.resolve()),
        "scale": scale,
        "alpha": 50 * scale if method == "aff_r50" else scale,
        "rank": 50 if method == "aff_r50" else 1,
        "boundary_lr_scale": boundary_lr_scale,
        "boundary_learning_rate": strict.BASE_LR * boundary_lr_scale,
        "trainable_parameters": int(params["trainable"]),
        "boundary_parameters": phase1.METHODS[method]["expected_boundary"],
        **dev,
        "tensor_audit": tensor_audit,
        "optimizer_audit": optimizer_audit,
    }
    return signatures, row, marker_kind, pipeline_hash


def select_and_describe_trend(
    rows_by_method: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    methods: dict[str, dict[str, Any]] = {}
    trends: dict[str, dict[str, Any]] = {}
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
        s16 = next(
            row
            for row in rows
            if row["scale"] == 16 and row["boundary_lr_scale"] == 1.0
        )
        s32 = next(
            row
            for row in rows
            if row["scale"] == 32 and row["boundary_lr_scale"] == 1.0
        )
        delta = s32["dev_avg_ce"] - s16["dev_avg_ce"]
        direction = "improving" if delta < 0 else "worsening" if delta > 0 else "flat"
        trends[method] = {
            "scale16_blr1_dev_avg_ce": s16["dev_avg_ce"],
            "scale32_blr1_dev_avg_ce": s32["dev_avg_ce"],
            "scale32_minus_scale16_dev_ce": delta,
            "scale32_less_than_scale16": bool(delta < 0),
            "direction": direction,
        }
        methods[method] = {
            "label": phase1.METHODS[method]["label"],
            "runs": rows,
            "selected": selected,
        }
    return methods, trends


def collect(experiment_dir: Path) -> dict[str, Any]:
    reference_signatures: list[tuple[Any, ...]] | None = None
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in phase1.METHODS
    }
    hash_marker_kinds: dict[str, str] = {}
    pipeline_hashes: set[str] = set()

    for method in phase1.METHODS:
        for scale, boundary_lr_scale in FINAL_CANDIDATES:
            signatures, row, marker_kind, pipeline_hash = validate_candidate(
                experiment_dir, method, scale, boundary_lr_scale
            )
            if reference_signatures is None:
                reference_signatures = signatures
            elif signatures != reference_signatures:
                raise ValueError(
                    f"{row['run_name']}: dev record/source/token signature is not paired"
                )
            rows_by_method[method].append(row)
            hash_marker_kinds[method] = marker_kind
            pipeline_hashes.add(pipeline_hash)

    phase1.require_equal(
        "all runs", "corrected pipeline hash count", len(pipeline_hashes), 1
    )
    methods, trends = select_and_describe_trend(rows_by_method)
    selected = {
        method: {
            "scale": int(block["selected"]["scale"]),
            "boundary_lr_scale": float(
                block["selected"]["boundary_lr_scale"]
            ),
            "boundary_learning_rate": float(
                block["selected"]["boundary_learning_rate"]
            ),
            "dev_avg_ce": float(block["selected"]["dev_avg_ce"]),
            "run_name": block["selected"]["run_name"],
        }
        for method, block in methods.items()
    }
    return {
        "experiment": str(experiment_dir.resolve()),
        "selection_protocol": {
            "split": "corrected dev",
            "seed": 42,
            "candidates_per_method": [
                {"scale": scale, "boundary_lr_scale": boundary_lr_scale}
                for scale, boundary_lr_scale in FINAL_CANDIDATES
            ],
            "criterion": "minimum token-weighted dev cross-entropy per method",
            "tie_break": "smaller scale, then smaller boundary LR multiplier",
            "test_or_ifeval_used_for_selection": False,
            "search_stopping_rule": (
                "fixed stop after round 2; scale32-vs-scale16 is descriptive "
                "and does not trigger another extension"
            ),
        },
        "validation": {
            "status": "passed",
            "runs_validated": sum(len(rows) for rows in rows_by_method.values()),
            "paired_dev_examples": len(reference_signatures or []),
            "paired_supervised_tokens": sum(
                int(signature[3]) for signature in (reference_signatures or [])
            ),
            "hidden_lora_init_sha256": strict.EXPECTED_HIDDEN_HASH,
            "hidden_hash_marker_by_method": hash_marker_kinds,
            "corrected_pipeline_sha256": next(iter(pipeline_hashes)),
            "safetensor_headers_and_parameter_counts": "passed",
            "optimizer_group_checks": "passed",
        },
        "search_stopped_after_round2": True,
        "methods": methods,
        "selected_hyperparameters": selected,
        "scale32_vs_scale16_trend": trends,
        "selected_dev_delta_ce_aff_minus_vocab": (
            selected["aff_r50"]["dev_avg_ce"]
            - selected["vocab_r1"]["dev_avg_ce"]
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    lines = [
        "# Output-only equal-budget final selection",
        "",
        "Selection uses only the corrected 1,000-example dev split at seed 42. "
        "The search is frozen after round 2.",
        "",
        "## Validation",
        "",
        f"- Runs validated: `{validation['runs_validated']}`",
        f"- Paired dev examples: `{validation['paired_dev_examples']}`",
        (
            "- Hidden LoRA initialization SHA256: "
            f"`{validation['hidden_lora_init_sha256']}`"
        ),
        "- Safetensor parameter audit: `passed`",
        "- Optimizer-group audit: `passed`",
        "- Search stopped after round 2: `true`",
        "",
        "## Candidates",
        "",
        "| Method | Scale | Boundary LR multiplier | Boundary LR | Dev CE | Selected |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for method in ("aff_r50", "vocab_r1"):
        block = summary["methods"][method]
        selected_name = block["selected"]["run_name"]
        for row in block["runs"]:
            lines.append(
                "| {label} | {scale} | {lr_scale:g} | {lr:.1e} | "
                "{ce:.9f} | {selected} |".format(
                    label=block["label"],
                    scale=row["scale"],
                    lr_scale=row["boundary_lr_scale"],
                    lr=row["boundary_learning_rate"],
                    ce=row["dev_avg_ce"],
                    selected="yes" if row["run_name"] == selected_name else "",
                )
            )
    lines.extend(
        [
            "",
            "## Scale 32 versus scale 16 trend",
            "",
            "This comparison holds the boundary-LR multiplier at 1. It is "
            "descriptive only; no further sweep is triggered.",
            "",
            "| Method | Scale 16 CE | Scale 32 CE | ΔCE (32−16) | Direction |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for method in ("aff_r50", "vocab_r1"):
        trend = summary["scale32_vs_scale16_trend"][method]
        lines.append(
            "| {label} | {s16:.9f} | {s32:.9f} | {delta:+.9f} | {direction} |".format(
                label=summary["methods"][method]["label"],
                s16=trend["scale16_blr1_dev_avg_ce"],
                s32=trend["scale32_blr1_dev_avg_ce"],
                delta=trend["scale32_minus_scale16_dev_ce"],
                direction=trend["direction"],
            )
        )
    lines.extend(["", "## Final selection", ""])
    for method in ("aff_r50", "vocab_r1"):
        selected = summary["selected_hyperparameters"][method]
        lines.append(
            f"- {summary['methods'][method]['label']}: "
            f"`scale={selected['scale']}`, "
            f"`boundary_lr_scale={selected['boundary_lr_scale']:g}`, "
            f"`dev_ce={selected['dev_avg_ce']:.9f}`, "
            f"`run={selected['run_name']}`"
        )
    lines.extend(
        [
            "",
            (
                "- Selected dev ΔCE (A-LoRA − Vocab-LoRA): "
                f"`{summary['selected_dev_delta_ce_aff_minus_vocab']:+.9f}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_json = (
        args.output_json.resolve()
        if args.output_json
        else experiment_dir / "final_selection.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md
        else experiment_dir / "final_selection.md"
    )
    summary = collect(experiment_dir)
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
                "selected_hyperparameters": summary[
                    "selected_hyperparameters"
                ],
                "scale32_vs_scale16_trend": summary[
                    "scale32_vs_scale16_trend"
                ],
                "search_stopped_after_round2": True,
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
