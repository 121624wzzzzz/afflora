#!/usr/bin/env python
"""Validate and summarize the seed-42 output-only equal-budget scale sweep.

The phase-1 launcher creates four A-LoRA rank-50 runs and four output-only
Vocab-LoRA rank-1 runs.  This script deliberately fails closed: selection is
written only after all eight dev reports, run arguments, adapter
configurations, trainable-parameter summaries, and evaluation examples agree
with the preregistered comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "reviewer_followup/output_only_equal_budget_qwen25"
SCALES = (1, 2, 4, 8)
HIDDEN_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
}

METHODS: dict[str, dict[str, Any]] = {
    "aff_r50": {
        "label": "output-only A-LoRA r50",
        "expected_trainable": 9_385_984,
        "expected_boundary": 153_600,
        "variant": "affine_lm_head_plus_hidden_lora",
    },
    "vocab_r1": {
        "label": "output-only Vocab-LoRA r1",
        "expected_trainable": 9_385_856,
        "expected_boundary": 153_472,
        "variant": "hidden_lora",
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
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/phase1_summary.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/phase1_summary.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_equal(
    run_name: str, source: str, actual: Any, expected: Any
) -> None:
    if actual != expected:
        raise ValueError(
            f"{run_name}: {source}={actual!r}, expected {expected!r}"
        )


def require_close(
    run_name: str,
    source: str,
    actual: Any,
    expected: float,
    *,
    abs_tol: float = 1e-12,
) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=abs_tol
    ):
        raise ValueError(
            f"{run_name}: {source}={actual!r}, expected {expected!r}"
        )


def resolve_recorded_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected a non-empty recorded path, got {value!r}")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_common_run_args(
    run_name: str, run_dir: Path, args: dict[str, Any]
) -> None:
    expected = {
        "hidden_lora_rank": 8,
        "hidden_lora_alpha": 16,
        "hidden_lora_dropout": 0.05,
        "hidden_lora_target_modules": (
            "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
        ),
        "max_seq_len": 1024,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "max_steps": -1,
        "eval_samples": 1000,
        "eval_steps": 250,
        "logging_steps": 10,
        "save_strategy": "no",
        "bf16": True,
        "fp16": False,
        "master_dtype": "fp32",
        "seed": 42,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "max_grad_norm": 1,
    }
    for key, expected_value in expected.items():
        require_equal(run_name, f"run_args.{key}", args.get(key), expected_value)

    require_equal(
        run_name,
        "run_args.model_path",
        resolve_recorded_path(args.get("model_path")),
        (ROOT / "../models/Qwen2.5-1.5B-Base").resolve(),
    )
    require_equal(
        run_name,
        "run_args.train_data",
        resolve_recorded_path(args.get("train_data")),
        (ROOT / "corrected_sft_experiment/data/train.jsonl").resolve(),
    )
    require_equal(
        run_name,
        "run_args.eval_data",
        resolve_recorded_path(args.get("eval_data")),
        (ROOT / "corrected_sft_experiment/data/dev.jsonl").resolve(),
    )
    require_equal(
        run_name,
        "run_args.output_dir",
        resolve_recorded_path(args.get("output_dir")),
        run_dir.resolve(),
    )


def validate_method_run_args(
    run_name: str, method: str, scale: int, args: dict[str, Any]
) -> None:
    require_equal(
        run_name, "run_args.variant", args.get("variant"), METHODS[method]["variant"]
    )
    if method == "aff_r50":
        expected = {
            "affine_rank": 50,
            "affine_alpha": 50 * scale,
            "affine_dropout": 0,
            "no_affine_input_bias": True,
            "affine_lm_head_bias": False,
            "include_emb_lmh_lora_rank": 0,
            "affine_energy_lambda": 0,
            "affine_bias_energy_lambda": 0,
            "tie_affine_input_lm_head_adapters": False,
        }
    else:
        expected = {
            "include_emb_lmh_lora_rank": 1,
            "emb_lmh_lora_alpha": scale,
        }
    for key, expected_value in expected.items():
        require_equal(run_name, f"run_args.{key}", args.get(key), expected_value)


def validate_adapter_config(
    run_name: str, run_dir: Path, method: str, scale: int
) -> None:
    peft = load_json(run_dir / "adapter_config.json")
    expected_targets = HIDDEN_TARGETS | ({"lm_head"} if method == "vocab_r1" else set())
    require_equal(
        run_name,
        "adapter_config.target_modules",
        set(peft.get("target_modules", [])),
        expected_targets,
    )
    require_equal(run_name, "adapter_config.r", peft.get("r"), 8)
    require_equal(run_name, "adapter_config.lora_alpha", peft.get("lora_alpha"), 16)
    require_close(
        run_name, "adapter_config.lora_dropout", peft.get("lora_dropout"), 0.05
    )

    if method == "vocab_r1":
        require_equal(
            run_name,
            "adapter_config.rank_pattern.lm_head",
            peft.get("rank_pattern", {}).get("lm_head"),
            1,
        )
        require_equal(
            run_name,
            "adapter_config.alpha_pattern.lm_head",
            peft.get("alpha_pattern", {}).get("lm_head"),
            scale,
        )
        if "embed_tokens" in set(peft.get("target_modules", [])):
            raise ValueError(f"{run_name}: input embeddings were unexpectedly targeted")
    else:
        affine = load_json(run_dir / "affine_vocab_config.json")
        expected_affine = {
            "hidden_size": 1536,
            "rank": 50,
            "alpha": 50 * scale,
            "dropout": 0,
            "use_input": False,
            "use_lm_head": True,
            "use_lm_head_bias": False,
            "tie_input_lm_head_adapters": False,
        }
        for key, expected_value in expected_affine.items():
            require_equal(
                run_name,
                f"affine_vocab_config.{key}",
                affine.get(key),
                expected_value,
            )


def validate_train_log(
    run_name: str, experiment_dir: Path, method: str
) -> None:
    path = experiment_dir / "logs" / f"{run_name}.train.log"
    if not path.is_file():
        raise FileNotFoundError(f"Required training log is missing: {path}")
    contents = path.read_text(encoding="utf-8", errors="replace")
    boundary = METHODS[method]["expected_boundary"]
    if method == "vocab_r1":
        marker = (
            "[output_only_vocab] targets=lm_head boundary_dropout=0 "
            f"boundary_params={boundary}"
        )
    else:
        marker = f"re-enabled affine vocab trainable params: {boundary}"
    if marker not in contents:
        raise ValueError(f"{run_name}: training log lacks expected marker {marker!r}")


def validate_dev_report(
    run_name: str, run_dir: Path, report: dict[str, Any]
) -> tuple[list[str], list[int]]:
    require_equal(run_name, "dev_report.num_examples", report.get("num_examples"), 1000)
    require_equal(run_name, "dev_report.seed", report.get("seed"), 42)
    require_equal(
        run_name, "dev_report.source_start_index", report.get("source_start_index"), 0
    )
    require_equal(
        run_name, "dev_report.source_end_index", report.get("source_end_index"), 1000
    )
    require_equal(
        run_name,
        "dev_report.run_dir",
        resolve_recorded_path(report.get("run_dir")),
        run_dir.resolve(),
    )
    require_equal(
        run_name,
        "dev_report.data",
        resolve_recorded_path(report.get("data")),
        (ROOT / "corrected_sft_experiment/data/dev.jsonl").resolve(),
    )

    rows = report.get("per_example")
    if not isinstance(rows, list) or len(rows) != 1000:
        raise ValueError(f"{run_name}: dev_report.per_example must contain 1000 rows")
    ids = [row.get("record_id") for row in rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ValueError(f"{run_name}: missing or invalid dev record_id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{run_name}: duplicate dev record_id")
    token_counts = [row.get("token_count") for row in rows]
    if any(not isinstance(count, int) or count <= 0 for count in token_counts):
        raise ValueError(f"{run_name}: invalid supervised token count")
    nll_values = [row.get("nll_sum") for row in rows]
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in nll_values
    ):
        raise ValueError(f"{run_name}: invalid per-example NLL")

    total_tokens = sum(token_counts)
    total_nll = math.fsum(float(value) for value in nll_values)
    require_equal(
        run_name, "dev_report.supervised_tokens", report.get("supervised_tokens"), total_tokens
    )
    require_close(
        run_name,
        "dev_report.total_nll",
        report.get("total_nll"),
        total_nll,
        abs_tol=1e-5,
    )
    require_close(
        run_name,
        "dev_report.avg_ce",
        report.get("avg_ce"),
        total_nll / total_tokens,
        abs_tol=1e-10,
    )
    if not math.isfinite(float(report["avg_ce"])):
        raise ValueError(f"{run_name}: non-finite avg_ce")
    return ids, token_counts


def collect(experiment_dir: Path) -> dict[str, Any]:
    checkpoints = experiment_dir / "checkpoints"
    reference_ids: list[str] | None = None
    reference_counts: list[int] | None = None
    method_rows: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHODS
    }

    for method in METHODS:
        for scale in SCALES:
            run_name = f"qwen25_15b_out_{method}_s{scale}_sd42"
            run_dir = checkpoints / run_name
            run_args = load_json(run_dir / "run_args.json")
            params = load_json(run_dir / "trainable_summary.json")
            report_path = run_dir / "dev_report.json"
            report = load_json(report_path)

            validate_common_run_args(run_name, run_dir, run_args)
            validate_method_run_args(run_name, method, scale, run_args)
            validate_adapter_config(run_name, run_dir, method, scale)
            validate_train_log(run_name, experiment_dir, method)
            require_equal(
                run_name,
                "trainable_summary.trainable",
                params.get("trainable"),
                METHODS[method]["expected_trainable"],
            )
            ids, counts = validate_dev_report(run_name, run_dir, report)
            if reference_ids is None:
                reference_ids, reference_counts = ids, counts
            elif ids != reference_ids or counts != reference_counts:
                raise ValueError(
                    f"{run_name}: dev examples or supervised-token counts are not paired"
                )

            method_rows[method].append(
                {
                    "scale": scale,
                    "alpha": 50 * scale if method == "aff_r50" else scale,
                    "rank": 50 if method == "aff_r50" else 1,
                    "run_name": run_name,
                    "run_dir": str(run_dir.resolve()),
                    "dev_report": str(report_path.resolve()),
                    "dev_avg_ce": float(report["avg_ce"]),
                    "dev_perplexity": float(report["perplexity"]),
                    "dev_supervised_tokens": int(report["supervised_tokens"]),
                    "trainable_parameters": int(params["trainable"]),
                    "boundary_parameters": METHODS[method]["expected_boundary"],
                }
            )

    methods: dict[str, dict[str, Any]] = {}
    selected_scales: dict[str, int] = {}
    for method, rows in method_rows.items():
        rows.sort(key=lambda row: row["scale"])
        selected = min(rows, key=lambda row: (row["dev_avg_ce"], row["scale"]))
        methods[method] = {
            "label": METHODS[method]["label"],
            "runs": rows,
            "selected": selected,
        }
        selected_scales[method] = int(selected["scale"])

    aff_selected = methods["aff_r50"]["selected"]
    vocab_selected = methods["vocab_r1"]["selected"]
    return {
        "experiment": str(experiment_dir.resolve()),
        "selection_protocol": {
            "split": "corrected dev",
            "seed": 42,
            "candidate_scales": list(SCALES),
            "criterion": "minimum token-weighted dev cross-entropy per method",
            "tie_break": "smaller scale",
            "test_used_for_selection": False,
        },
        "validation": {
            "status": "passed",
            "runs_validated": 8,
            "paired_dev_examples": len(reference_ids or []),
            "paired_supervised_tokens": sum(reference_counts or []),
            "boundary_parameter_difference_aff_minus_vocab": (
                METHODS["aff_r50"]["expected_boundary"]
                - METHODS["vocab_r1"]["expected_boundary"]
            ),
        },
        "methods": methods,
        "selected_scales": selected_scales,
        "selected_dev_delta_ce_aff_minus_vocab": (
            aff_selected["dev_avg_ce"] - vocab_selected["dev_avg_ce"]
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Output-only equal-budget Qwen2.5 phase-1 summary",
        "",
        "Selection used only the corrected 1,000-example dev split at seed 42. "
        "All artifact, configuration, parameter-budget, and paired-example checks passed.",
        "",
        "| Method | Scale | Alpha | Rank | Trainable | Boundary | Dev CE | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for method in ("aff_r50", "vocab_r1"):
        block = summary["methods"][method]
        selected_scale = block["selected"]["scale"]
        for row in block["runs"]:
            lines.append(
                "| {label} | {scale} | {alpha} | {rank} | {trainable:,} | "
                "{boundary:,} | {ce:.9f} | {selected} |".format(
                    label=block["label"],
                    scale=row["scale"],
                    alpha=row["alpha"],
                    rank=row["rank"],
                    trainable=row["trainable_parameters"],
                    boundary=row["boundary_parameters"],
                    ce=row["dev_avg_ce"],
                    selected="yes" if row["scale"] == selected_scale else "",
                )
            )
    lines.extend(
        [
            "",
            "## Frozen phase-2 scales",
            "",
            f"- A-LoRA r50 scale: `{summary['selected_scales']['aff_r50']}`",
            f"- Vocab-LoRA r1 scale: `{summary['selected_scales']['vocab_r1']}`",
            (
                "- Selected dev ΔCE (A-LoRA − Vocab-LoRA): "
                f"`{summary['selected_dev_delta_ce_aff_minus_vocab']:+.9f}`"
            ),
            "",
            "Run phase 2 with:",
            "",
            "```bash",
            "bash reviewer_followup/run_output_only_equal_budget_qwen25_phase2.sh "
            f"{summary['selected_scales']['aff_r50']} "
            f"{summary['selected_scales']['vocab_r1']}",
            "```",
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
        else experiment_dir / "phase1_summary.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md
        else experiment_dir / "phase1_summary.md"
    )
    summary = collect(experiment_dir)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "selected_scales": summary["selected_scales"],
                "selected_dev_delta_ce_aff_minus_vocab": summary[
                    "selected_dev_delta_ce_aff_minus_vocab"
                ],
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
