#!/usr/bin/env python
"""Strict 14-run adaptive summary for the output-only equal-budget sweep.

Candidate set per method:

* original scale={1,2,4,8}, boundary-LR multiplier 1;
* scale=16, boundary-LR multiplier 1;
* scale=8, boundary-LR multiplier {0.5,2}.

Selection uses corrected dev only.  The script fails closed on missing or
inconsistent artifacts, tensor parameter counts, paired hidden initialization,
optimizer grouping, or per-example dev pairing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open

import summarize_output_only_equal_budget_qwen25_phase1 as phase1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "reviewer_followup/output_only_equal_budget_qwen25"
EXPECTED_HIDDEN_HASH = (
    "12de5ac2e5a9fa8032228014a6da12226c6132ea4b62799b4a5f8e1d26e9c469"
)
EXPECTED_HIDDEN_TENSORS = 392
EXPECTED_HIDDEN_PARAMS = 9_232_384
BASE_LR = 2e-4
CANDIDATES = (
    (1, 1.0),
    (2, 1.0),
    (4, 1.0),
    (8, 1.0),
    (16, 1.0),
    (8, 0.5),
    (8, 2.0),
)
EXPECTED_TOTAL = {
    "aff_r50": 1_553_100_288,
    "vocab_r1": 1_553_100_160,
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
        help="Defaults to <experiment-dir>/adaptive_summary.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/adaptive_summary.md.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(method: str, scale: int, boundary_lr_scale: float) -> str:
    stem = f"qwen25_15b_out_{method}_s{scale}"
    if boundary_lr_scale == 0.5:
        stem += "_blr0p5"
    elif boundary_lr_scale == 2.0:
        stem += "_blr2"
    elif boundary_lr_scale != 1.0:
        raise ValueError(
            f"Unsupported boundary LR multiplier {boundary_lr_scale}"
        )
    return f"{stem}_sd42"


def require_artifacts(
    name: str,
    run_dir: Path,
    method: str,
    boundary_lr_scale: float,
) -> None:
    required = [
        run_dir / "adapter_model.safetensors",
        run_dir / "adapter_config.json",
        run_dir / "run_args.json",
        run_dir / "trainable_summary.json",
        run_dir / "tokenizer.json",
        run_dir / "tokenizer_config.json",
        run_dir / "dev_report.json",
    ]
    if method == "aff_r50":
        required.extend(
            [
                run_dir / "affine_vocab_adapter.safetensors",
                run_dir / "affine_vocab_config.json",
            ]
        )
    elif boundary_lr_scale != 1.0:
        required.append(run_dir / "optimizer_groups.json")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"{name}: missing or empty artifacts: {missing}")

    forbidden = []
    if method == "aff_r50":
        forbidden.append(run_dir / "optimizer_groups.json")
    else:
        forbidden.extend(
            [
                run_dir / "affine_vocab_adapter.safetensors",
                run_dir / "affine_vocab_config.json",
            ]
        )
        if boundary_lr_scale == 1.0:
            forbidden.append(run_dir / "optimizer_groups.json")
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise ValueError(f"{name}: unexpected method-specific artifacts: {present}")


def safetensor_shapes(path: Path) -> dict[str, list[int]]:
    with safe_open(path, framework="pt") as tensors:
        return {
            key: list(tensors.get_slice(key).get_shape())
            for key in tensors.keys()
        }


def numel(shape: list[int]) -> int:
    return math.prod(shape)


def validate_tensor_artifacts(
    name: str, run_dir: Path, method: str
) -> dict[str, Any]:
    adapter_shapes = safetensor_shapes(run_dir / "adapter_model.safetensors")
    hidden = {
        key: shape
        for key, shape in adapter_shapes.items()
        if "lora_" in key
        and "lm_head" not in key
        and "embed_tokens" not in key
    }
    hidden_params = sum(numel(shape) for shape in hidden.values())
    phase1.require_equal(
        name, "hidden adapter tensor count", len(hidden), EXPECTED_HIDDEN_TENSORS
    )
    phase1.require_equal(
        name, "hidden adapter parameter count", hidden_params, EXPECTED_HIDDEN_PARAMS
    )

    input_lora = [
        key
        for key in adapter_shapes
        if "embed_tokens" in key and "lora_" in key
    ]
    if input_lora:
        raise ValueError(f"{name}: input embedding LoRA tensors found: {input_lora}")

    if method == "aff_r50":
        phase1.require_equal(
            name,
            "PEFT adapter tensor keys",
            set(adapter_shapes),
            set(hidden),
        )
        affine_shapes = safetensor_shapes(
            run_dir / "affine_vocab_adapter.safetensors"
        )
        expected_affine = {
            "lm_head.affine.down.weight": [50, 1536],
            "lm_head.affine.up.weight": [1536, 50],
        }
        phase1.require_equal(
            name, "affine adapter tensor shapes", affine_shapes, expected_affine
        )
        boundary_params = sum(numel(shape) for shape in affine_shapes.values())
    else:
        boundary = {
            key: shape
            for key, shape in adapter_shapes.items()
            if "lm_head" in key and "lora_" in key
        }
        expected_boundary_shapes = sorted(([1, 1536], [151936, 1]))
        phase1.require_equal(
            name,
            "output Vocab-LoRA tensor shapes",
            sorted(boundary.values()),
            expected_boundary_shapes,
        )
        base = {
            key: shape
            for key, shape in adapter_shapes.items()
            if "lora_" not in key
        }
        phase1.require_equal(
            name,
            "saved tied base tensor",
            base,
            {
                "base_model.model.lm_head.base_layer.weight": [151936, 1536]
            },
        )
        phase1.require_equal(
            name,
            "PEFT adapter tensor keys",
            set(adapter_shapes),
            set(hidden) | set(boundary) | set(base),
        )
        boundary_params = sum(numel(shape) for shape in boundary.values())

    phase1.require_equal(
        name,
        "boundary tensor parameter count",
        boundary_params,
        phase1.METHODS[method]["expected_boundary"],
    )
    return {
        "hidden_tensors": len(hidden),
        "hidden_parameters": hidden_params,
        "boundary_parameters": boundary_params,
        "adapter_tensor_count": len(adapter_shapes),
    }


def validate_trainable_summary(
    name: str, method: str, summary: dict[str, Any]
) -> None:
    expected_trainable = phase1.METHODS[method]["expected_trainable"]
    expected_total = EXPECTED_TOTAL[method]
    phase1.require_equal(
        name, "trainable_summary.trainable", summary.get("trainable"), expected_trainable
    )
    phase1.require_equal(
        name, "trainable_summary.total", summary.get("total"), expected_total
    )
    phase1.require_close(
        name,
        "trainable_summary.pct",
        summary.get("pct"),
        100.0 * expected_trainable / expected_total,
    )


def validate_pipeline_metadata(name: str, args: dict[str, Any]) -> str:
    pipeline = args.get("corrected_data_pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError(f"{name}: corrected_data_pipeline metadata is missing")
    implementation = (ROOT / "corrected_sft_experiment/train_corrected_sft.py").resolve()
    data_pipeline = ROOT / "corrected_sft_experiment/data_pipeline.py"
    expected = {
        "version": 1,
        "implementation": str(implementation),
        "implementation_sha256": sha256_file(data_pipeline),
        "chat_template": "tokenizer_native",
        "loss_mask": "assistant_content_plus_im_end",
        "initialization": "seed_before_model_build_and_isolated_afflora_rng",
    }
    for key, expected_value in expected.items():
        phase1.require_equal(
            name,
            f"run_args.corrected_data_pipeline.{key}",
            pipeline.get(key),
            expected_value,
        )
    return expected["implementation_sha256"]


def validate_hidden_hash_and_completion(
    name: str, experiment_dir: Path, method: str
) -> tuple[str, str]:
    path = experiment_dir / "logs" / f"{name}.train.log"
    if not path.is_file():
        raise FileNotFoundError(f"{name}: missing training log {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in text:
        raise ValueError(f"{name}: training log contains a traceback")

    if method == "aff_r50":
        pattern = re.compile(
            r"^\[repro\] seed=42 hidden_lora_init_sha256=([0-9a-f]{64})$",
            re.MULTILINE,
        )
        marker_kind = "hidden_lora_init_sha256"
    else:
        pattern = re.compile(
            r"^\[repro\] hidden_only_lora_init_sha256=([0-9a-f]{64}) "
            r"tensors=(\d+)$",
            re.MULTILINE,
        )
        marker_kind = "hidden_only_lora_init_sha256"
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"{name}: expected exactly one paired hidden hash marker, found {matches}"
        )
    if method == "aff_r50":
        hidden_hash = str(matches[0])
    else:
        hidden_hash, tensor_count = matches[0]
        phase1.require_equal(
            name,
            "hidden hash marker tensor count",
            int(tensor_count),
            EXPECTED_HIDDEN_TENSORS,
        )
    phase1.require_equal(
        name, "hidden LoRA initialization SHA256", hidden_hash, EXPECTED_HIDDEN_HASH
    )

    expected_trainable = phase1.METHODS[method]["expected_trainable"]
    completion_markers = (
        f"trainable params: {expected_trainable:,}",
        "[precision] cast_to_master_dtype=fp32",
        "[precision] post_train adam_state_dtypes=",
    )
    missing = [marker for marker in completion_markers if marker not in text]
    if missing:
        raise ValueError(
            f"{name}: training log lacks completion/precision markers {missing}"
        )
    return text, marker_kind


def validate_float_mapping(
    name: str,
    source: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key, expected_value in expected.items():
        value = actual.get(key)
        if isinstance(expected_value, float):
            phase1.require_close(name, f"{source}.{key}", value, expected_value)
        else:
            phase1.require_equal(name, f"{source}.{key}", value, expected_value)


def validate_optimizer(
    name: str,
    run_dir: Path,
    method: str,
    boundary_lr_scale: float,
    args: dict[str, Any],
    train_log: str,
) -> dict[str, Any]:
    phase1.require_close(
        name,
        "run_args.affine_learning_rate_scale",
        args.get("affine_learning_rate_scale"),
        boundary_lr_scale if method == "aff_r50" else 1.0,
    )
    phase1.require_close(
        name,
        "run_args.affine_bias_learning_rate_scale",
        args.get("affine_bias_learning_rate_scale"),
        1.0,
    )

    special_affine = "[optimizer] separate_affine_lr"
    special_vocab = "[optimizer] output_vocab_boundary_lr"
    if boundary_lr_scale == 1.0:
        if special_affine in train_log or special_vocab in train_log:
            raise ValueError(
                f"{name}: LR=1 run unexpectedly used a special boundary optimizer"
            )
        if args.get("output_vocab_lr") is not None:
            raise ValueError(f"{name}: LR=1 run has unexpected boundary-LR metadata")
        return {
            "mode": "shared_default_lr",
            "hidden_learning_rate": BASE_LR,
            "boundary_learning_rate": BASE_LR,
            "hidden_parameters": EXPECTED_HIDDEN_PARAMS,
            "boundary_parameters": phase1.METHODS[method]["expected_boundary"],
        }

    if method == "aff_r50":
        pattern = re.compile(
            r"^\[optimizer\] separate_affine_lr "
            r"base_lr=(\S+) affine_lr=(\S+) affine_scale=(\S+) "
            r"affine_bias_lr=(\S+) bias_scale=(\S+) counts=(\{.*\})$",
            re.MULTILINE,
        )
        matches = pattern.findall(train_log)
        if len(matches) != 1:
            raise ValueError(
                f"{name}: expected one separate-affine optimizer marker, found {matches}"
            )
        base_lr, boundary_lr, scale, bias_lr, bias_scale, counts_text = matches[0]
        for source, value, expected in (
            ("base_lr", base_lr, BASE_LR),
            ("affine_lr", boundary_lr, BASE_LR * boundary_lr_scale),
            ("affine_scale", scale, boundary_lr_scale),
            ("affine_bias_lr", bias_lr, BASE_LR * boundary_lr_scale),
            ("bias_scale", bias_scale, 1.0),
        ):
            phase1.require_close(name, f"optimizer.{source}", float(value), expected)
        counts = ast.literal_eval(counts_text)
        expected_counts = {
            "hidden_or_other": EXPECTED_HIDDEN_PARAMS,
            "affine_weight": 153_600,
            "affine_bias": 0,
        }
        phase1.require_equal(name, "optimizer counts", counts, expected_counts)
        return {
            "mode": "separate_affine_lr",
            "hidden_learning_rate": BASE_LR,
            "boundary_learning_rate": BASE_LR * boundary_lr_scale,
            "groups": counts,
        }

    marker_lines = [
        line.split(special_vocab, 1)[1].strip()
        for line in train_log.splitlines()
        if special_vocab in line
    ]
    if len(marker_lines) != 1:
        raise ValueError(
            f"{name}: expected one output-Vocab optimizer marker, found {marker_lines}"
        )
    log_audit = json.loads(marker_lines[0])
    file_audit = phase1.load_json(run_dir / "optimizer_groups.json")
    run_args_audit = args.get("output_vocab_lr")
    if not isinstance(run_args_audit, dict):
        raise ValueError(f"{name}: run_args.output_vocab_lr is missing")
    phase1.require_equal(
        name,
        "run_args.output_vocab_lr.implementation",
        Path(str(run_args_audit.get("implementation"))).resolve(),
        (
            ROOT
            / "reviewer_followup/train_corrected_sft_output_vocab_boundary_lr.py"
        ).resolve(),
    )

    expected = {
        "hidden_learning_rate": BASE_LR,
        "output_vocab_learning_rate": BASE_LR * boundary_lr_scale,
        "output_vocab_lr_scale": boundary_lr_scale,
        "hidden_lora_params": EXPECTED_HIDDEN_PARAMS,
        "output_vocab_lora_params": 153_472,
        "total_trainable_params": 9_385_856,
        "output_lora_rank": 1,
        "output_lora_a_shape": [1, 1536],
        "output_lora_b_shape": [151936, 1],
        "coverage_assertion": "passed",
    }
    validate_float_mapping(name, "optimizer log", log_audit, expected)
    validate_float_mapping(name, "optimizer_groups", file_audit, expected)
    validate_float_mapping(name, "run_args.output_vocab_lr", run_args_audit, expected)
    phase1.require_equal(name, "optimizer log/file audit", log_audit, file_audit)

    groups = file_audit.get("optimizer_group_params")
    if not isinstance(groups, dict):
        raise ValueError(f"{name}: optimizer_group_params is missing")
    hidden_group = sum(
        int(value)
        for key, value in groups.items()
        if key.startswith("hidden_lora_")
    )
    boundary_group = sum(
        int(value)
        for key, value in groups.items()
        if key.startswith("output_vocab_lora_")
    )
    phase1.require_equal(
        name, "optimizer hidden group parameters", hidden_group, EXPECTED_HIDDEN_PARAMS
    )
    phase1.require_equal(
        name, "optimizer boundary group parameters", boundary_group, 153_472
    )
    return {
        "mode": "separate_output_vocab_lr",
        "hidden_learning_rate": BASE_LR,
        "boundary_learning_rate": BASE_LR * boundary_lr_scale,
        "groups": groups,
    }


def validate_dev(
    name: str,
    run_dir: Path,
    args: dict[str, Any],
    report: dict[str, Any],
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    phase1.validate_dev_report(name, run_dir, report)
    phase1.require_equal(
        name, "dev_report.variant", report.get("variant"), args.get("variant")
    )
    phase1.require_equal(
        name, "dev_report.affine_ablation", report.get("affine_ablation"), "none"
    )
    phase1.require_equal(
        name,
        "dev_report.model_path",
        phase1.resolve_recorded_path(report.get("model_path")),
        (ROOT / "../models/Qwen2.5-1.5B-Base").resolve(),
    )

    signatures: list[tuple[Any, ...]] = []
    for index, row in enumerate(report["per_example"]):
        source_file = row.get("source_file")
        source_index = row.get("source_index")
        if not isinstance(source_file, str) or not source_file:
            raise ValueError(f"{name}: row {index} has invalid source_file")
        if not isinstance(source_index, int) or source_index < 0:
            raise ValueError(f"{name}: row {index} has invalid source_index")
        expected_mean = float(row["nll_sum"]) / int(row["token_count"])
        phase1.require_close(
            name,
            f"dev_report.per_example[{index}].mean_ce",
            row.get("mean_ce"),
            expected_mean,
            abs_tol=1e-10,
        )
        signatures.append(
            (
                row["record_id"],
                source_file,
                source_index,
                int(row["token_count"]),
            )
        )
    return signatures, {
        "dev_avg_ce": float(report["avg_ce"]),
        "dev_perplexity": float(report["perplexity"]),
        "dev_supervised_tokens": int(report["supervised_tokens"]),
    }


def select_and_check_boundary(
    rows_by_method: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    bool,
]:
    methods: dict[str, dict[str, Any]] = {}
    boundary_checks: dict[str, dict[str, Any]] = {}
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
        s8 = next(
            row
            for row in rows
            if row["scale"] == 8 and row["boundary_lr_scale"] == 1.0
        )
        s16 = next(
            row
            for row in rows
            if row["scale"] == 16 and row["boundary_lr_scale"] == 1.0
        )
        improving = bool(s16["dev_avg_ce"] < s8["dev_avg_ce"])
        boundary_checks[method] = {
            "scale8_blr1_dev_avg_ce": s8["dev_avg_ce"],
            "scale16_blr1_dev_avg_ce": s16["dev_avg_ce"],
            "scale16_minus_scale8_dev_ce": (
                s16["dev_avg_ce"] - s8["dev_avg_ce"]
            ),
            "scale16_less_than_scale8": improving,
            "needs_scale32_extension": improving,
        }
        methods[method] = {
            "label": phase1.METHODS[method]["label"],
            "runs": rows,
            "selected": selected,
        }
    needs_scale32 = any(
        check["needs_scale32_extension"] for check in boundary_checks.values()
    )
    return methods, boundary_checks, needs_scale32


def collect(experiment_dir: Path) -> dict[str, Any]:
    checkpoints = experiment_dir / "checkpoints"
    reference_signatures: list[tuple[Any, ...]] | None = None
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in phase1.METHODS
    }
    hash_marker_kinds: dict[str, str] = {}
    pipeline_hashes: set[str] = set()

    for method in phase1.METHODS:
        for scale, boundary_lr_scale in CANDIDATES:
            name = run_name(method, scale, boundary_lr_scale)
            run_dir = checkpoints / name
            require_artifacts(name, run_dir, method, boundary_lr_scale)
            args = phase1.load_json(run_dir / "run_args.json")
            params = phase1.load_json(run_dir / "trainable_summary.json")
            report_path = run_dir / "dev_report.json"
            report = phase1.load_json(report_path)

            phase1.validate_common_run_args(name, run_dir, args)
            phase1.validate_method_run_args(name, method, scale, args)
            phase1.validate_adapter_config(name, run_dir, method, scale)
            phase1.validate_train_log(name, experiment_dir, method)
            validate_trainable_summary(name, method, params)
            tensor_audit = validate_tensor_artifacts(name, run_dir, method)
            pipeline_hashes.add(validate_pipeline_metadata(name, args))
            train_log, marker_kind = validate_hidden_hash_and_completion(
                name, experiment_dir, method
            )
            hash_marker_kinds[method] = marker_kind
            optimizer_audit = validate_optimizer(
                name,
                run_dir,
                method,
                boundary_lr_scale,
                args,
                train_log,
            )
            signatures, dev = validate_dev(name, run_dir, args, report)
            if reference_signatures is None:
                reference_signatures = signatures
            elif signatures != reference_signatures:
                raise ValueError(
                    f"{name}: dev record/source/token signature is not paired"
                )

            rows_by_method[method].append(
                {
                    "run_name": name,
                    "run_dir": str(run_dir.resolve()),
                    "dev_report": str(report_path.resolve()),
                    "scale": scale,
                    "alpha": 50 * scale if method == "aff_r50" else scale,
                    "rank": 50 if method == "aff_r50" else 1,
                    "boundary_lr_scale": boundary_lr_scale,
                    "boundary_learning_rate": BASE_LR * boundary_lr_scale,
                    "trainable_parameters": int(params["trainable"]),
                    "boundary_parameters": phase1.METHODS[method][
                        "expected_boundary"
                    ],
                    **dev,
                    "tensor_audit": tensor_audit,
                    "optimizer_audit": optimizer_audit,
                }
            )

    phase1.require_equal(
        "all runs", "corrected pipeline hash count", len(pipeline_hashes), 1
    )
    methods, boundary_checks, needs_scale32 = select_and_check_boundary(
        rows_by_method
    )
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
                for scale, boundary_lr_scale in CANDIDATES
            ],
            "criterion": "minimum token-weighted dev cross-entropy per method",
            "tie_break": "smaller scale, then smaller boundary LR multiplier",
            "test_or_ifeval_used_for_selection": False,
            "scale32_rule": (
                "per method, scale16@blr1 dev CE < scale8@blr1 dev CE; "
                "overall flag is true if either method satisfies the rule"
            ),
        },
        "validation": {
            "status": "passed",
            "runs_validated": sum(len(rows) for rows in rows_by_method.values()),
            "paired_dev_examples": len(reference_signatures or []),
            "paired_supervised_tokens": sum(
                int(signature[3]) for signature in (reference_signatures or [])
            ),
            "hidden_lora_init_sha256": EXPECTED_HIDDEN_HASH,
            "hidden_hash_marker_by_method": hash_marker_kinds,
            "corrected_pipeline_sha256": next(iter(pipeline_hashes)),
            "safetensor_headers_and_parameter_counts": "passed",
            "optimizer_group_checks": "passed",
        },
        "methods": methods,
        "selected_hyperparameters": selected,
        "scale_boundary_checks": boundary_checks,
        "needs_scale32_extension": needs_scale32,
        "selected_dev_delta_ce_aff_minus_vocab": (
            selected["aff_r50"]["dev_avg_ce"]
            - selected["vocab_r1"]["dev_avg_ce"]
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    lines = [
        "# Output-only equal-budget adaptive summary",
        "",
        "Selection uses only the corrected 1,000-example dev split at seed 42.",
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
            "## Scale-boundary check",
            "",
            (
                f"`needs_scale32_extension="
                f"{str(summary['needs_scale32_extension']).lower()}`"
            ),
            "",
            "The check compares scale 16 against scale 8 at matched boundary-LR "
            "multiplier 1. The overall flag is true if either method still improves.",
            "",
            "| Method | Scale 8 CE | Scale 16 CE | ΔCE (16−8) | Scale16 < Scale8 | Needs scale32 |",
            "|---|---:|---:|---:|:---:|:---:|",
        ]
    )
    for method in ("aff_r50", "vocab_r1"):
        check = summary["scale_boundary_checks"][method]
        lines.append(
            "| {label} | {s8:.9f} | {s16:.9f} | {delta:+.9f} | "
            "{improves} | {needs} |".format(
                label=summary["methods"][method]["label"],
                s8=check["scale8_blr1_dev_avg_ce"],
                s16=check["scale16_blr1_dev_avg_ce"],
                delta=check["scale16_minus_scale8_dev_ce"],
                improves="yes" if check["scale16_less_than_scale8"] else "no",
                needs="yes" if check["needs_scale32_extension"] else "no",
            )
        )
    lines.extend(["", "## Selected hyperparameters", ""])
    for method in ("aff_r50", "vocab_r1"):
        selected = summary["selected_hyperparameters"][method]
        lines.append(
            f"- {summary['methods'][method]['label']}: "
            f"`scale={selected['scale']}`, "
            f"`boundary_lr_scale={selected['boundary_lr_scale']:g}`, "
            f"`dev_ce={selected['dev_avg_ce']:.9f}`"
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
        else experiment_dir / "adaptive_summary.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md
        else experiment_dir / "adaptive_summary.md"
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
                "needs_scale32_extension": summary[
                    "needs_scale32_extension"
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
