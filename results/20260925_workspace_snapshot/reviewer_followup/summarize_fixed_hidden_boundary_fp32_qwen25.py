#!/usr/bin/env python
"""Fail-closed CE summary for the fixed-hidden FP32 boundary control.

The primary comparison is output A-LoRA r50/scale16 versus direct output
Vocab-LoRA r1/scale32 on frozen hidden-LoRA checkpoints from seeds 42--44.
The corrected test split is primary; corrected dev is reported descriptively.

This program deliberately recomputes report arithmetic and paired-item
bootstraps from the per-example records.  It does not trust aggregate CE
fields alone.  The paired-t interval uses independently trained seeds as the
inferential units; item bootstraps condition on one trained seed and are kept
separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = (
    ROOT / "reviewer_followup/fixed_hidden_boundary_fp32_qwen25"
)
MODEL = (ROOT.parent / "models/Qwen2.5-1.5B-Base").resolve()
DATA = {
    split: (ROOT / f"corrected_sft_experiment/data/{split}.jsonl").resolve()
    for split in ("dev", "test")
}
SEEDS = (42, 43, 44)
SPLITS = ("dev", "test")
EXPECTED_EXAMPLES = 1_000
BOOTSTRAP_SAMPLES = 10_000
T_CRITICAL_975_DF2 = 4.302652729696142
HEX_DIGITS = frozenset("0123456789abcdef")

METHODS: dict[str, dict[str, Any]] = {
    "alora": {
        "label": "output A-LoRA r50 s16",
        "kind": "alora",
        "rank": 50,
        "scale": 16,
        "alpha": 800.0,
        "trainable_parameters": 153_600,
    },
    "vocab_lora": {
        "label": "output Vocab-LoRA r1 s32",
        "kind": "vocab_lora",
        "rank": 1,
        "scale": 32,
        "alpha": 32.0,
        "trainable_parameters": 153_472,
    },
}


@dataclass(frozen=True)
class Report:
    path: Path
    payload: dict[str, Any]
    record_ids: tuple[str, ...]
    nll: np.ndarray
    token_counts: np.ndarray

    @property
    def avg_ce(self) -> float:
        return float(self.payload["avg_ce"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="Fixed-hidden FP32 experiment directory.",
    )
    parser.add_argument(
        "--completion-manifest",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/completion_manifest.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ce_summary.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ce_summary.md.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic arithmetic/order/token fail-closed tests and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}={actual!r}, expected {expected!r}")


def require_close(
    label: str,
    actual: Any,
    expected: float,
    *,
    abs_tol: float = 1e-10,
) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=abs_tol
        )
    ):
        raise ValueError(f"{label}={actual!r}, expected {expected!r}")


def require_sha256(label: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return value


def resolve_recorded_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected a non-empty recorded path, got {value!r}")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def run_name(method: str, seed: int) -> str:
    if method == "alora":
        suffix = "alora_r50_s16"
    elif method == "vocab_lora":
        suffix = "vocab_lora_r1_s32"
    else:
        raise ValueError(f"Unsupported method: {method}")
    return f"qwen25_15b_fhfp32_hsd{seed}_{suffix}_bsd{seed}"


def baseline_dir(seed: int) -> Path:
    return (
        ROOT
        / f"corrected_sft_experiment/outputs/formal/qwen25_15b_hidden_sd{seed}"
    ).resolve()


def expected_run_names() -> set[str]:
    return {run_name(method, seed) for seed in SEEDS for method in METHODS}


def validate_completion_payload_structure(
    completion: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    require_equal(
        "completion.marker",
        completion.get("marker"),
        "fixed_hidden_boundary_fp32_qwen25_complete_v1",
    )
    require_equal(
        "completion.pair_identity_assertion",
        completion.get("pair_identity_assertion"),
        "passed",
    )
    require_equal(
        "completion.all_artifact_validation_assertion",
        completion.get("all_artifact_validation_assertion"),
        "passed",
    )
    require_sha256(
        "completion.training_entry_sha256",
        completion.get("training_entry_sha256"),
    )
    require_sha256(
        "completion.evaluator_sha256", completion.get("evaluator_sha256")
    )
    runs = completion.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("completion.runs must be an object")
    require_equal("completion run names", set(runs), expected_run_names())

    for seed in SEEDS:
        pair_hidden: set[str] = set()
        pair_frozen: set[str] = set()
        for method, metadata in METHODS.items():
            name = run_name(method, seed)
            item = runs[name]
            if not isinstance(item, dict):
                raise ValueError(f"completion.runs.{name} must be an object")
            require_equal(f"{name}.seed", item.get("seed"), seed)
            require_equal(f"{name}.kind", item.get("kind"), metadata["kind"])
            pair_hidden.add(
                require_sha256(
                    f"{name}.source_hidden_canonical_sha256",
                    item.get("source_hidden_canonical_sha256"),
                )
            )
            pair_frozen.add(
                require_sha256(
                    f"{name}.frozen_native_weight_sha256",
                    item.get("frozen_native_weight_sha256"),
                )
            )
            require_sha256(
                f"{name}.boundary_file_sha256",
                item.get("boundary_file_sha256"),
            )
            for split in SPLITS:
                require_sha256(
                    f"{name}.{split}_report_sha256",
                    item.get(f"{split}_report_sha256"),
                )
                value = item.get(f"{split}_avg_ce")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise ValueError(
                        f"{name}.{split}_avg_ce is invalid: {value!r}"
                    )
        require_equal(f"seed{seed} paired hidden hashes", len(pair_hidden), 1)
        require_equal(f"seed{seed} paired frozen weight hashes", len(pair_frozen), 1)
    return runs


def validate_manifests(
    experiment_dir: Path, completion_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    completion = load_json(completion_path)
    runs = validate_completion_payload_structure(completion)
    launch_path = experiment_dir / "launch_manifest.json"
    launch = load_json(launch_path)
    require_equal(
        "launch.marker",
        launch.get("marker"),
        "fixed_hidden_boundary_fp32_qwen25_launch_v1",
    )
    require_equal(
        "launch.fixed_endpoints",
        launch.get("fixed_endpoints"),
        {
            "alora": {"rank": 50, "alpha": 800, "scale": 16},
            "vocab_lora": {"rank": 1, "alpha": 32, "scale": 32},
        },
    )
    require_equal(
        "launch.endpoint_selection",
        launch.get("endpoint_selection"),
        "none_in_this_experiment",
    )
    require_equal(
        "training implementation digest across manifests",
        completion.get("training_entry_sha256"),
        launch.get("training_entry_sha256"),
    )
    require_equal(
        "evaluator digest across manifests",
        completion.get("evaluator_sha256"),
        launch.get("evaluator_sha256"),
    )

    for label, path_key, sha_key in (
        ("training entry", "training_entry", "training_entry_sha256"),
        ("evaluator", "evaluator", "evaluator_sha256"),
    ):
        path = resolve_recorded_path(launch.get(path_key))
        require_equal(
            f"{label} current SHA-256",
            sha256_file(path),
            launch.get(sha_key),
        )

    data_digests = launch.get("data_sha256")
    if not isinstance(data_digests, dict):
        raise ValueError("launch.data_sha256 must be an object")
    for split in SPLITS:
        require_equal(
            f"launch {split} data SHA-256",
            sha256_file(DATA[split]),
            data_digests.get(split),
        )
    train_path = (ROOT / "corrected_sft_experiment/data/train.jsonl").resolve()
    require_equal(
        "launch train data SHA-256",
        sha256_file(train_path),
        data_digests.get("train"),
    )
    require_equal(
        "launch model config SHA-256",
        sha256_file(MODEL / "config.json"),
        launch.get("model_config_sha256"),
    )
    return completion, runs, launch


def read_source_rows(path: Path, expected_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    require_equal(f"{path} source row count", len(rows), expected_examples)
    ids = [row.get("record_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{path}: every source row must have a record_id")
    require_equal(f"{path} unique source IDs", len(set(ids)), expected_examples)
    return rows


def validate_report(
    path: Path,
    *,
    run_dir: Path,
    data_path: Path,
    model_path: Path,
    source_rows: list[dict[str, Any]],
    seed: int,
    variant: str,
    fixed_boundary: bool,
) -> Report:
    payload = load_json(path)
    expected_examples = len(source_rows)
    for key, expected in {
        "num_examples": expected_examples,
        "seed": seed,
        "variant": variant,
    }.items():
        require_equal(f"{path} {key}", payload.get(key), expected)
    require_equal(
        f"{path} run_dir",
        resolve_recorded_path(payload.get("run_dir")),
        run_dir.resolve(),
    )
    require_equal(
        f"{path} data",
        resolve_recorded_path(payload.get("data")),
        data_path.resolve(),
    )
    require_equal(
        f"{path} model_path",
        resolve_recorded_path(payload.get("model_path")),
        model_path.resolve(),
    )
    if fixed_boundary:
        require_equal(f"{path} source_start_index", payload.get("source_start_index"), 0)
        require_equal(
            f"{path} source_end_index",
            payload.get("source_end_index"),
            expected_examples,
        )
        require_equal(f"{path} affine_ablation", payload.get("affine_ablation"), "none")
    else:
        if "source_start_index" in payload:
            require_equal(f"{path} source_start_index", payload["source_start_index"], 0)
        if "source_end_index" in payload:
            require_equal(
                f"{path} source_end_index",
                payload["source_end_index"],
                expected_examples,
            )

    rows = payload.get("per_example")
    if not isinstance(rows, list) or len(rows) != expected_examples:
        raise ValueError(
            f"{path}: expected {expected_examples} ordered per-example rows"
        )
    record_ids: list[str] = []
    nll_values: list[float] = []
    token_counts: list[int] = []
    for index, (row, source) in enumerate(zip(rows, source_rows)):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: per_example[{index}] is not an object")
        record_id = row.get("record_id")
        require_equal(
            f"{path} record_id at source row {index}",
            record_id,
            source.get("record_id"),
        )
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}: invalid record_id at row {index}")
        if record_id in record_ids:
            raise ValueError(f"{path}: duplicate record_id {record_id}")
        for source_key in ("source_file", "source_index"):
            if source_key in source:
                require_equal(
                    f"{path} {source_key} for {record_id}",
                    row.get(source_key),
                    source.get(source_key),
                )
        count = row.get("token_count")
        nll = row.get("nll_sum")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{path}: invalid token_count for {record_id}")
        if (
            isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(float(nll))
            or float(nll) < 0
        ):
            raise ValueError(f"{path}: invalid nll_sum for {record_id}")
        require_close(
            f"{path} mean_ce for {record_id}",
            row.get("mean_ce"),
            float(nll) / count,
        )
        record_ids.append(record_id)
        nll_values.append(float(nll))
        token_counts.append(count)

    total_nll = math.fsum(nll_values)
    total_tokens = sum(token_counts)
    avg_ce = total_nll / total_tokens
    require_equal(
        f"{path} supervised_tokens",
        payload.get("supervised_tokens"),
        total_tokens,
    )
    require_close(
        f"{path} total_nll",
        payload.get("total_nll"),
        total_nll,
        abs_tol=1e-5,
    )
    require_close(f"{path} avg_ce", payload.get("avg_ce"), avg_ce)
    require_close(
        f"{path} perplexity",
        payload.get("perplexity"),
        math.exp(avg_ce),
    )
    return Report(
        path=path.resolve(),
        payload=payload,
        record_ids=tuple(record_ids),
        nll=np.asarray(nll_values, dtype=np.float64),
        token_counts=np.asarray(token_counts, dtype=np.int64),
    )


def assert_paired_order_and_tokens(
    reports: Iterable[Report], *, label: str
) -> None:
    reports = list(reports)
    if not reports:
        raise ValueError(f"{label}: no reports supplied for pairing")
    reference = reports[0]
    for report in reports[1:]:
        if report.record_ids != reference.record_ids:
            raise ValueError(
                f"{label}: record-ID order differs between "
                f"{reference.path} and {report.path}"
            )
        if not np.array_equal(report.token_counts, reference.token_counts):
            mismatch = int(
                np.flatnonzero(report.token_counts != reference.token_counts)[0]
            )
            raise ValueError(
                f"{label}: supervised token count differs at row {mismatch} "
                f"between {reference.path} and {report.path}"
            )


def paired_item_bootstrap(
    alora: Report,
    vocab: Report,
    *,
    rng_seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    assert_paired_order_and_tokens([alora, vocab], label="paired bootstrap")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    counts = alora.token_counts.astype(np.float64)
    observed = float((alora.nll.sum() - vocab.nll.sum()) / counts.sum())
    rng = np.random.default_rng(rng_seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(counts), size=len(counts))
        deltas[index] = float(
            (alora.nll[selected].sum() - vocab.nll[selected].sum())
            / counts[selected].sum()
        )
    return {
        "bootstrap_samples": samples,
        "rng_seed": rng_seed,
        "delta_ce_alora_minus_vocab": observed,
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "probability_alora_better": float(np.mean(deltas < 0)),
        "inferential_unit": (
            "paired evaluation item conditional on this trained seed"
        ),
    }


def direction(delta: float) -> str:
    return "alora_better" if delta < 0 else "vocab_better" if delta > 0 else "tie"


def per_seed_result(
    seed: int,
    hidden: Report,
    alora: Report,
    vocab: Report,
    *,
    split: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    assert_paired_order_and_tokens(
        [hidden, alora, vocab], label=f"{split}/seed{seed}"
    )
    delta = alora.avg_ce - vocab.avg_ce
    bootstrap = paired_item_bootstrap(
        alora, vocab, rng_seed=bootstrap_seed
    )
    require_close(
        f"{split}/seed{seed} aggregate-vs-item delta",
        bootstrap["delta_ce_alora_minus_vocab"],
        delta,
    )
    alora_improvement = hidden.avg_ce - alora.avg_ce
    vocab_improvement = hidden.avg_ce - vocab.avg_ce
    return {
        "seed": seed,
        "hidden_zero_boundary_ce": hidden.avg_ce,
        "alora_ce": alora.avg_ce,
        "vocab_ce": vocab.avg_ce,
        "delta_ce_alora_minus_vocab": delta,
        "direction": direction(delta),
        "alora_improvement_vs_hidden": alora_improvement,
        "vocab_improvement_vs_hidden": vocab_improvement,
        "alora_relative_ce_reduction_vs_hidden_percent": (
            100.0 * alora_improvement / hidden.avg_ce
        ),
        "vocab_relative_ce_reduction_vs_hidden_percent": (
            100.0 * vocab_improvement / hidden.avg_ce
        ),
        "paired_item_bootstrap": bootstrap,
    }


def seed_level_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require_equal("seed-level row order", [row["seed"] for row in rows], list(SEEDS))
    metric_names = (
        "hidden_zero_boundary_ce",
        "alora_ce",
        "vocab_ce",
        "delta_ce_alora_minus_vocab",
        "alora_improvement_vs_hidden",
        "vocab_improvement_vs_hidden",
        "alora_relative_ce_reduction_vs_hidden_percent",
        "vocab_relative_ce_reduction_vs_hidden_percent",
    )
    result: dict[str, Any] = {"n_seeds": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        result[f"{name}_mean"] = statistics.fmean(values)
        result[f"{name}_sample_sd"] = statistics.stdev(values)
    delta_mean = result["delta_ce_alora_minus_vocab_mean"]
    delta_sd = result["delta_ce_alora_minus_vocab_sample_sd"]
    margin = T_CRITICAL_975_DF2 * delta_sd / math.sqrt(len(rows))
    ci = [delta_mean - margin, delta_mean + margin]
    result.update(
        {
            "paired_t_df": 2,
            "paired_t_critical_975": T_CRITICAL_975_DF2,
            "paired_t_ci95": ci,
            "paired_t_ci95_contains_zero": ci[0] <= 0 <= ci[1],
            "directions": {
                "alora_better": sum(
                    row["direction"] == "alora_better" for row in rows
                ),
                "vocab_better": sum(
                    row["direction"] == "vocab_better" for row in rows
                ),
                "ties": sum(row["direction"] == "tie" for row in rows),
            },
            "inferential_unit": "independently trained hidden/boundary seed pair",
        }
    )
    return result


def validate_main_run_artifacts(
    run_dir: Path,
    method: str,
    seed: int,
    manifest_item: dict[str, Any],
) -> None:
    metadata = METHODS[method]
    config = load_json(run_dir / "fixed_boundary_config.json")
    audit = load_json(run_dir / "fixed_hidden_boundary_audit.json")
    for source, payload in (("config", config), ("audit", audit)):
        require_equal(
            f"{run_dir.name} {source}.marker",
            payload.get("marker"),
            "fixed_hidden_output_boundary_fp32_v1",
        )
        require_equal(
            f"{run_dir.name} {source}.kind",
            payload.get("kind"),
            metadata["kind"],
        )
    for key, expected in {
        "rank": metadata["rank"],
        "hidden_size": 1536,
        "vocab_size": 151936,
        "trainable_parameters": metadata["trainable_parameters"],
    }.items():
        require_equal(f"{run_dir.name} config.{key}", config.get(key), expected)
    require_close(
        f"{run_dir.name} config.alpha", config.get("alpha"), metadata["alpha"]
    )
    require_close(
        f"{run_dir.name} config.scale", config.get("scale"), metadata["scale"]
    )
    require_equal(
        f"{run_dir.name} completion audit",
        audit.get("completion_audit_assertion"),
        "passed",
    )
    require_equal(
        f"{run_dir.name} hidden digest",
        audit.get("saved_hidden_canonical_tensor_sha256"),
        manifest_item["source_hidden_canonical_sha256"],
    )
    require_equal(
        f"{run_dir.name} frozen weight digest",
        config.get("frozen_native_weight_sha256"),
        manifest_item["frozen_native_weight_sha256"],
    )
    boundary_path = run_dir / "fixed_boundary_adapter.safetensors"
    require_equal(
        f"{run_dir.name} boundary file digest",
        sha256_file(boundary_path),
        manifest_item["boundary_file_sha256"],
    )
    require_equal(
        f"{run_dir.name} audit boundary file digest",
        audit.get("saved_boundary_file_sha256"),
        manifest_item["boundary_file_sha256"],
    )


def sensitivity_run_name(method: str) -> str:
    if method == "vocab_lora":
        suffix = "vocab_lora_r1_s16"
    elif method == "alora":
        suffix = "alora_r50_s32"
    else:
        raise ValueError(f"Unsupported sensitivity method: {method}")
    return f"qwen25_15b_fhfp32_hsd42_{suffix}_bsd42"


def validate_sensitivity_config(run_dir: Path, method: str) -> None:
    expected = (
        {"kind": "vocab_lora", "rank": 1, "scale": 16.0, "alpha": 16.0}
        if method == "vocab_lora"
        else {"kind": "alora", "rank": 50, "scale": 32.0, "alpha": 1600.0}
    )
    config = load_json(run_dir / "fixed_boundary_config.json")
    audit = load_json(run_dir / "fixed_hidden_boundary_audit.json")
    for source, payload in (("config", config), ("audit", audit)):
        require_equal(
            f"{run_dir.name} {source}.marker",
            payload.get("marker"),
            "fixed_hidden_output_boundary_fp32_v1",
        )
        require_equal(
            f"{run_dir.name} {source}.kind",
            payload.get("kind"),
            expected["kind"],
        )
    require_equal(f"{run_dir.name} rank", config.get("rank"), expected["rank"])
    require_close(f"{run_dir.name} scale", config.get("scale"), expected["scale"])
    require_close(f"{run_dir.name} alpha", config.get("alpha"), expected["alpha"])
    require_equal(
        f"{run_dir.name} completion audit",
        audit.get("completion_audit_assertion"),
        "passed",
    )


def collect_sensitivity(
    experiment_dir: Path,
    source_rows: dict[str, list[dict[str, Any]]],
    main_reports: dict[tuple[str, int, str], Report],
    baseline_reports: dict[tuple[int, str], Report],
) -> dict[str, Any]:
    checkpoints = experiment_dir / "checkpoints"
    required = {
        (method, split): checkpoints
        / sensitivity_run_name(method)
        / f"{split}_report.json"
        for method in METHODS
        for split in SPLITS
    }
    missing = [str(path.resolve()) for path in required.values() if not path.is_file()]
    present_reports: dict[tuple[str, str], Report] = {}
    for (method, split), path in required.items():
        if not path.is_file():
            continue
        run_dir = path.parent
        validate_sensitivity_config(run_dir, method)
        present_reports[(method, split)] = validate_report(
            path,
            run_dir=run_dir,
            data_path=DATA[split],
            model_path=MODEL,
            source_rows=source_rows[split],
            seed=42,
            variant="affine_lm_head_plus_hidden_lora",
            fixed_boundary=True,
        )
    if missing:
        return {
            "status": "pending",
            "role": (
                "seed-42 common-scale diagnostic only; not part of the "
                "three-seed primary comparison"
            ),
            "missing_reports": missing,
            "present_reports_validated": [
                str(report.path) for report in present_reports.values()
            ],
        }

    results: dict[str, Any] = {}
    for split_index, split in enumerate(SPLITS):
        hidden = baseline_reports[(42, split)]
        alora_s16 = main_reports[("alora", 42, split)]
        vocab_s16 = present_reports[("vocab_lora", split)]
        alora_s32 = present_reports[("alora", split)]
        vocab_s32 = main_reports[("vocab_lora", 42, split)]
        assert_paired_order_and_tokens(
            [hidden, alora_s16, vocab_s16, alora_s32, vocab_s32],
            label=f"sensitivity/{split}",
        )
        rows = []
        for scale, alora, vocab in (
            (16, alora_s16, vocab_s16),
            (32, alora_s32, vocab_s32),
        ):
            row = per_seed_result(
                42,
                hidden,
                alora,
                vocab,
                split=split,
                bootstrap_seed=300_000 + 1_000 * split_index + scale,
            )
            row["common_scale"] = scale
            rows.append(row)
        results[split] = rows
    return {
        "status": "complete",
        "role": (
            "seed-42 common-scale diagnostic only; scales were not a "
            "three-seed randomized factor"
        ),
        "comparisons": results,
        "reports_validated": len(present_reports),
    }


def collect(experiment_dir: Path, completion_path: Path) -> dict[str, Any]:
    completion, manifest_runs, launch = validate_manifests(
        experiment_dir, completion_path
    )
    source_rows = {
        split: read_source_rows(DATA[split], EXPECTED_EXAMPLES)
        for split in SPLITS
    }
    main_reports: dict[tuple[str, int, str], Report] = {}
    baseline_reports: dict[tuple[int, str], Report] = {}
    checkpoints = experiment_dir / "checkpoints"

    for seed in SEEDS:
        base_dir = baseline_dir(seed)
        for split in SPLITS:
            baseline_reports[(seed, split)] = validate_report(
                base_dir / f"{split}_report.json",
                run_dir=base_dir,
                data_path=DATA[split],
                model_path=MODEL,
                source_rows=source_rows[split],
                seed=seed,
                variant="hidden_lora",
                fixed_boundary=False,
            )
        for method in METHODS:
            name = run_name(method, seed)
            run_dir = checkpoints / name
            manifest_item = manifest_runs[name]
            validate_main_run_artifacts(run_dir, method, seed, manifest_item)
            for split in SPLITS:
                path = run_dir / f"{split}_report.json"
                report = validate_report(
                    path,
                    run_dir=run_dir,
                    data_path=DATA[split],
                    model_path=MODEL,
                    source_rows=source_rows[split],
                    seed=seed,
                    variant="affine_lm_head_plus_hidden_lora",
                    fixed_boundary=True,
                )
                require_equal(
                    f"{name} {split} report digest",
                    sha256_file(path),
                    manifest_item[f"{split}_report_sha256"],
                )
                require_close(
                    f"{name} {split} manifest CE",
                    report.avg_ce,
                    manifest_item[f"{split}_avg_ce"],
                )
                main_reports[(method, seed, split)] = report

    for split in SPLITS:
        assert_paired_order_and_tokens(
            [
                report
                for seed in SEEDS
                for report in (
                    baseline_reports[(seed, split)],
                    main_reports[("alora", seed, split)],
                    main_reports[("vocab_lora", seed, split)],
                )
            ],
            label=f"all primary {split} reports",
        )

    split_results: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(SPLITS):
        rows = [
            per_seed_result(
                seed,
                baseline_reports[(seed, split)],
                main_reports[("alora", seed, split)],
                main_reports[("vocab_lora", seed, split)],
                split=split,
                bootstrap_seed=100_000 + 10_000 * split_index + seed,
            )
            for seed in SEEDS
        ]
        split_results[split] = {
            "role": (
                "primary fixed-endpoint result"
                if split == "test"
                else "descriptive fixed-endpoint development result"
            ),
            "per_seed": rows,
            "seed_level_summary": seed_level_summary(rows),
        }

    sensitivity = collect_sensitivity(
        experiment_dir,
        source_rows,
        main_reports,
        baseline_reports,
    )
    return {
        "marker": "fixed_hidden_boundary_fp32_qwen25_ce_summary_v1",
        "experiment": str(experiment_dir.resolve()),
        "completion_manifest": str(completion_path.resolve()),
        "completion_manifest_sha256": sha256_file(completion_path),
        "estimand": {
            "metric": "token-weighted assistant-only cross-entropy",
            "primary_split": "corrected test",
            "difference": "A-LoRA CE minus Vocab-LoRA CE",
            "better_direction": "negative",
            "hidden_baseline_improvement": (
                "hidden-only zero-boundary CE minus method CE; positive is better"
            ),
            "primary_inferential_unit": (
                "independently trained frozen-hidden/boundary seed pair"
            ),
            "paired_item_bootstrap_unit": (
                "evaluation item conditional on one trained seed; it is not "
                "a substitute for seed-level uncertainty"
            ),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "fixed_endpoints": launch["fixed_endpoints"],
            "endpoint_selection": launch["endpoint_selection"],
            "bootstrap_samples_per_seed_split": BOOTSTRAP_SAMPLES,
        },
        "validation": {
            "status": "passed",
            "completion_manifest_validated": True,
            "main_fixed_boundary_reports_validated": 12,
            "formal_hidden_baseline_reports_validated": 6,
            "examples_per_report": EXPECTED_EXAMPLES,
            "record_id_order_assertion": "passed",
            "supervised_token_pairing_assertion": "passed",
            "report_arithmetic_recomputed": True,
            "paired_item_bootstraps_recomputed": 6,
        },
        "primary_test": split_results["test"],
        "development_descriptive": split_results["dev"],
        "sensitivity_common_scale_seed42": sensitivity,
    }


def fmt(value: float) -> str:
    return f"{value:.9f}"


def fmt_improvement(absolute: float, relative_percent: float) -> str:
    return f"{absolute:+.9f} ({relative_percent:+.3f}%)"


def render_split(title: str, result: dict[str, Any]) -> list[str]:
    rows = result["per_seed"]
    stats = result["seed_level_summary"]
    lines = [
        f"## {title}",
        "",
        "| Seed | Hidden zero-boundary CE | A-LoRA CE | Vocab CE | A−V | "
        "Hidden→A improvement | Hidden→V improvement | Item-bootstrap 95% CI | "
        "P(A better) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        bootstrap = row["paired_item_bootstrap"]
        ci = bootstrap["ci95"]
        lines.append(
            f"| {row['seed']} | {fmt(row['hidden_zero_boundary_ce'])} | "
            f"{fmt(row['alora_ce'])} | {fmt(row['vocab_ce'])} | "
            f"{row['delta_ce_alora_minus_vocab']:+.9f} | "
            f"{fmt_improvement(row['alora_improvement_vs_hidden'], row['alora_relative_ce_reduction_vs_hidden_percent'])} | "
            f"{fmt_improvement(row['vocab_improvement_vs_hidden'], row['vocab_relative_ce_reduction_vs_hidden_percent'])} | "
            f"[{ci[0]:+.9f}, {ci[1]:+.9f}] | "
            f"{bootstrap['probability_alora_better']:.4f} |"
        )
    ci = stats["paired_t_ci95"]
    directions = stats["directions"]
    lines.extend(
        [
            "",
            f"- Mean hidden CE: `{fmt(stats['hidden_zero_boundary_ce_mean'])}` "
            f"(sample SD `{fmt(stats['hidden_zero_boundary_ce_sample_sd'])}`).",
            f"- Mean A-LoRA CE: `{fmt(stats['alora_ce_mean'])}` "
            f"(sample SD `{fmt(stats['alora_ce_sample_sd'])}`).",
            f"- Mean Vocab-LoRA CE: `{fmt(stats['vocab_ce_mean'])}` "
            f"(sample SD `{fmt(stats['vocab_ce_sample_sd'])}`).",
            f"- Mean A−V: `{stats['delta_ce_alora_minus_vocab_mean']:+.9f}` "
            f"(sample SD `{fmt(stats['delta_ce_alora_minus_vocab_sample_sd'])}`).",
            f"- Seed-level paired-t 95% CI (df=2): "
            f"`[{ci[0]:+.9f}, {ci[1]:+.9f}]`.",
            f"- Directions: A-LoRA `{directions['alora_better']}/3`; "
            f"Vocab-LoRA `{directions['vocab_better']}/3`; "
            f"ties `{directions['ties']}/3`.",
            f"- Mean CE reduction versus hidden: A-LoRA "
            f"`{stats['alora_improvement_vs_hidden_mean']:+.9f}`; Vocab-LoRA "
            f"`{stats['vocab_improvement_vs_hidden_mean']:+.9f}`.",
            "",
        ]
    )
    return lines


def render_sensitivity(value: dict[str, Any]) -> list[str]:
    lines = ["## Seed-42 common-scale sensitivity", ""]
    if value["status"] == "pending":
        lines.extend(
            [
                "Status: **pending**. The primary three-seed summary is complete, "
                "but the following optional sensitivity reports are absent:",
                "",
            ]
        )
        lines.extend(f"- `{path}`" for path in value["missing_reports"])
        lines.append("")
        return lines
    lines.extend(
        [
            "This is a diagnostic only; it is not part of the primary "
            "three-seed inference.",
            "",
        ]
    )
    for split in ("test", "dev"):
        lines.extend(
            [
                f"### {split}",
                "",
                "| Common scale | A-LoRA CE | Vocab CE | A−V | Direction |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in value["comparisons"][split]:
            lines.append(
                f"| {row['common_scale']} | {fmt(row['alora_ce'])} | "
                f"{fmt(row['vocab_ce'])} | "
                f"{row['delta_ce_alora_minus_vocab']:+.9f} | "
                f"{row['direction']} |"
            )
        lines.append("")
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fixed-hidden FP32 boundary CE summary",
        "",
        "The sign convention is **A-LoRA CE − Vocab-LoRA CE**; negative "
        "values favor A-LoRA. Hidden-baseline improvement is **hidden-only "
        "zero-boundary CE − method CE**; positive values are improvements.",
        "",
    ]
    lines.extend(render_split("Primary corrected test (seeds 42–44)", summary["primary_test"]))
    lines.extend(
        render_split(
            "Corrected dev, descriptive (seeds 42–44)",
            summary["development_descriptive"],
        )
    )
    lines.extend(
        [
            "## Uncertainty interpretation",
            "",
            "The paired-t interval treats the three independently trained "
            "seed pairs as the inferential units. Each 10,000-sample item "
            "bootstrap instead conditions on one already-trained seed and "
            "measures evaluation-set sampling uncertainty. Narrow item "
            "intervals do not establish robustness across training seeds.",
            "",
        ]
    )
    lines.extend(render_sensitivity(summary["sensitivity_common_scale_seed42"]))
    validation = summary["validation"]
    lines.extend(
        [
            "## Validation",
            "",
            f"Validated `{validation['main_fixed_boundary_reports_validated']}` "
            "fixed-boundary reports and "
            f"`{validation['formal_hidden_baseline_reports_validated']}` formal "
            "hidden-only reports. Completion-manifest digests, report "
            "arithmetic, source record-ID order, and supervised-token pairing "
            "all passed.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def synthetic_report_payload(
    run_dir: Path,
    data_path: Path,
    model_path: Path,
    *,
    nll_offset: float,
    token_offset_at: int | None = None,
) -> dict[str, Any]:
    source_rows = read_source_rows(data_path, 4)
    rows = []
    for index, source in enumerate(source_rows):
        count = 2 + index + (1 if token_offset_at == index else 0)
        nll = float(count) * (0.5 + nll_offset + 0.01 * index)
        rows.append(
            {
                "record_id": source["record_id"],
                "source_file": source["source_file"],
                "source_index": source["source_index"],
                "nll_sum": nll,
                "token_count": count,
                "mean_ce": nll / count,
            }
        )
    total_nll = math.fsum(row["nll_sum"] for row in rows)
    total_tokens = sum(row["token_count"] for row in rows)
    avg_ce = total_nll / total_tokens
    return {
        "num_examples": 4,
        "supervised_tokens": total_tokens,
        "total_nll": total_nll,
        "avg_ce": avg_ce,
        "perplexity": math.exp(avg_ce),
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "data": str(data_path),
        "variant": "affine_lm_head_plus_hidden_lora",
        "seed": 42,
        "source_start_index": 0,
        "source_end_index": 4,
        "affine_ablation": "none",
        "per_example": rows,
    }


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="fhfp32_summary_test_") as temporary:
        directory = Path(temporary)
        data = directory / "data.jsonl"
        source_rows = [
            {
                "record_id": f"id-{index}",
                "source_file": "synthetic.jsonl",
                "source_index": index,
            }
            for index in range(4)
        ]
        data.write_text(
            "".join(json.dumps(row) + "\n" for row in source_rows),
            encoding="utf-8",
        )
        model = directory / "model"
        model.mkdir()
        run_a = directory / "alora"
        run_v = directory / "vocab"
        run_a.mkdir()
        run_v.mkdir()
        path_a = run_a / "test_report.json"
        path_v = run_v / "test_report.json"
        atomic_write(
            path_a,
            json.dumps(
                synthetic_report_payload(
                    run_a, data, model, nll_offset=-0.05
                )
            ),
        )
        atomic_write(
            path_v,
            json.dumps(
                synthetic_report_payload(run_v, data, model, nll_offset=0.05)
            ),
        )
        sources = read_source_rows(data, 4)
        report_a = validate_report(
            path_a,
            run_dir=run_a,
            data_path=data,
            model_path=model,
            source_rows=sources,
            seed=42,
            variant="affine_lm_head_plus_hidden_lora",
            fixed_boundary=True,
        )
        report_v = validate_report(
            path_v,
            run_dir=run_v,
            data_path=data,
            model_path=model,
            source_rows=sources,
            seed=42,
            variant="affine_lm_head_plus_hidden_lora",
            fixed_boundary=True,
        )
        bootstrap = paired_item_bootstrap(
            report_a, report_v, rng_seed=7, samples=500
        )
        if not (
            bootstrap["delta_ce_alora_minus_vocab"] < 0
            and bootstrap["ci95"][1] < 0
            and bootstrap["probability_alora_better"] == 1.0
        ):
            raise AssertionError("Synthetic bootstrap direction test failed")

        bad_order = synthetic_report_payload(
            run_v, data, model, nll_offset=0.05
        )
        bad_order["per_example"][0], bad_order["per_example"][1] = (
            bad_order["per_example"][1],
            bad_order["per_example"][0],
        )
        bad_path = run_v / "bad_order.json"
        atomic_write(bad_path, json.dumps(bad_order))
        try:
            validate_report(
                bad_path,
                run_dir=run_v,
                data_path=data,
                model_path=model,
                source_rows=sources,
                seed=42,
                variant="affine_lm_head_plus_hidden_lora",
                fixed_boundary=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Record-order corruption was not rejected")

        token_bad_path = run_v / "bad_tokens.json"
        atomic_write(
            token_bad_path,
            json.dumps(
                synthetic_report_payload(
                    run_v, data, model, nll_offset=0.05, token_offset_at=2
                )
            ),
        )
        token_bad = validate_report(
            token_bad_path,
            run_dir=run_v,
            data_path=data,
            model_path=model,
            source_rows=sources,
            seed=42,
            variant="affine_lm_head_plus_hidden_lora",
            fixed_boundary=True,
        )
        try:
            assert_paired_order_and_tokens(
                [report_a, token_bad], label="synthetic token pairing"
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Token-count corruption was not rejected")

        digest = "0" * 64
        fake_runs: dict[str, Any] = {}
        for seed in SEEDS:
            for method, metadata in METHODS.items():
                name = run_name(method, seed)
                fake_runs[name] = {
                    "seed": seed,
                    "kind": metadata["kind"],
                    "source_hidden_canonical_sha256": digest,
                    "frozen_native_weight_sha256": digest,
                    "boundary_file_sha256": digest,
                    "dev_avg_ce": 1.0,
                    "test_avg_ce": 1.0,
                    "dev_report_sha256": digest,
                    "test_report_sha256": digest,
                }
        fake_completion = {
            "marker": "fixed_hidden_boundary_fp32_qwen25_complete_v1",
            "training_entry_sha256": digest,
            "evaluator_sha256": digest,
            "pair_identity_assertion": "passed",
            "all_artifact_validation_assertion": "passed",
            "runs": fake_runs,
        }
        validate_completion_payload_structure(fake_completion)
        removed = fake_runs.pop(run_name("alora", 44))
        try:
            validate_completion_payload_structure(fake_completion)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing manifest run was not rejected")
        fake_runs[run_name("alora", 44)] = removed
    print(
        "SELF-TEST PASSED: arithmetic, bootstrap direction, record order, "
        "token pairing, and manifest completeness fail-closed checks"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    experiment_dir = args.experiment_dir.resolve()
    completion_path = (
        args.completion_manifest.resolve()
        if args.completion_manifest
        else experiment_dir / "completion_manifest.json"
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json
        else experiment_dir / "ce_summary.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md
        else experiment_dir / "ce_summary.md"
    )
    summary = collect(experiment_dir, completion_path)
    atomic_write(
        output_json,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(output_md, render_markdown(summary))
    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "primary_test_summary": summary["primary_test"][
                    "seed_level_summary"
                ],
                "sensitivity_status": summary[
                    "sensitivity_common_scale_seed42"
                ]["status"],
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
