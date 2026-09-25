#!/usr/bin/env python
"""Fail-closed validation for the fixed-hidden boundary IFEval control.

The validator binds every response artifact to:

* one of the six predeclared fixed-hidden checkpoints;
* that checkpoint's audited hidden and boundary tensor hashes;
* the exact 541-row canonical Google IFEval order; and
* the frozen greedy/native-template generation protocol.

The protocol manifest is immutable once any IFEval artifact exists.  This
prevents a stale shard from being silently reused after a checkpoint or
generation implementation changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import load_dataset
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "reviewer_followup/fixed_hidden_boundary_fp32_qwen25"
CHECKPOINT_ROOT = EXPERIMENT / "checkpoints"
TRAINING_ENTRY = (
    ROOT / "reviewer_followup/train_corrected_sft_fixed_hidden_boundary_fp32.py"
)
GENERATOR = ROOT / "reviewer_followup/evaluate_fixed_hidden_boundary_ifeval.py"
MERGER = ROOT / "reviewer_followup/merge_fixed_hidden_boundary_ifeval_shards.py"

EXPECTED_COUNT = 541
SHARD_BOUNDS = (0, 68, 136, 204, 271, 339, 407, 475, 541)
EXPECTED_CANONICAL_SHA256 = (
    "4d49ac039cbebdfc4beb3f9f30435c4fa25320caffaae441c6b9d868126744eb"
)
EXPECTED_TRAINING_ENTRY_SHA256 = (
    "b493235315646d96ff27d8da373aa79eb216eb333375756e770270709da8fa81"
)
EXPECTED_GENERATOR_SHA256 = (
    "fb5ab319d6335342c25d7fdd52e04ab383a5cf1135660842f76c00552a41d459"
)
EXPECTED_MERGER_SHA256 = (
    "97e45d868ffe6d38894cd2927710833a27671d680e77df5e914e20a25126197b"
)
EXPECTED_SCORER_SHA256 = (
    "e6df07a04d25a0e7134933ba2400c26e0129a243e449c88df55043e583fc4b4a"
)
EXPECTED_MODEL_CONFIG_SHA256 = (
    "0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77"
)
EXPECTED_TRAIN_SHA256 = (
    "e56c2eb28a09aa409015b9b809d79403c0e43f33741d56d7df28bf76e11a00ed"
)
EXPECTED_PIPELINE_SHA256 = (
    "47ef96c81e32234d220765279ba634e710162518f08678dc3a32309176c6e460"
)
EXPECTED_FROZEN_NATIVE_SHA256 = (
    "15eea567d71805884a49191e04210363eb9fe4af05cdc5e6a58765698e0805a9"
)
EXPECTED_FROZEN_FP32_SHA256 = (
    "c72663402ed2b590c76a6567649b0ad3993ababb9112f06961f57b5ef6c09c29"
)
SOURCE_FILE_SHA256 = {
    42: "354ed4961fb4d6dc7cf77d5e1e0ef528726b90cd71776e533088fa6d9b77e167",
    43: "16958535a58d7cb3dd6c536fd1ff89beb7f88e053a9a67fb20eeb8a90135a093",
    44: "2890f3f5c0eef7104616179d36916aa3ce1999047247e27e06d2c31b4d379476",
}
SOURCE_TENSOR_SHA256 = {
    42: "92fe20f386d39b89e99591108b5fbcf464b0705423fa379e48e6361950381b44",
    43: "5485e347c6315cbef11099fd17e6a9b03431b33b3beaa0b15e00a9791f0f3d1f",
    44: "baceb5f2f02f3327aeae089f103e7faa792121b732d3b55b1c5078f189404d62",
}
RUN_NAME_RE = re.compile(
    r"^qwen25_15b_fhfp32_hsd(?P<hidden_seed>42|43|44)_"
    r"(?P<endpoint>alora_r50_s16|vocab_lora_r1_s32)_"
    r"bsd(?P<boundary_seed>42|43|44)$"
)
RESPONSE_FIELDS = {
    "key",
    "prompt",
    "response",
    "instruction_id_list",
    "kwargs",
    "run_dir",
    "variant",
    "seed",
    "affine_ablation",
}
SCORE_FIELDS = {
    "follow_all_instructions",
    "follow_instruction_list",
    "instruction_id_list",
    "prompt",
    "response",
}
PROTOCOL_FIELDS = {
    "marker",
    "purpose",
    "ifeval_used_for_endpoint_selection",
    "canonical_dataset",
    "generation",
    "merge",
    "official_scoring",
    "run_order",
    "runs",
}
HIDDEN_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
}


def run_name(kind: str, seed: int) -> str:
    if kind == "alora":
        endpoint = "alora_r50_s16"
    elif kind == "vocab_lora":
        endpoint = "vocab_lora_r1_s32"
    else:
        raise ValueError(f"Unknown boundary kind: {kind!r}")
    return f"qwen25_15b_fhfp32_hsd{seed}_{endpoint}_bsd{seed}"


EXPECTED_RUN_NAMES = tuple(
    run_name(kind, seed)
    for seed in (42, 43, 44)
    for kind in ("alora", "vocab_lora")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    checkpoint = commands.add_parser("checkpoint")
    checkpoint.add_argument("--run-dir", required=True, type=Path)

    protocol = commands.add_parser("protocol")
    protocol.add_argument("--path", required=True, type=Path)
    protocol.add_argument(
        "--create",
        action="store_true",
        help="Create only when the output root contains no other artifacts.",
    )
    protocol.add_argument(
        "--scorer-path",
        required=True,
        type=Path,
        help="Resolved instruction_following_eval/evaluation_main.py.",
    )

    shard = commands.add_parser("shard")
    shard.add_argument("--path", required=True, type=Path)
    shard.add_argument("--run-dir", required=True, type=Path)
    shard.add_argument("--protocol-manifest", required=True, type=Path)
    shard.add_argument("--start-index", required=True, type=int)
    shard.add_argument("--end-index", required=True, type=int)

    merged = commands.add_parser("merged")
    merged.add_argument("--path", required=True, type=Path)
    merged.add_argument("--run-dir", required=True, type=Path)
    merged.add_argument("--protocol-manifest", required=True, type=Path)

    scores = commands.add_parser("scores")
    scores.add_argument("--response-path", required=True, type=Path)
    scores.add_argument("--score-dir", required=True, type=Path)
    scores.add_argument("--run-dir", required=True, type=Path)
    scores.add_argument("--protocol-manifest", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read UTF-8 JSONL from {path}: {exc}") from exc
    if not raw:
        raise ValueError(f"Empty JSONL artifact: {path}")
    if not raw.endswith("\n"):
        raise ValueError(f"JSONL artifact lacks a final newline: {path}")
    lines = raw.split("\n")
    lines.pop()
    if any(line == "" for line in lines):
        raise ValueError(f"Blank record inside JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON record {number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}: record {number} is not an object")
        rows.append(value)
    return rows


def require_equal(source: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{source}={actual!r}, expected {expected!r}")


def require_close(source: str, actual: Any, expected: float) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{source}={actual!r}, expected {expected!r}")


def resolve_from_root(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected a nonempty path string, got {value!r}")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def inspect_tensors(
    path: Path, expected_shapes: dict[str, tuple[int, ...]]
) -> tuple[int, dict[str, str]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty safetensors file: {path}")
    with safe_open(path, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
        require_equal(f"{path}: tensor names", set(keys), set(expected_shapes))
        dtypes: dict[str, str] = {}
        count = 0
        for key in keys:
            tensor = stream.get_tensor(key)
            require_equal(
                f"{path}: {key} shape", tuple(tensor.shape), expected_shapes[key]
            )
            dtypes[key] = str(tensor.dtype)
            count += tensor.numel()
    return count, dtypes


def parse_run_dir(run_dir: Path) -> tuple[str, int]:
    match = RUN_NAME_RE.fullmatch(run_dir.name)
    if match is None:
        raise ValueError(f"Not one of the six fixed main endpoints: {run_dir.name}")
    hidden_seed = int(match["hidden_seed"])
    boundary_seed = int(match["boundary_seed"])
    require_equal(f"{run_dir.name}: paired seeds", boundary_seed, hidden_seed)
    kind = "alora" if match["endpoint"].startswith("alora") else "vocab_lora"
    require_equal(f"{run_dir.name}: canonical name", run_dir.name, run_name(kind, hidden_seed))
    return kind, hidden_seed


def validate_checkpoint(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    kind, seed = parse_run_dir(run_dir)
    if run_dir.parent != CHECKPOINT_ROOT.resolve():
        raise ValueError(
            f"{run_dir}: checkpoint must be directly under {CHECKPOINT_ROOT.resolve()}"
        )

    required = (
        "run_args.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "fixed_boundary_config.json",
        "fixed_boundary_adapter.safetensors",
        "fixed_hidden_boundary_audit.json",
        "trainable_summary.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"{run_dir.name}: incomplete checkpoint, missing {missing}")

    args = read_json(run_dir / "run_args.json")
    adapter = read_json(run_dir / "adapter_config.json")
    config = read_json(run_dir / "fixed_boundary_config.json")
    audit = read_json(run_dir / "fixed_hidden_boundary_audit.json")
    summary = read_json(run_dir / "trainable_summary.json")

    if kind == "alora":
        rank, alpha, scale, count = 50, 800.0, 16.0, 153_600
        residual_path = (
            "autocast_off_FP32_F.linear(scale*U(D(hidden_fp32)),W_fp32)"
        )
        shapes = {
            "affine.down.weight": (50, 1536),
            "affine.up.weight": (1536, 50),
        }
    else:
        rank, alpha, scale, count = 1, 32.0, 32.0, 153_472
        residual_path = "autocast_off_FP32_scale*B(A(hidden_fp32))"
        shapes = {
            "affine.down.weight": (1, 1536),
            "affine.up.weight": (151936, 1),
        }

    exact_args = {
        "variant": "affine_lm_head_plus_hidden_lora",
        "seed": seed,
        "freeze_initial_hidden_lora": True,
        "affine_rank": rank,
        "affine_dropout": 0.0,
        "no_affine_input_bias": True,
        "affine_lm_head_bias": False,
        "tie_affine_input_lm_head_adapters": False,
        "include_emb_lmh_lora_rank": 0,
        "initial_affine_adapter": None,
        "max_seq_len": 1024,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "max_train_samples": None,
        "dataset_split": "train",
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,
        "base_dtype": "auto",
        "master_dtype": "fp32",
        "save_strategy": "no",
        "skip_final_model_save": False,
        "logging_steps": 10,
        "resume_from_checkpoint": None,
        "eval_data": None,
        "eval_samples": 0,
        "reference_run_dir": None,
        "anchor_data": None,
        "auxiliary_anchor_data": None,
    }
    for key, expected in exact_args.items():
        require_equal(f"{run_dir.name}: run_args.{key}", args.get(key), expected)
    for key, expected in {
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "affine_alpha": alpha,
        "affine_learning_rate_scale": 1.0,
        "affine_bias_learning_rate_scale": 1.0,
        "affine_energy_lambda": 0.0,
        "affine_bias_energy_lambda": 0.0,
        "reference_kl_lambda": 0.0,
        "anchor_fraction": 0.0,
        "auxiliary_anchor_lambda": 0.0,
    }.items():
        require_close(f"{run_dir.name}: run_args.{key}", args.get(key), expected)
    require_equal(
        f"{run_dir.name}: run_args.output_dir",
        resolve_from_root(args.get("output_dir")),
        run_dir,
    )
    require_equal(
        f"{run_dir.name}: run_args.model_path config hash",
        sha256_file(resolve_from_root(args.get("model_path")) / "config.json"),
        EXPECTED_MODEL_CONFIG_SHA256,
    )
    require_equal(
        f"{run_dir.name}: run_args.train_data",
        resolve_from_root(args.get("train_data")),
        (ROOT / "corrected_sft_experiment/data/train.jsonl").resolve(),
    )
    require_equal(
        f"{run_dir.name}: train data hash",
        sha256_file(resolve_from_root(args.get("train_data"))),
        EXPECTED_TRAIN_SHA256,
    )
    source_dir = (
        ROOT
        / f"corrected_sft_experiment/outputs/formal/qwen25_15b_hidden_sd{seed}"
    ).resolve()
    require_equal(
        f"{run_dir.name}: initial hidden adapter",
        resolve_from_root(args.get("initial_hidden_lora_adapter")),
        source_dir,
    )
    require_equal(
        f"{run_dir.name}: corrected pipeline hash",
        args.get("corrected_data_pipeline", {}).get("implementation_sha256"),
        EXPECTED_PIPELINE_SHA256,
    )

    require_equal(f"{run_dir.name}: adapter type", adapter.get("peft_type"), "LORA")
    require_equal(f"{run_dir.name}: hidden rank", adapter.get("r"), 8)
    require_equal(f"{run_dir.name}: hidden alpha", adapter.get("lora_alpha"), 16)
    require_close(
        f"{run_dir.name}: hidden dropout", adapter.get("lora_dropout"), 0.05
    )
    require_equal(
        f"{run_dir.name}: hidden targets",
        set(adapter.get("target_modules", [])),
        HIDDEN_TARGETS,
    )
    with safe_open(
        run_dir / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as stream:
        hidden_keys = list(stream.keys())
        if (
            len(hidden_keys) != 392
            or any(
                "lora_" not in key
                or "lm_head" in key
                or "embed_tokens" in key
                for key in hidden_keys
            )
        ):
            raise ValueError(f"{run_dir.name}: saved adapter is not pure hidden LoRA")
        hidden_count = sum(
            math.prod(stream.get_slice(key).get_shape()) for key in hidden_keys
        )
    require_equal(f"{run_dir.name}: hidden parameter count", hidden_count, 9_232_384)
    require_equal(
        f"{run_dir.name}: saved hidden file hash",
        sha256_file(run_dir / "adapter_model.safetensors"),
        SOURCE_FILE_SHA256[seed],
    )
    require_equal(f"{run_dir.name}: trainable count", summary.get("trainable"), count)

    for payload_name, payload in (("config", config), ("audit", audit)):
        require_equal(
            f"{run_dir.name}: {payload_name}.marker",
            payload.get("marker"),
            "fixed_hidden_output_boundary_fp32_v1",
        )
        require_equal(
            f"{run_dir.name}: {payload_name}.kind", payload.get("kind"), kind
        )
    for key, expected in {
        "rank": rank,
        "hidden_size": 1536,
        "vocab_size": 151936,
        "trainable_parameters": count,
        "boundary_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "autocast_disabled": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "input_adapter": False,
        "boundary_bias": False,
        "boundary_dropout": 0.0,
        "frozen_lm_head_weight_dtype": "torch.bfloat16",
        "zero_initialized_residual": True,
        "common_base_path": "autocast_off_F.linear(hidden_native,W_native).float",
        "residual_path": residual_path,
        "frozen_native_weight_sha256": EXPECTED_FROZEN_NATIVE_SHA256,
        "frozen_fp32_weight_sha256": EXPECTED_FROZEN_FP32_SHA256,
    }.items():
        require_equal(f"{run_dir.name}: config.{key}", config.get(key), expected)
    require_close(f"{run_dir.name}: config.alpha", config.get("alpha"), alpha)
    require_close(f"{run_dir.name}: config.scale", config.get("scale"), scale)

    require_equal(f"{run_dir.name}: audit.tf32_mode", audit.get("tf32_mode"), "deny")
    require_equal(
        f"{run_dir.name}: audit.cuda matmul TF32",
        audit.get("cuda_matmul_allow_tf32"),
        False,
    )
    require_equal(
        f"{run_dir.name}: audit.cudnn TF32",
        audit.get("cudnn_allow_tf32"),
        False,
    )
    require_equal(
        f"{run_dir.name}: audit.completion",
        audit.get("completion_audit_assertion"),
        "passed",
    )
    require_equal(
        f"{run_dir.name}: audit.training implementation hash",
        audit.get("implementation_sha256"),
        EXPECTED_TRAINING_ENTRY_SHA256,
    )
    require_equal(
        f"{run_dir.name}: audit.training implementation",
        resolve_from_root(audit.get("implementation")),
        TRAINING_ENTRY.resolve(),
    )
    require_equal(
        f"{run_dir.name}: audit.saved boundary count",
        audit.get("saved_boundary_parameter_count"),
        count,
    )
    require_equal(
        f"{run_dir.name}: audit.saved hidden unchanged",
        audit.get("saved_hidden_unchanged_assertion"),
        "passed",
    )
    hidden = audit.get("hidden_checkpoint")
    if not isinstance(hidden, dict):
        raise ValueError(f"{run_dir.name}: missing hidden checkpoint audit")
    for key, expected in {
        "source_adapter_file_sha256": SOURCE_FILE_SHA256[seed],
        "canonical_tensor_sha256": SOURCE_TENSOR_SHA256[seed],
        "tensor_count": 392,
        "parameter_count": 9_232_384,
        "tensor_exact_assertion": "passed",
        "dropout_identity_assertion": "passed",
        "dropout_modules_replaced_with_identity": 196,
    }.items():
        require_equal(f"{run_dir.name}: hidden audit.{key}", hidden.get(key), expected)
    require_equal(
        f"{run_dir.name}: hidden source",
        resolve_from_root(hidden.get("source_adapter_dir")),
        source_dir,
    )
    require_equal(
        f"{run_dir.name}: config/audit hidden metadata",
        config.get("hidden_checkpoint_audit"),
        hidden,
    )
    require_equal(
        f"{run_dir.name}: saved hidden canonical hash",
        audit.get("saved_hidden_canonical_tensor_sha256"),
        SOURCE_TENSOR_SHA256[seed],
    )
    require_equal(
        f"{run_dir.name}: saved hidden file hash audit",
        audit.get("saved_hidden_file_sha256"),
        SOURCE_FILE_SHA256[seed],
    )
    require_equal(
        f"{run_dir.name}: embedded run audit",
        args.get("fixed_hidden_boundary_control"),
        audit,
    )

    optimizer = audit.get("boundary_optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError(f"{run_dir.name}: missing boundary optimizer audit")
    for key, expected in {
        "only_boundary_trainable_assertion": "passed",
        "hidden_frozen_assertion": "passed",
        "optimizer_parameter_coverage_assertion": "passed",
        "trainable_parameters": count,
        "optimizer_parameter_dtypes": ["torch.float32"],
        "learning_rates": [2e-4],
        "max_grad_norm": 0.0,
    }.items():
        require_equal(
            f"{run_dir.name}: boundary optimizer.{key}", optimizer.get(key), expected
        )
    require_equal(
        f"{run_dir.name}: boundary trainable dtypes",
        set(optimizer.get("trainable_dtypes", {}).values()),
        {"torch.float32"},
    )
    expected_optimizer_shapes = {
        "base_model.model.lm_head.affine.down.weight": list(
            shapes["affine.down.weight"]
        ),
        "base_model.model.lm_head.affine.up.weight": list(
            shapes["affine.up.weight"]
        ),
    }
    require_equal(
        f"{run_dir.name}: optimizer group count",
        optimizer.get("optimizer_group_count"),
        2,
    )
    require_equal(
        f"{run_dir.name}: optimizer trainable shapes",
        optimizer.get("trainable_shapes"),
        expected_optimizer_shapes,
    )
    require_equal(
        f"{run_dir.name}: optimizer trainable names",
        optimizer.get("trainable_names"),
        list(expected_optimizer_shapes),
    )

    boundary_path = run_dir / "fixed_boundary_adapter.safetensors"
    boundary_count, boundary_dtypes = inspect_tensors(boundary_path, shapes)
    require_equal(f"{run_dir.name}: boundary tensor count", boundary_count, count)
    require_equal(
        f"{run_dir.name}: boundary tensor dtypes",
        set(boundary_dtypes.values()),
        {"torch.float32"},
    )
    boundary_sha = sha256_file(boundary_path)
    require_equal(
        f"{run_dir.name}: audited boundary hash",
        audit.get("saved_boundary_file_sha256"),
        boundary_sha,
    )

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "kind": kind,
        "hidden_seed": seed,
        "boundary_seed": seed,
        "hidden_canonical_tensor_sha256": SOURCE_TENSOR_SHA256[seed],
        "hidden_adapter_file_sha256": SOURCE_FILE_SHA256[seed],
        "boundary_file_sha256": boundary_sha,
        "boundary_parameter_count": count,
        "frozen_native_weight_sha256": EXPECTED_FROZEN_NATIVE_SHA256,
        "training_entry_sha256": EXPECTED_TRAINING_ENTRY_SHA256,
    }


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
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
        digest.update(b"\n")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def canonical_rows_cached() -> tuple[dict[str, Any], ...]:
    rows = tuple(dict(row) for row in load_dataset("google/IFEval", split="train"))
    require_equal("canonical IFEval row count", len(rows), EXPECTED_COUNT)
    require_equal(
        "canonical IFEval digest", canonical_digest(list(rows)), EXPECTED_CANONICAL_SHA256
    )
    keys = [row["key"] for row in rows]
    if len(set(keys)) != EXPECTED_COUNT:
        raise ValueError("Canonical IFEval keys are not unique")
    return rows


def canonical_rows() -> list[dict[str, Any]]:
    return list(canonical_rows_cached())


def expected_protocol_manifest(scorer_path: Path) -> dict[str, Any]:
    scorer_path = scorer_path.resolve()
    require_equal(
        "generation implementation hash",
        sha256_file(GENERATOR),
        EXPECTED_GENERATOR_SHA256,
    )
    require_equal("merger hash", sha256_file(MERGER), EXPECTED_MERGER_SHA256)
    require_equal(
        "official scorer hash", sha256_file(scorer_path), EXPECTED_SCORER_SHA256
    )
    rows = canonical_rows()
    runs = {
        name: validate_checkpoint(CHECKPOINT_ROOT / name)
        for name in EXPECTED_RUN_NAMES
    }
    for seed in (42, 43, 44):
        pair = [runs[run_name(kind, seed)] for kind in ("alora", "vocab_lora")]
        require_equal(
            f"seed {seed}: paired hidden hash",
            len({row["hidden_canonical_tensor_sha256"] for row in pair}),
            1,
        )
        require_equal(
            f"seed {seed}: paired frozen lm_head hash",
            len({row["frozen_native_weight_sha256"] for row in pair}),
            1,
        )
    return {
        "marker": "fixed_hidden_boundary_fp32_qwen25_ifeval_v1",
        "purpose": "frozen_endpoint_evaluation_only",
        "ifeval_used_for_endpoint_selection": False,
        "canonical_dataset": {
            "dataset": "google/IFEval",
            "split": "train",
            "rows": EXPECTED_COUNT,
            "sha256": canonical_digest(rows),
        },
        "generation": {
            "implementation": str(GENERATOR.resolve()),
            "implementation_sha256": EXPECTED_GENERATOR_SHA256,
            "native_chat_template": True,
            "add_generation_prompt": True,
            "do_sample": False,
            "max_new_tokens": 512,
            "batch_size_per_shard": 8,
            "affine_ablation": "none",
            "exact_shard_bounds": list(SHARD_BOUNDS),
            "shards_per_model": 8,
        },
        "merge": {
            "implementation": str(MERGER.resolve()),
            "implementation_sha256": EXPECTED_MERGER_SHA256,
            "canonical_order_required": True,
        },
        "official_scoring": {
            "module": "instruction_following_eval.evaluation_main",
            "source_path": str(scorer_path),
            "source_sha256": EXPECTED_SCORER_SHA256,
            "modes": ["strict", "loose"],
        },
        "run_order": list(EXPECTED_RUN_NAMES),
        "runs": runs,
    }


def validate_protocol_manifest(
    path: Path,
    *,
    scorer_path: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(path.resolve())
    require_equal(f"{path}: protocol fields", set(manifest), PROTOCOL_FIELDS)
    require_equal(
        f"{path}: marker",
        manifest.get("marker"),
        "fixed_hidden_boundary_fp32_qwen25_ifeval_v1",
    )
    require_equal(
        f"{path}: IFEval selection flag",
        manifest.get("ifeval_used_for_endpoint_selection"),
        False,
    )
    require_equal(
        f"{path}: canonical dataset",
        manifest.get("canonical_dataset"),
        {
            "dataset": "google/IFEval",
            "split": "train",
            "rows": EXPECTED_COUNT,
            "sha256": EXPECTED_CANONICAL_SHA256,
        },
    )
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{path}: missing generation protocol")
    require_equal(
        f"{path}: generation protocol",
        generation,
        {
            "implementation": str(GENERATOR.resolve()),
            "implementation_sha256": EXPECTED_GENERATOR_SHA256,
            "native_chat_template": True,
            "add_generation_prompt": True,
            "do_sample": False,
            "max_new_tokens": 512,
            "batch_size_per_shard": 8,
            "affine_ablation": "none",
            "exact_shard_bounds": list(SHARD_BOUNDS),
            "shards_per_model": 8,
        },
    )
    require_equal(
        f"{path}: purpose",
        manifest.get("purpose"),
        "frozen_endpoint_evaluation_only",
    )
    require_equal(
        f"{path}: merge protocol",
        manifest.get("merge"),
        {
            "implementation": str(MERGER.resolve()),
            "implementation_sha256": EXPECTED_MERGER_SHA256,
            "canonical_order_required": True,
        },
    )
    scoring = manifest.get("official_scoring")
    if not isinstance(scoring, dict):
        raise ValueError(f"{path}: missing official scoring protocol")
    require_equal(
        f"{path}: scorer module",
        scoring.get("module"),
        "instruction_following_eval.evaluation_main",
    )
    require_equal(
        f"{path}: scorer modes", scoring.get("modes"), ["strict", "loose"]
    )
    require_equal(
        f"{path}: scorer source hash",
        scoring.get("source_sha256"),
        EXPECTED_SCORER_SHA256,
    )
    scorer_source = Path(str(scoring.get("source_path", ""))).resolve()
    require_equal(
        f"{path}: scorer source on disk",
        sha256_file(scorer_source),
        EXPECTED_SCORER_SHA256,
    )
    require_equal(f"{path}: run order", manifest.get("run_order"), list(EXPECTED_RUN_NAMES))
    runs = manifest.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(EXPECTED_RUN_NAMES):
        raise ValueError(f"{path}: protocol does not name exactly the six main runs")

    if scorer_path is not None:
        expected = expected_protocol_manifest(scorer_path)
        require_equal(f"{path}: complete protocol", manifest, expected)
    elif run_dir is not None:
        identity = validate_checkpoint(run_dir)
        require_equal(
            f"{path}: checkpoint binding for {run_dir.name}",
            runs.get(run_dir.name),
            identity,
        )
    return manifest


def create_or_validate_protocol(path: Path, scorer_path: Path, create: bool) -> None:
    path = path.resolve()
    expected = expected_protocol_manifest(scorer_path)
    if path.exists():
        require_equal(f"{path}: immutable protocol", read_json(path), expected)
        return
    if not create:
        raise ValueError(f"Protocol manifest does not exist: {path}")
    output_root = path.parent
    if output_root.exists():
        existing = [item for item in output_root.iterdir() if item != path]
        if existing:
            raise ValueError(
                f"Refusing to create protocol beside existing artifacts: {existing[:8]}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_responses(
    *,
    path: Path,
    run_dir: Path,
    protocol_manifest: Path,
    start_index: int,
    end_index: int,
    expected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    identity = validate_checkpoint(run_dir)
    validate_protocol_manifest(protocol_manifest, run_dir=run_dir)
    if not 0 <= start_index <= end_index <= EXPECTED_COUNT:
        raise ValueError(f"Invalid interval [{start_index}, {end_index})")
    if expected is None:
        expected = canonical_rows()
    rows = read_jsonl(path.resolve())
    require_equal(
        f"{path}: response count", len(rows), end_index - start_index
    )
    seen: set[Any] = set()
    for offset, (row, reference) in enumerate(
        zip(rows, expected[start_index:end_index], strict=True)
    ):
        index = start_index + offset
        require_equal(f"{path}: row {index} fields", set(row), RESPONSE_FIELDS)
        for field in ("key", "prompt", "instruction_id_list", "kwargs"):
            require_equal(
                f"{path}: row {index} canonical {field}",
                row[field],
                reference[field],
            )
        if not isinstance(row["response"], str):
            raise ValueError(f"{path}: row {index} response is not a string")
        require_equal(
            f"{path}: row {index} run_dir", row["run_dir"], str(run_dir)
        )
        require_equal(
            f"{path}: row {index} variant",
            row["variant"],
            "affine_lm_head_plus_hidden_lora",
        )
        require_equal(
            f"{path}: row {index} seed", row["seed"], identity["boundary_seed"]
        )
        require_equal(
            f"{path}: row {index} ablation", row["affine_ablation"], "none"
        )
        if row["key"] in seen:
            raise ValueError(f"{path}: duplicate key at row {index}")
        seen.add(row["key"])
    return rows


def validate_score_rows(
    path: Path, response_rows: list[dict[str, Any]]
) -> list[bool]:
    rows = read_jsonl(path)
    require_equal(f"{path}: score count", len(rows), EXPECTED_COUNT)
    outcomes: list[bool] = []
    for index, (row, response) in enumerate(
        zip(rows, response_rows, strict=True)
    ):
        require_equal(f"{path}: row {index} fields", set(row), SCORE_FIELDS)
        for field in ("prompt", "response", "instruction_id_list"):
            require_equal(
                f"{path}: row {index} response {field}",
                row[field],
                response[field],
            )
        follow_all = row["follow_all_instructions"]
        follow_list = row["follow_instruction_list"]
        if type(follow_all) is not bool:
            raise ValueError(f"{path}: row {index} aggregate is not boolean")
        if (
            not isinstance(follow_list, list)
            or not follow_list
            or not all(type(item) is bool for item in follow_list)
        ):
            raise ValueError(f"{path}: row {index} malformed instruction results")
        require_equal(
            f"{path}: row {index} instruction-result count",
            len(follow_list),
            len(response["instruction_id_list"]),
        )
        require_equal(
            f"{path}: row {index} aggregate consistency",
            follow_all,
            all(follow_list),
        )
        outcomes.append(follow_all)
    return outcomes


def validate_scores(
    response_path: Path,
    score_dir: Path,
    run_dir: Path,
    protocol_manifest: Path,
) -> dict[str, list[bool]]:
    responses = validate_responses(
        path=response_path,
        run_dir=run_dir,
        protocol_manifest=protocol_manifest,
        start_index=0,
        end_index=EXPECTED_COUNT,
    )
    expected_files = {
        "eval_results_strict.jsonl",
        "eval_results_loose.jsonl",
    }
    if not score_dir.is_dir():
        raise ValueError(f"Score directory is missing: {score_dir}")
    actual_files = {item.name for item in score_dir.iterdir() if item.is_file()}
    require_equal(f"{score_dir}: score files", actual_files, expected_files)
    return {
        mode: validate_score_rows(
            score_dir / f"eval_results_{mode}.jsonl", responses
        )
        for mode in ("strict", "loose")
    }


def main() -> None:
    args = parse_args()
    if args.command == "checkpoint":
        identity = validate_checkpoint(args.run_dir)
        print(json.dumps(identity, ensure_ascii=False, sort_keys=True))
    elif args.command == "protocol":
        create_or_validate_protocol(args.path, args.scorer_path, args.create)
        print(f"valid protocol manifest: {args.path.resolve()}")
    elif args.command == "shard":
        validate_responses(
            path=args.path,
            run_dir=args.run_dir,
            protocol_manifest=args.protocol_manifest,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        print(
            f"valid shard: {args.path.resolve()} "
            f"[{args.start_index}, {args.end_index})"
        )
    elif args.command == "merged":
        validate_responses(
            path=args.path,
            run_dir=args.run_dir,
            protocol_manifest=args.protocol_manifest,
            start_index=0,
            end_index=EXPECTED_COUNT,
        )
        print(f"valid merged responses: {args.path.resolve()}")
    elif args.command == "scores":
        validate_scores(
            args.response_path,
            args.score_dir,
            args.run_dir,
            args.protocol_manifest,
        )
        print(f"valid official scores: {args.score_dir.resolve()}")
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
