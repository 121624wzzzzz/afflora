#!/usr/bin/env python
"""Strict artifact checks for the output-only equal-budget IFEval run.

This helper is intentionally stricter than the generic shard merger.  It
checks checkpoint identity, every generated row against the canonical
541-prompt Google IFEval split, and the two official scorer outputs.  The
launcher uses these checks to decide whether an artifact is safe to reuse
after an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from safetensors import safe_open


EXPECTED_COUNT = 541
RUN_NAME_RE = re.compile(
    r"^qwen25_15b_out_(?P<method>aff_r50|vocab_r1)_"
    r"s(?P<scale>1|2|4|8|16|32)"
    r"(?:_blr(?P<boundary_lr>0p5|2))?_sd(?P<seed>\d+)$"
)
REQUIRED_RESPONSE_FIELDS = {
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
HIDDEN_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser(
        "checkpoint", help="validate a completed final-experiment checkpoint"
    )
    checkpoint.add_argument("--run-dir", required=True)

    shard = subparsers.add_parser(
        "shard", help="validate one response shard against canonical IFEval rows"
    )
    shard.add_argument("--path", required=True)
    shard.add_argument("--run-dir", required=True)
    shard.add_argument("--start-index", required=True, type=int)
    shard.add_argument("--end-index", required=True, type=int)

    merged = subparsers.add_parser(
        "merged", help="validate a merged 541-response artifact"
    )
    merged.add_argument("--path", required=True)
    merged.add_argument("--run-dir", required=True)

    scores = subparsers.add_parser(
        "scores", help="validate strict and loose official scorer artifacts"
    )
    scores.add_argument("--response-path", required=True)
    scores.add_argument("--score-dir", required=True)
    scores.add_argument("--run-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}")
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

    # U+2028 may legitimately occur in a JSON response string.  Split only on
    # the JSONL byte delimiter rather than using str.splitlines().
    lines = raw.split("\n")
    if lines[-1] != "":
        raise AssertionError("final-newline check and split result disagree")
    lines.pop()
    if any(line == "" for line in lines):
        raise ValueError(f"Blank record inside JSONL artifact: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} at record {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected object in {path} at record {line_number}, "
                f"got {type(value).__name__}"
            )
        rows.append(value)
    return rows


def inspect_safetensors(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty checkpoint tensor file: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
    except Exception as exc:  # safetensors exposes several format-error classes.
        raise ValueError(f"Cannot read safetensors header from {path}: {exc}") from exc
    if not keys:
        raise ValueError(f"No tensors stored in {path}")
    return keys


def require_close(source: str, actual: Any, expected: float) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{source}={actual!r}, expected {expected!r}")


def parse_run_identity(
    run_dir: Path,
) -> tuple[dict[str, Any], str, int, float, int]:
    run_dir = run_dir.resolve()
    match = RUN_NAME_RE.fullmatch(run_dir.name)
    if match is None:
        raise ValueError(
            "Run directory must follow the fixed experiment naming scheme "
            f"{RUN_NAME_RE.pattern!r}; got {run_dir.name!r}"
        )
    run_args_path = run_dir / "run_args.json"
    if not run_args_path.is_file():
        raise ValueError(f"Checkpoint is not complete: missing {run_args_path}")
    run_args = read_json(run_args_path)
    seed = int(match["seed"])
    scale = int(match["scale"])
    boundary_lr_tag = match["boundary_lr"]
    boundary_lr = 1.0 if boundary_lr_tag is None else (
        0.5 if boundary_lr_tag == "0p5" else 2.0
    )
    if run_args.get("seed") != seed:
        raise ValueError(
            f"Seed mismatch for {run_dir}: name={seed}, run_args={run_args.get('seed')!r}"
        )
    return run_args, match["method"], scale, boundary_lr, seed


def validate_vocab_optimizer_audit(
    run_dir: Path,
    run_args: dict[str, Any],
    boundary_lr: float,
    *,
    required: bool,
) -> None:
    audit_path = run_dir / "optimizer_groups.json"
    embedded = run_args.get("output_vocab_lr")
    if not required and not audit_path.exists() and embedded is None:
        return
    if not audit_path.is_file():
        raise ValueError(f"{run_dir}: missing optimizer audit {audit_path}")
    if not isinstance(embedded, dict):
        raise ValueError(f"{run_dir}: run_args.output_vocab_lr is missing")

    audit = read_json(audit_path)
    expected: dict[str, Any] = {
        "hidden_learning_rate": 2e-4,
        "output_vocab_learning_rate": 2e-4 * boundary_lr,
        "output_vocab_lr_scale": boundary_lr,
        "hidden_lora_params": 9_232_384,
        "output_vocab_lora_params": 153_472,
        "total_trainable_params": 9_385_856,
        "output_lora_rank": 1,
        "output_lora_a_shape": [1, 1536],
        "output_lora_b_shape": [151936, 1],
        "coverage_assertion": "passed",
    }
    for key, expected_value in expected.items():
        for source_name, source in (
            ("optimizer_groups", audit),
            ("run_args.output_vocab_lr", embedded),
        ):
            actual = source.get(key)
            if isinstance(expected_value, float):
                require_close(
                    f"{run_dir}: {source_name}.{key}", actual, expected_value
                )
            elif actual != expected_value:
                raise ValueError(
                    f"{run_dir}: {source_name}.{key}={actual!r}, "
                    f"expected {expected_value!r}"
                )

    groups = audit.get("optimizer_group_params")
    embedded_groups = embedded.get("optimizer_group_params")
    if groups != embedded_groups or not isinstance(groups, dict):
        raise ValueError(
            f"{run_dir}: optimizer group parameter metadata is missing or inconsistent"
        )
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
    if hidden_group_count != 9_232_384 or boundary_group_count != 153_472:
        raise ValueError(
            f"{run_dir}: optimizer group counts are "
            f"hidden={hidden_group_count}, boundary={boundary_group_count}"
        )

    implementation = embedded.get("implementation")
    expected_implementation = (
        Path(__file__).resolve().parent
        / "train_corrected_sft_output_vocab_boundary_lr.py"
    ).resolve()
    if not isinstance(implementation, str) or Path(implementation).resolve() != (
        expected_implementation
    ):
        raise ValueError(
            f"{run_dir}: unexpected output-boundary optimizer implementation "
            f"{implementation!r}"
        )


def validate_affine_optimizer_metadata(
    run_dir: Path, run_args: dict[str, Any], boundary_lr: float
) -> None:
    require_close(
        f"{run_dir}: run_args.affine_learning_rate_scale",
        run_args.get("affine_learning_rate_scale"),
        boundary_lr,
    )
    require_close(
        f"{run_dir}: run_args.affine_bias_learning_rate_scale",
        run_args.get("affine_bias_learning_rate_scale"),
        1.0,
    )
    if run_args.get("output_vocab_lr") is not None:
        raise ValueError(f"{run_dir}: A-LoRA contains Vocab-LoRA optimizer metadata")
    if (run_dir / "optimizer_groups.json").exists():
        raise ValueError(f"{run_dir}: A-LoRA contains a Vocab-LoRA optimizer audit")
    if boundary_lr == 1.0:
        return

    log_path = run_dir.parent.parent / "logs" / f"{run_dir.name}.train.log"
    if not log_path.is_file():
        raise ValueError(
            f"{run_dir}: non-default A-LoRA boundary LR requires {log_path}"
        )
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "[optimizer] separate_affine_lr",
        "base_lr=0.0002",
        f"affine_lr={2e-4 * boundary_lr:g}",
        f"affine_scale={boundary_lr:g}",
        "bias_scale=1",
        "'hidden_or_other': 9232384",
        "'affine_weight': 153600",
        "'affine_bias': 0",
    )
    missing = [marker for marker in required_markers if marker not in contents]
    if missing:
        raise ValueError(
            f"{run_dir}: affine optimizer log lacks expected markers {missing}"
        )


def validate_checkpoint(
    run_dir: Path,
) -> tuple[dict[str, Any], str, int, float, int]:
    run_dir = run_dir.resolve()
    run_args, method, scale, boundary_lr, seed = parse_run_identity(run_dir)

    if run_args.get("hidden_lora_rank") != 8:
        raise ValueError(f"{run_dir}: hidden_lora_rank is not 8")
    if run_args.get("hidden_lora_alpha") != 16:
        raise ValueError(f"{run_dir}: hidden_lora_alpha is not 16")
    if float(run_args.get("hidden_lora_dropout", -1)) != 0.05:
        raise ValueError(f"{run_dir}: hidden_lora_dropout is not 0.05")
    targets = {
        item.strip()
        for item in str(run_args.get("hidden_lora_target_modules", "")).split(",")
        if item.strip()
    }
    if targets != HIDDEN_TARGETS:
        raise ValueError(
            f"{run_dir}: hidden target modules are {sorted(targets)}, "
            f"expected {sorted(HIDDEN_TARGETS)}"
        )
    require_close(
        f"{run_dir}: run_args.learning_rate",
        run_args.get("learning_rate"),
        2e-4,
    )

    adapter_config_path = run_dir / "adapter_config.json"
    adapter_config = read_json(adapter_config_path)
    adapter_keys = inspect_safetensors(run_dir / "adapter_model.safetensors")
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError(f"{adapter_config_path}: expected PEFT LORA adapter")
    trainable_summary = read_json(run_dir / "trainable_summary.json")

    if method == "aff_r50":
        if run_args.get("variant") != "affine_lm_head_plus_hidden_lora":
            raise ValueError(f"{run_dir}: wrong A-LoRA variant")
        if run_args.get("affine_rank") != 50:
            raise ValueError(f"{run_dir}: affine_rank is not 50")
        if float(run_args.get("affine_alpha", -1)) != float(50 * scale):
            raise ValueError(f"{run_dir}: affine_alpha does not match scale {scale}")
        if float(run_args.get("affine_dropout", -1)) != 0.0:
            raise ValueError(f"{run_dir}: affine boundary dropout is not zero")
        if run_args.get("affine_lm_head_bias") is not False:
            raise ValueError(f"{run_dir}: affine lm_head bias is not disabled")

        affine_config_path = run_dir / "affine_vocab_config.json"
        affine_config = read_json(affine_config_path)
        affine_keys = inspect_safetensors(
            run_dir / "affine_vocab_adapter.safetensors"
        )
        expected_affine = {
            "rank": 50,
            "alpha": float(50 * scale),
            "dropout": 0.0,
            "use_input": False,
            "use_lm_head": True,
            "use_lm_head_bias": False,
            "tie_input_lm_head_adapters": False,
        }
        for field, expected in expected_affine.items():
            if affine_config.get(field) != expected:
                raise ValueError(
                    f"{affine_config_path}: {field}={affine_config.get(field)!r}, "
                    f"expected {expected!r}"
                )
        if not any("lm_head.affine" in key for key in affine_keys):
            raise ValueError(f"{run_dir}: no lm_head affine tensors in checkpoint")
        if any("embed_tokens.affine" in key for key in affine_keys):
            raise ValueError(f"{run_dir}: unexpected input affine tensors")
        if trainable_summary.get("trainable") != 9_385_984:
            raise ValueError(f"{run_dir}: wrong A-LoRA trainable parameter count")
        validate_affine_optimizer_metadata(run_dir, run_args, boundary_lr)
    else:
        if run_args.get("variant") != "hidden_lora":
            raise ValueError(f"{run_dir}: wrong Vocab-LoRA variant")
        if run_args.get("include_emb_lmh_lora_rank") != 1:
            raise ValueError(f"{run_dir}: Vocab-LoRA boundary rank is not 1")
        if run_args.get("emb_lmh_lora_alpha") != scale:
            raise ValueError(f"{run_dir}: Vocab-LoRA alpha does not match scale {scale}")
        rank_pattern = adapter_config.get("rank_pattern", {})
        alpha_pattern = adapter_config.get("alpha_pattern", {})
        target_modules = set(adapter_config.get("target_modules", []))
        if rank_pattern.get("lm_head") != 1:
            raise ValueError(f"{adapter_config_path}: lm_head rank pattern is not 1")
        if alpha_pattern.get("lm_head") != scale:
            raise ValueError(
                f"{adapter_config_path}: lm_head alpha does not match scale {scale}"
            )
        if "lm_head" not in target_modules or "embed_tokens" in target_modules:
            raise ValueError(
                f"{adapter_config_path}: boundary target is not output-only lm_head"
            )
        if not any("lm_head.lora_A" in key for key in adapter_keys):
            raise ValueError(f"{run_dir}: missing lm_head LoRA A tensor")
        if not any("lm_head.lora_B" in key for key in adapter_keys):
            raise ValueError(f"{run_dir}: missing lm_head LoRA B tensor")
        if any("embed_tokens" in key and "lora_" in key for key in adapter_keys):
            raise ValueError(f"{run_dir}: unexpected input-embedding LoRA tensor")
        if trainable_summary.get("trainable") != 9_385_856:
            raise ValueError(f"{run_dir}: wrong Vocab-LoRA trainable parameter count")
        require_close(
            f"{run_dir}: run_args.affine_learning_rate_scale",
            run_args.get("affine_learning_rate_scale"),
            1.0,
        )
        require_close(
            f"{run_dir}: run_args.affine_bias_learning_rate_scale",
            run_args.get("affine_bias_learning_rate_scale"),
            1.0,
        )
        validate_vocab_optimizer_audit(
            run_dir,
            run_args,
            boundary_lr,
            required=boundary_lr != 1.0,
        )

    return run_args, method, scale, boundary_lr, seed


def canonical_rows() -> list[dict[str, Any]]:
    rows = [dict(row) for row in load_dataset("google/IFEval", split="train")]
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(
            f"Canonical Google IFEval split has {len(rows)} rows, "
            f"expected exactly {EXPECTED_COUNT}"
        )
    keys = [row["key"] for row in rows]
    if len(set(keys)) != EXPECTED_COUNT:
        raise ValueError("Canonical Google IFEval split contains duplicate keys")
    return rows


def validate_responses(
    *,
    path: Path,
    run_dir: Path,
    start_index: int,
    end_index: int,
    expected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    run_args, _, _, _, seed = validate_checkpoint(run_dir)
    if not 0 <= start_index <= end_index <= EXPECTED_COUNT:
        raise ValueError(f"Invalid expected interval [{start_index}, {end_index})")
    if expected is None:
        expected = canonical_rows()

    rows = read_jsonl(path)
    if len(rows) != end_index - start_index:
        raise ValueError(
            f"{path}: found {len(rows)} records for expected interval "
            f"[{start_index}, {end_index})"
        )
    expected_run_dir = str(run_dir)
    seen_keys: set[Any] = set()
    for offset, (row, reference) in enumerate(
        zip(rows, expected[start_index:end_index], strict=True)
    ):
        global_index = start_index + offset
        missing = REQUIRED_RESPONSE_FIELDS - row.keys()
        if missing:
            raise ValueError(
                f"{path}: row {global_index} is missing fields {sorted(missing)}"
            )
        for field in ("key", "prompt", "instruction_id_list", "kwargs"):
            if row[field] != reference[field]:
                raise ValueError(
                    f"{path}: row {global_index} has mismatched {field}"
                )
        if not isinstance(row["response"], str):
            raise ValueError(f"{path}: row {global_index} response is not a string")
        if row["run_dir"] != expected_run_dir:
            raise ValueError(
                f"{path}: row {global_index} run_dir={row['run_dir']!r}, "
                f"expected {expected_run_dir!r}"
            )
        if row["variant"] != run_args["variant"]:
            raise ValueError(f"{path}: row {global_index} variant mismatch")
        if row["seed"] != seed:
            raise ValueError(f"{path}: row {global_index} seed mismatch")
        if row["affine_ablation"] != "none":
            raise ValueError(
                f"{path}: row {global_index} is an ablation, not the fixed model"
            )
        if row["key"] in seen_keys:
            raise ValueError(f"{path}: duplicate key {row['key']!r}")
        seen_keys.add(row["key"])
    return rows


def validate_scores(
    response_path: Path, score_dir: Path, run_dir: Path
) -> None:
    expected = canonical_rows()
    responses = validate_responses(
        path=response_path,
        run_dir=run_dir,
        start_index=0,
        end_index=EXPECTED_COUNT,
        expected=expected,
    )
    for mode in ("strict", "loose"):
        path = score_dir / f"eval_results_{mode}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != EXPECTED_COUNT:
            raise ValueError(
                f"{path}: found {len(rows)} scorer rows; expected {EXPECTED_COUNT}"
            )
        for index, (row, response) in enumerate(zip(rows, responses, strict=True)):
            for field in ("prompt", "response", "instruction_id_list"):
                if row.get(field) != response[field]:
                    raise ValueError(f"{path}: scorer row {index} mismatched {field}")
            follow_all = row.get("follow_all_instructions")
            follow_list = row.get("follow_instruction_list")
            if type(follow_all) is not bool:  # bool, not truthy integer/string
                raise ValueError(
                    f"{path}: scorer row {index} has non-boolean aggregate result"
                )
            if not isinstance(follow_list, list) or not all(
                type(item) is bool for item in follow_list
            ):
                raise ValueError(
                    f"{path}: scorer row {index} has malformed instruction results"
                )
            if len(follow_list) != len(response["instruction_id_list"]):
                raise ValueError(
                    f"{path}: scorer row {index} instruction-result count mismatch"
                )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.command == "checkpoint":
        validate_checkpoint(run_dir)
        print(f"valid checkpoint: {run_dir}")
    elif args.command == "shard":
        path = Path(args.path).resolve()
        validate_responses(
            path=path,
            run_dir=run_dir,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        print(
            f"valid shard: {path} [{args.start_index}, {args.end_index})"
        )
    elif args.command == "merged":
        path = Path(args.path).resolve()
        validate_responses(
            path=path,
            run_dir=run_dir,
            start_index=0,
            end_index=EXPECTED_COUNT,
        )
        print(f"valid merged responses: {path}")
    elif args.command == "scores":
        response_path = Path(args.response_path).resolve()
        score_dir = Path(args.score_dir).resolve()
        validate_scores(response_path, score_dir, run_dir)
        print(f"valid official scores: {score_dir}")
    else:  # argparse enforces a known subcommand.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
