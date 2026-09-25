#!/usr/bin/env python
"""Independent diagnostics for the output-only equal-budget IFEval study.

This script complements the final prompt-level IFEval summary with:

* response-length, empty-response, and cutoff-signal diagnostics;
* paired strict instruction-level results grouped by the top-level IFEval
  family (the prefix before ``:`` in an instruction id);
* short, key-addressable examples of strict A-only and Vocab-only prompts.

The default invocation is fail-closed: it requires exactly the eight frozen
runs (two methods x seeds 42--45), validates every response against the
canonical 541-row Google IFEval split, and validates both official strict and
loose score files.  ``--seeds`` exists only so a completed seed pair can be
used for development-time structural checks while the other runs are still
being generated.

No instruction-level row is treated as an independent training replicate.
Family totals pool nested instruction checks descriptively; seed 42 is kept
separate from the independently trained seeds 43--45.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import summarize_output_only_equal_budget_qwen25_confirmation as confirmation  # noqa: E402
import validate_output_only_ifeval as artifact_validation  # noqa: E402


DEFAULT_EXPERIMENT = ROOT / "reviewer_followup/output_only_equal_budget_qwen25"
EXPECTED_PROMPTS = 541
ALL_SEEDS = (42, 43, 44, 45)
PRIMARY_SEEDS = (43, 44, 45)
METHODS = ("aff_r50", "vocab_r1")
MODES = ("strict", "loose")
SCORE_FIELDS = {
    "follow_all_instructions",
    "follow_instruction_list",
    "instruction_id_list",
    "prompt",
    "response",
}
TOKEN_COUNT_FIELDS = (
    "num_generated_tokens",
    "generated_token_count",
    "generation_token_count",
    "output_token_count",
    "completion_tokens",
)
TOKEN_LIST_FIELDS = ("generated_tokens", "output_tokens")
LENGTH_FINISH_REASONS = {
    "length",
    "max_length",
    "max_tokens",
    "max_new_tokens",
    "token_limit",
}
TERMINAL_RE = re.compile(r"""[.!?。！？…;；:：)\]}'"”’》」】）`>]$""")


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
        "--seeds",
        default="42,43,44,45",
        help=(
            "Comma-separated completed seed pairs to validate. The default "
            "requires the exact final eight-run artifact set."
        ),
    )
    parser.add_argument(
        "--representatives",
        type=int,
        default=6,
        help="Maximum short A-only/V-only prompt examples per seed group.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation cap used only to interpret explicit token metadata.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ifeval_diagnostics.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/ifeval_diagnostics.md.",
    )
    args = parser.parse_args()
    if args.representatives < 0:
        parser.error("--representatives must be non-negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --seeds value {raw!r}") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicates: {seeds}")
    if any(seed not in ALL_SEEDS for seed in seeds):
        raise ValueError(
            f"--seeds must be a subset of {ALL_SEEDS}; received {seeds}"
        )
    return tuple(sorted(seeds))


def read_score_rows(
    path: Path,
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = artifact_validation.read_jsonl(path)
    if len(rows) != EXPECTED_PROMPTS:
        raise ValueError(
            f"{path}: found {len(rows)} score rows; expected {EXPECTED_PROMPTS}"
        )
    for index, (row, response) in enumerate(zip(rows, responses, strict=True)):
        if set(row) != SCORE_FIELDS:
            raise ValueError(
                f"{path}: row {index} fields={sorted(row)}, "
                f"expected exactly {sorted(SCORE_FIELDS)}"
            )
        for field in ("prompt", "response", "instruction_id_list"):
            if row[field] != response[field]:
                raise ValueError(f"{path}: row {index} mismatched {field}")
        follow_all = row["follow_all_instructions"]
        follow_list = row["follow_instruction_list"]
        if type(follow_all) is not bool:
            raise ValueError(f"{path}: row {index} aggregate score is not bool")
        if not isinstance(follow_list, list) or not all(
            type(value) is bool for value in follow_list
        ):
            raise ValueError(
                f"{path}: row {index} instruction scores are not booleans"
            )
        if len(follow_list) != len(row["instruction_id_list"]):
            raise ValueError(
                f"{path}: row {index} instruction score count mismatch"
            )
        if not follow_list:
            raise ValueError(f"{path}: row {index} has no instruction checks")
        if follow_all != all(follow_list):
            raise ValueError(
                f"{path}: row {index} aggregate score disagrees with "
                "per-instruction scores"
            )
    return rows


def exact_expected_artifacts(
    ifeval_dir: Path,
    expected_names: set[str],
    *,
    final_eight: bool,
) -> None:
    response_dir = ifeval_dir / "responses"
    score_root = ifeval_dir / "scores"
    if not response_dir.is_dir():
        raise FileNotFoundError(f"Missing response directory: {response_dir}")
    if not score_root.is_dir():
        raise FileNotFoundError(f"Missing score directory: {score_root}")

    if final_eight:
        actual_responses = {path.stem for path in response_dir.glob("*.jsonl")}
        if actual_responses != expected_names:
            raise ValueError(
                "Final response set is not exactly the frozen eight runs: "
                f"missing={sorted(expected_names - actual_responses)}, "
                f"unexpected={sorted(actual_responses - expected_names)}"
            )
        actual_score_dirs = {
            path.name for path in score_root.iterdir() if path.is_dir()
        }
        if actual_score_dirs != expected_names:
            raise ValueError(
                "Final score-directory set is not exactly the frozen eight runs: "
                f"missing={sorted(expected_names - actual_score_dirs)}, "
                f"unexpected={sorted(actual_score_dirs - expected_names)}"
            )

    expected_score_files = {
        "eval_results_strict.jsonl",
        "eval_results_loose.jsonl",
    }
    for name in sorted(expected_names):
        response = response_dir / f"{name}.jsonl"
        score_dir = score_root / name
        if not response.is_file():
            raise FileNotFoundError(f"Missing response artifact: {response}")
        if not score_dir.is_dir():
            raise FileNotFoundError(f"Missing score directory: {score_dir}")
        files = {path.name for path in score_dir.iterdir() if path.is_file()}
        if files != expected_score_files:
            raise ValueError(
                f"{score_dir}: expected exactly {sorted(expected_score_files)}, "
                f"found {sorted(files)}"
            )


def percentile(values: list[int], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def describe_counts(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot describe an empty count vector")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def optional_generated_token_count(row: dict[str, Any]) -> int | None:
    candidates: list[tuple[str, int]] = []
    for field in TOKEN_COUNT_FIELDS:
        if field not in row:
            continue
        value = row[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"Malformed {field} token count: {value!r}")
        candidates.append((field, value))
    for field in TOKEN_LIST_FIELDS:
        if field not in row:
            continue
        value = row[field]
        if not isinstance(value, list):
            raise ValueError(f"Malformed {field} token list: {type(value).__name__}")
        candidates.append((field, len(value)))
    if not candidates:
        return None
    unique = {value for _, value in candidates}
    if len(unique) != 1:
        raise ValueError(f"Disagreeing generated-token metadata: {candidates}")
    return candidates[0][1]


def explicit_cutoff_signal(
    row: dict[str, Any],
    token_count: int | None,
    max_new_tokens: int,
) -> bool | None:
    observed = False
    cutoff = False
    for field in ("truncated", "hit_max_new_tokens", "reached_max_new_tokens"):
        if field in row:
            value = row[field]
            if type(value) is not bool:
                raise ValueError(f"Malformed boolean cutoff field {field}: {value!r}")
            observed = True
            cutoff = cutoff or value
    if "finish_reason" in row:
        value = row["finish_reason"]
        if not isinstance(value, str):
            raise ValueError(f"Malformed finish_reason: {value!r}")
        observed = True
        cutoff = cutoff or value.lower() in LENGTH_FINISH_REASONS
    if token_count is not None:
        observed = True
        cutoff = cutoff or token_count >= max_new_tokens
    return cutoff if observed else None


def ends_nonterminal(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and TERMINAL_RE.search(stripped) is None


def response_length_statistics(
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[int], list[int], list[int | None]]:
    characters = [len(row["response"]) for row in rows]
    whitespace_units = [
        len(re.findall(r"\S+", row["response"])) for row in rows
    ]
    token_counts = [optional_generated_token_count(row) for row in rows]
    present_token_counts = [value for value in token_counts if value is not None]
    if present_token_counts and len(present_token_counts) != len(rows):
        raise ValueError(
            "Generated-token metadata is present for only part of one run"
        )

    explicit_signals = [
        explicit_cutoff_signal(row, count, max_new_tokens)
        for row, count in zip(rows, token_counts, strict=True)
    ]
    observed_signals = [value for value in explicit_signals if value is not None]
    if observed_signals and len(observed_signals) != len(rows):
        raise ValueError("Cutoff metadata is present for only part of one run")

    long_threshold = percentile(characters, 0.90)
    long_nonterminal = sum(
        length >= long_threshold and ends_nonterminal(row["response"])
        for length, row in zip(characters, rows, strict=True)
    )
    empty = sum(not row["response"].strip() for row in rows)
    replacement_character_responses = sum(
        "\ufffd" in row["response"] for row in rows
    )
    result: dict[str, Any] = {
        "n_responses": len(rows),
        "unicode_codepoint_characters": describe_counts(characters),
        "whitespace_separated_units": describe_counts(whitespace_units),
        "empty_response_count": empty,
        "empty_response_rate": empty / len(rows),
        "very_short_le_10_characters_count": sum(
            length <= 10 for length in characters
        ),
        "nonterminal_ending_heuristic_count": sum(
            ends_nonterminal(row["response"]) for row in rows
        ),
        "long_p90_and_nonterminal_heuristic_count": long_nonterminal,
        "long_p90_character_threshold": long_threshold,
        "replacement_character_response_count": replacement_character_responses,
        "generated_token_count": (
            {
                "available": True,
                "field_policy": {
                    "numeric_candidates": list(TOKEN_COUNT_FIELDS),
                    "list_candidates": list(TOKEN_LIST_FIELDS),
                },
                "distribution": describe_counts(present_token_counts),
            }
            if present_token_counts
            else {
                "available": False,
                "reason": (
                    "response artifacts contain neither generated-token counts "
                    "nor generated-token id lists"
                ),
            }
        ),
        "explicit_cutoff_signal": (
            {
                "observable": True,
                "count": sum(observed_signals),
                "rate": sum(observed_signals) / len(observed_signals),
                "max_new_tokens_interpretation": max_new_tokens,
            }
            if observed_signals
            else {
                "observable": False,
                "count": None,
                "rate": None,
                "reason": (
                    "no token counts, finish_reason, or explicit truncation "
                    "flags were recorded; text-ending heuristics are not proof "
                    "of max-token truncation"
                ),
            }
        ),
    }
    return result, characters, whitespace_units, token_counts


def paired_length_statistics(
    aff: tuple[list[int], list[int], list[int | None]],
    vocab: tuple[list[int], list[int], list[int | None]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels = ("characters", "whitespace_separated_units")
    for label, aff_values, vocab_values in zip(
        labels, aff[:2], vocab[:2], strict=True
    ):
        if len(aff_values) != EXPECTED_PROMPTS or len(vocab_values) != (
            EXPECTED_PROMPTS
        ):
            raise ValueError("Paired length vectors are not 541 rows")
        deltas = [
            a - v for a, v in zip(aff_values, vocab_values, strict=True)
        ]
        result[label] = {
            "mean_delta_aff_minus_vocab": statistics.fmean(deltas),
            "median_delta_aff_minus_vocab": statistics.median(deltas),
            "aff_longer_count": sum(value > 0 for value in deltas),
            "vocab_longer_count": sum(value < 0 for value in deltas),
            "equal_count": sum(value == 0 for value in deltas),
        }
    aff_tokens = aff[2]
    vocab_tokens = vocab[2]
    if all(value is not None for value in aff_tokens + vocab_tokens):
        token_deltas = [
            int(a) - int(v)
            for a, v in zip(aff_tokens, vocab_tokens, strict=True)
        ]
        result["generated_tokens"] = {
            "available": True,
            "mean_delta_aff_minus_vocab": statistics.fmean(token_deltas),
            "median_delta_aff_minus_vocab": statistics.median(token_deltas),
            "aff_longer_count": sum(value > 0 for value in token_deltas),
            "vocab_longer_count": sum(value < 0 for value in token_deltas),
            "equal_count": sum(value == 0 for value in token_deltas),
        }
    else:
        result["generated_tokens"] = {
            "available": False,
            "reason": "one or both paired artifacts lack generated-token metadata",
        }
    return result


def paired_prompt_direction_statistics(
    seeds: Iterable[int],
    score_rows: dict[tuple[str, int, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Prompt-level paired directions, kept separate by trained seed."""

    seeds = tuple(seeds)
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        per_seed[str(seed)] = {}
        for mode in MODES:
            aff = score_rows[("aff_r50", seed, mode)]
            vocab = score_rows[("vocab_r1", seed, mode)]
            if len(aff) != EXPECTED_PROMPTS or len(vocab) != EXPECTED_PROMPTS:
                raise ValueError("Prompt-direction score vectors are not 541 rows")
            aff_outcomes = [row["follow_all_instructions"] for row in aff]
            vocab_outcomes = [
                row["follow_all_instructions"] for row in vocab
            ]
            both_success = sum(
                a and v
                for a, v in zip(aff_outcomes, vocab_outcomes, strict=True)
            )
            aff_only = sum(
                a and not v
                for a, v in zip(aff_outcomes, vocab_outcomes, strict=True)
            )
            vocab_only = sum(
                not a and v
                for a, v in zip(aff_outcomes, vocab_outcomes, strict=True)
            )
            both_failure = (
                EXPECTED_PROMPTS - both_success - aff_only - vocab_only
            )
            aff_success = both_success + aff_only
            vocab_success = both_success + vocab_only
            per_seed[str(seed)][mode] = {
                "aff_success": aff_success,
                "vocab_success": vocab_success,
                "aff_accuracy": aff_success / EXPECTED_PROMPTS,
                "vocab_accuracy": vocab_success / EXPECTED_PROMPTS,
                "delta_accuracy_aff_minus_vocab": (
                    aff_success - vocab_success
                )
                / EXPECTED_PROMPTS,
                "paired_prompt_cells": {
                    "both_success": both_success,
                    "aff_only_success": aff_only,
                    "vocab_only_success": vocab_only,
                    "both_failure": both_failure,
                },
            }

    primary_seeds = tuple(seed for seed in PRIMARY_SEEDS if seed in seeds)
    primary_descriptive: dict[str, Any] = {}
    if primary_seeds:
        for mode in MODES:
            deltas = [
                per_seed[str(seed)][mode][
                    "delta_accuracy_aff_minus_vocab"
                ]
                for seed in primary_seeds
            ]
            primary_descriptive[mode] = {
                "seeds": list(primary_seeds),
                "mean_seed_delta_accuracy_aff_minus_vocab": (
                    statistics.fmean(deltas)
                ),
                "aff_better_seed_count": sum(value > 0 for value in deltas),
                "vocab_better_seed_count": sum(value < 0 for value in deltas),
                "tied_seed_count": sum(value == 0 for value in deltas),
                "role": (
                    "descriptive cross-seed direction only; formal seed-level "
                    "uncertainty is reported in ifeval_summary.json"
                ),
            }
    return {
        "unit": "paired prompt within each separately trained seed",
        "per_seed": per_seed,
        "independent_seeds_43_45_descriptive": primary_descriptive,
        "selected_seed42_is_separate": 42 in seeds,
    }


def instruction_family(instruction_id: Any) -> str:
    if not isinstance(instruction_id, str) or not instruction_id:
        raise ValueError(f"Malformed IFEval instruction id: {instruction_id!r}")
    family, separator, subtype = instruction_id.partition(":")
    if not separator or not family or not subtype:
        raise ValueError(
            f"Instruction id lacks expected family:subtype form: {instruction_id!r}"
        )
    return family


def empty_family_counter() -> dict[str, Any]:
    return {
        "observations": 0,
        "aff_success": 0,
        "vocab_success": 0,
        "both_success": 0,
        "aff_only_success": 0,
        "vocab_only_success": 0,
        "both_failure": 0,
        "unique_instruction_slots": set(),
        "unique_prompts": set(),
    }


def family_statistics_for_seeds(
    seeds: Iterable[int],
    strict_rows: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    seeds = tuple(seeds)
    pooled: defaultdict[str, dict[str, Any]] = defaultdict(empty_family_counter)
    per_seed_raw: dict[int, defaultdict[str, dict[str, Any]]] = {
        seed: defaultdict(empty_family_counter) for seed in seeds
    }
    for seed in seeds:
        aff_rows = strict_rows[("aff_r50", seed)]
        vocab_rows = strict_rows[("vocab_r1", seed)]
        for prompt_index, (aff_row, vocab_row) in enumerate(
            zip(aff_rows, vocab_rows, strict=True)
        ):
            if aff_row["instruction_id_list"] != vocab_row["instruction_id_list"]:
                raise ValueError(
                    f"Seed {seed}, prompt {prompt_index}: instruction ids differ"
                )
            for instruction_index, (
                instruction_id,
                aff_success,
                vocab_success,
            ) in enumerate(
                zip(
                    aff_row["instruction_id_list"],
                    aff_row["follow_instruction_list"],
                    vocab_row["follow_instruction_list"],
                    strict=True,
                )
            ):
                family = instruction_family(instruction_id)
                key = (prompt_index, instruction_index)
                prompt_slot = prompt_index
                for target in (pooled[family], per_seed_raw[seed][family]):
                    target["observations"] += 1
                    target["aff_success"] += int(aff_success)
                    target["vocab_success"] += int(vocab_success)
                    target["both_success"] += int(aff_success and vocab_success)
                    target["aff_only_success"] += int(
                        aff_success and not vocab_success
                    )
                    target["vocab_only_success"] += int(
                        not aff_success and vocab_success
                    )
                    target["both_failure"] += int(
                        not aff_success and not vocab_success
                    )
                    target["unique_instruction_slots"].add(key)
                    target["unique_prompts"].add(prompt_slot)

    def finalize(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        final: dict[str, dict[str, Any]] = {}
        for family, counts in sorted(raw.items()):
            total = counts["observations"]
            if (
                counts["both_success"]
                + counts["aff_only_success"]
                + counts["vocab_only_success"]
                + counts["both_failure"]
                != total
            ):
                raise AssertionError(f"{family}: paired cells do not sum")
            final[family] = {
                "instruction_observations": total,
                "unique_instruction_slots": len(
                    counts["unique_instruction_slots"]
                ),
                "unique_prompts": len(counts["unique_prompts"]),
                "aff_success": counts["aff_success"],
                "vocab_success": counts["vocab_success"],
                "aff_success_rate": counts["aff_success"] / total,
                "vocab_success_rate": counts["vocab_success"] / total,
                "delta_success_rate_aff_minus_vocab": (
                    counts["aff_success"] - counts["vocab_success"]
                )
                / total,
                "paired_instruction_cells": {
                    "both_success": counts["both_success"],
                    "aff_only_success": counts["aff_only_success"],
                    "vocab_only_success": counts["vocab_only_success"],
                    "both_failure": counts["both_failure"],
                },
            }
        return final

    per_seed = {
        str(seed): finalize(per_seed_raw[seed]) for seed in seeds
    }
    pooled_final = finalize(pooled)
    for family, row in pooled_final.items():
        row["per_seed_delta_success_rate_aff_minus_vocab"] = {
            str(seed): per_seed[str(seed)][family][
                "delta_success_rate_aff_minus_vocab"
            ]
            for seed in seeds
        }
    return {
        "seeds": list(seeds),
        "aggregation_unit": (
            "instruction checks nested within prompts and trained seeds; "
            "descriptive only, not independent seed-level inference"
        ),
        "family_definition": "prefix before ':' in instruction_id",
        "pooled_descriptive": pooled_final,
        "per_seed": per_seed,
    }


def short_excerpt(text: str, limit: int = 160) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def representative_prompt_analysis(
    seeds: Iterable[int],
    strict_rows: dict[tuple[str, int], list[dict[str, Any]]],
    canonical_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    seeds = tuple(seeds)
    by_key: dict[Any, dict[str, Any]] = {}
    event_family_counts = {
        "aff_only": Counter(),
        "vocab_only": Counter(),
    }
    for index, canonical in enumerate(canonical_rows):
        instruction_ids = canonical["instruction_id_list"]
        families = sorted({instruction_family(item) for item in instruction_ids})
        by_key[canonical["key"]] = {
            "key": canonical["key"],
            "instruction_ids": list(instruction_ids),
            "families": families,
            "prompt_excerpt": short_excerpt(canonical["prompt"]),
            "aff_only_seeds": [],
            "vocab_only_seeds": [],
            "both_success_seeds": [],
            "both_failure_seeds": [],
        }
        for seed in seeds:
            aff = strict_rows[("aff_r50", seed)][index][
                "follow_all_instructions"
            ]
            vocab = strict_rows[("vocab_r1", seed)][index][
                "follow_all_instructions"
            ]
            if aff and not vocab:
                direction = "aff_only"
            elif not aff and vocab:
                direction = "vocab_only"
            elif aff and vocab:
                direction = "both_success"
            else:
                direction = "both_failure"
            by_key[canonical["key"]][f"{direction}_seeds"].append(seed)
            if direction in event_family_counts:
                # A prompt can contain multiple families. Count each family
                # once per directional seed-prompt event (multi-label).
                event_family_counts[direction].update(families)

    def representatives(direction: str) -> list[dict[str, Any]]:
        opposite = "vocab_only" if direction == "aff_only" else "aff_only"
        candidates = [
            item for item in by_key.values() if item[f"{direction}_seeds"]
        ]
        candidates.sort(
            key=lambda item: (
                -len(item[f"{direction}_seeds"]),
                len(item[f"{opposite}_seeds"]),
                str(item["key"]),
            )
        )
        rows: list[dict[str, Any]] = []
        for item in candidates[:limit]:
            rows.append(
                {
                    "key": item["key"],
                    "directional_seeds": item[f"{direction}_seeds"],
                    "opposite_direction_seeds": item[f"{opposite}_seeds"],
                    "instruction_ids": item["instruction_ids"],
                    "families": item["families"],
                    "prompt_excerpt": item["prompt_excerpt"],
                }
            )
        return rows

    return {
        "seeds": list(seeds),
        "metric": "strict prompt-level follow_all_instructions",
        "selection_rule": (
            "rank by number of directional seeds, then fewer opposite-direction "
            "seeds, then key; prompt excerpts are capped at 160 characters"
        ),
        "directional_prompt_event_family_counts_multilabel": {
            direction: dict(sorted(counter.items()))
            for direction, counter in event_family_counts.items()
        },
        "aff_only_representatives": representatives("aff_only"),
        "vocab_only_representatives": representatives("vocab_only"),
    }


def collect(
    *,
    experiment_dir: Path,
    selection_path: Path,
    ifeval_dir: Path,
    seeds: tuple[int, ...],
    representatives: int,
    max_new_tokens: int,
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
        for seed in seeds
    }
    final_eight = seeds == ALL_SEEDS
    if final_eight and len(expected_names) != 8:
        raise AssertionError("Final expected run-name set is not eight")
    exact_expected_artifacts(
        ifeval_dir,
        expected_names,
        final_eight=final_eight,
    )

    canonical_rows = artifact_validation.canonical_rows()
    if len(canonical_rows) != EXPECTED_PROMPTS:
        raise ValueError(
            f"Canonical IFEval has {len(canonical_rows)} rows, "
            f"expected {EXPECTED_PROMPTS}"
        )
    canonical_keys = [row["key"] for row in canonical_rows]
    if len(set(canonical_keys)) != EXPECTED_PROMPTS:
        raise ValueError("Canonical IFEval keys are not unique")

    response_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    score_rows: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    run_names: dict[tuple[str, int], str] = {}
    length_summary: dict[str, dict[str, Any]] = {}
    raw_lengths: dict[
        tuple[str, int], tuple[list[int], list[int], list[int | None]]
    ] = {}

    for seed in seeds:
        length_summary[str(seed)] = {}
        for method in METHODS:
            selected_method = selected[method]
            name = confirmation.run_name(
                method,
                int(selected_method["scale"]),
                float(selected_method["boundary_lr_scale"]),
                seed,
            )
            run_names[(method, seed)] = name
            run_dir = experiment_dir / "checkpoints" / name
            response_path = ifeval_dir / "responses" / f"{name}.jsonl"
            rows = artifact_validation.validate_responses(
                path=response_path,
                run_dir=run_dir,
                start_index=0,
                end_index=EXPECTED_PROMPTS,
                expected=canonical_rows,
            )
            response_rows[(method, seed)] = rows
            stats, characters, words, tokens = response_length_statistics(
                rows, max_new_tokens=max_new_tokens
            )
            length_summary[str(seed)][method] = stats
            raw_lengths[(method, seed)] = (characters, words, tokens)
            for mode in MODES:
                score_path = (
                    ifeval_dir
                    / "scores"
                    / name
                    / f"eval_results_{mode}.jsonl"
                )
                score_rows[(method, seed, mode)] = read_score_rows(
                    score_path, rows
                )

        # Pair identity is checked again independently of canonical alignment.
        for index, (aff, vocab) in enumerate(
            zip(
                response_rows[("aff_r50", seed)],
                response_rows[("vocab_r1", seed)],
                strict=True,
            )
        ):
            for field in ("key", "prompt", "instruction_id_list", "kwargs"):
                if aff[field] != vocab[field]:
                    raise ValueError(
                        f"Seed {seed}, row {index}: A/V mismatched {field}"
                    )
        length_summary[str(seed)]["paired_aff_minus_vocab"] = (
            paired_length_statistics(
                raw_lengths[("aff_r50", seed)],
                raw_lengths[("vocab_r1", seed)],
            )
        )

    strict = {
        (method, seed): score_rows[(method, seed, "strict")]
        for method in METHODS
        for seed in seeds
    }
    primary_present = tuple(seed for seed in PRIMARY_SEEDS if seed in seeds)
    seed42_present = (42,) if 42 in seeds else ()
    family_analysis: dict[str, Any] = {}
    representatives_by_group: dict[str, Any] = {}
    if primary_present:
        family_analysis["independent_seeds_43_45"] = (
            family_statistics_for_seeds(primary_present, strict)
        )
        representatives_by_group["independent_seeds_43_45"] = (
            representative_prompt_analysis(
                primary_present,
                strict,
                canonical_rows,
                limit=representatives,
            )
        )
    if seed42_present:
        family_analysis["selected_seed42"] = family_statistics_for_seeds(
            seed42_present, strict
        )
        representatives_by_group["selected_seed42"] = (
            representative_prompt_analysis(
                seed42_present,
                strict,
                canonical_rows,
                limit=representatives,
            )
        )

    return {
        "experiment": str(experiment_dir.resolve()),
        "ifeval_dir": str(ifeval_dir.resolve()),
        "final_selection": str(selection_path.resolve()),
        "selected_hyperparameters": selected,
        "requested_seeds": list(seeds),
        "scope": (
            "final_exact_eight_runs"
            if final_eight
            else "development_partial_seed_pair_check"
        ),
        "validation": {
            "status": "passed",
            "canonical_prompts": EXPECTED_PROMPTS,
            "canonical_pair_alignment": "passed",
            "methods": list(METHODS),
            "seeds": list(seeds),
            "runs_validated": 2 * len(seeds),
            "response_artifacts_validated": 2 * len(seeds),
            "strict_score_artifacts_validated": 2 * len(seeds),
            "loose_score_artifacts_validated": 2 * len(seeds),
            "exact_final_artifact_set_required_and_passed": final_eight,
            "run_names": {
                f"{method}_seed{seed}": run_names[(method, seed)]
                for seed in seeds
                for method in METHODS
            },
        },
        "response_length_and_cutoff_diagnostics": {
            "character_definition": "Python Unicode code points via len(text)",
            "word_proxy_definition": "non-empty whitespace-separated units",
            "max_new_tokens_used_by_launcher": max_new_tokens,
            "important_limitation": (
                "The current generator does not record generated-token counts "
                "or finish reasons. Therefore max-token truncation is not "
                "directly observable; nonterminal-ending counts are heuristic "
                "signals only."
            ),
            "per_seed": length_summary,
        },
        "paired_prompt_directions": paired_prompt_direction_statistics(
            seeds, score_rows
        ),
        "strict_instruction_family_analysis": {
            "important_limitation": (
                "Rows are instruction checks nested within prompts and trained "
                "seeds. Pooled family counts are descriptive and must not be "
                "presented as independent-seed significance."
            ),
            **family_analysis,
        },
        "strict_directional_prompt_examples": representatives_by_group,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def render_family_table(group: dict[str, Any]) -> list[str]:
    rows = group["pooled_descriptive"]
    seeds = group["seeds"]
    lines = [
        "| Family | Unique slots | Obs. | A success | Vocab success | "
        "A−V (pp) | A-only / V-only | Per-seed A−V (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family, row in rows.items():
        cells = row["paired_instruction_cells"]
        per_seed = ", ".join(
            f"{seed}:{pp(row['per_seed_delta_success_rate_aff_minus_vocab'][str(seed)])}"
            for seed in seeds
        )
        lines.append(
            f"| `{family}` | {row['unique_instruction_slots']} | "
            f"{row['instruction_observations']} | "
            f"{row['aff_success']} ({pct(row['aff_success_rate'])}) | "
            f"{row['vocab_success']} ({pct(row['vocab_success_rate'])}) | "
            f"{pp(row['delta_success_rate_aff_minus_vocab'])} | "
            f"{cells['aff_only_success']} / {cells['vocab_only_success']} | "
            f"{per_seed} |"
        )
    return lines


def render_representatives(group: dict[str, Any], direction: str) -> list[str]:
    key = f"{direction}_representatives"
    label = "A-only" if direction == "aff_only" else "Vocab-only"
    lines = [
        f"#### {label} representative prompts",
        "",
        "| Key | Directional seeds | Opposite seeds | Families | Prompt excerpt |",
        "|---:|---|---|---|---|",
    ]
    rows = group[key]
    if not rows:
        lines.append("| — | — | — | — | No directional prompt found |")
        return lines
    for row in rows:
        directional = ",".join(str(seed) for seed in row["directional_seeds"])
        opposite = ",".join(
            str(seed) for seed in row["opposite_direction_seeds"]
        ) or "—"
        families = ", ".join(f"`{value}`" for value in row["families"])
        excerpt = (
            row["prompt_excerpt"]
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        lines.append(
            f"| {row['key']} | {directional} | {opposite} | "
            f"{families} | {excerpt} |"
        )
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    length = summary["response_length_and_cutoff_diagnostics"]
    directions = summary["paired_prompt_directions"]
    families = summary["strict_instruction_family_analysis"]
    examples = summary["strict_directional_prompt_examples"]
    lines = [
        "# Output-only equal-budget IFEval diagnostics",
        "",
        f"Scope: `{summary['scope']}`. Validation passed for "
        f"{validation['runs_validated']} runs × {EXPECTED_PROMPTS} canonical "
        "paired prompts, including both strict and loose official score files.",
        "",
        "## Response length and cutoff signals",
        "",
        "The current response artifacts do **not** contain generated-token "
        "counts or finish reasons. Empty responses are directly observable; "
        "nonterminal endings and long nonterminal endings are only heuristics, "
        "not proof that the 512-token cap was reached.",
        "",
        "| Seed | Method | Mean chars | Median chars | P95 chars | "
        "Mean whitespace units | Empty | Nonterminal | Long-P90 nonterminal | "
        "Replacement-char responses |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in validation["seeds"]:
        seed_rows = length["per_seed"][str(seed)]
        for method in METHODS:
            row = seed_rows[method]
            chars = row["unicode_codepoint_characters"]
            words = row["whitespace_separated_units"]
            lines.append(
                f"| {seed} | `{method}` | {chars['mean']:.1f} | "
                f"{chars['median']:.1f} | {chars['p95']:.1f} | "
                f"{words['mean']:.1f} | {row['empty_response_count']} | "
                f"{row['nonterminal_ending_heuristic_count']} | "
                f"{row['long_p90_and_nonterminal_heuristic_count']} | "
                f"{row['replacement_character_response_count']} |"
            )
    lines.extend(
        [
            "",
            "### Paired response-length differences",
            "",
            "| Seed | Mean char A−V | Median char A−V | "
            "A longer / Vocab longer / equal | Mean whitespace-unit A−V |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in validation["seeds"]:
        pair = length["per_seed"][str(seed)]["paired_aff_minus_vocab"]
        chars = pair["characters"]
        words = pair["whitespace_separated_units"]
        lines.append(
            f"| {seed} | {chars['mean_delta_aff_minus_vocab']:+.1f} | "
            f"{chars['median_delta_aff_minus_vocab']:+.1f} | "
            f"{chars['aff_longer_count']} / {chars['vocab_longer_count']} / "
            f"{chars['equal_count']} | "
            f"{words['mean_delta_aff_minus_vocab']:+.1f} |"
        )

    lines.extend(
        [
            "",
            "## Paired prompt directions",
            "",
            "These are prompt-level pairs within each trained seed. Seed 42 "
            "is selected-seed description; seeds 43–45 are the independent "
            "training-seed group.",
            "",
            "| Seed | Mode | A correct | Vocab correct | A−V (pp) | "
            "Both / A-only / Vocab-only / neither |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for seed in validation["seeds"]:
        for mode in MODES:
            row = directions["per_seed"][str(seed)][mode]
            cells = row["paired_prompt_cells"]
            lines.append(
                f"| {seed} | {mode} | {row['aff_success']} | "
                f"{row['vocab_success']} | "
                f"{pp(row['delta_accuracy_aff_minus_vocab'])} | "
                f"{cells['both_success']} / {cells['aff_only_success']} / "
                f"{cells['vocab_only_success']} / {cells['both_failure']} |"
            )
    primary_direction = directions[
        "independent_seeds_43_45_descriptive"
    ]
    if primary_direction:
        lines.extend(
            ["", "Independent-seed descriptive mean directions:", ""]
        )
        for mode in MODES:
            row = primary_direction[mode]
            lines.append(
                f"- {mode}: mean A−V "
                f"{pp(row['mean_seed_delta_accuracy_aff_minus_vocab'])} pp; "
                f"A better / tie / Vocab better seeds = "
                f"{row['aff_better_seed_count']} / {row['tied_seed_count']} / "
                f"{row['vocab_better_seed_count']}."
            )

    lines.extend(
        [
            "",
            "## Strict instruction-family results",
            "",
            "Family means the prefix before `:` in each IFEval instruction id. "
            "Counts below are paired instruction checks nested within prompts "
            "and seeds. They are descriptive diagnostics—not independent-seed "
            "significance tests.",
        ]
    )
    if "independent_seeds_43_45" in families:
        available = families["independent_seeds_43_45"]["seeds"]
        available_label = ", ".join(str(seed) for seed in available)
        lines.extend(
            [
                "",
                f"### Available independently trained seed(s): {available_label}",
                "",
                *render_family_table(families["independent_seeds_43_45"]),
            ]
        )
    if "selected_seed42" in families:
        lines.extend(
            [
                "",
                "### Selected seed 42 (reported separately)",
                "",
                *render_family_table(families["selected_seed42"]),
            ]
        )

    lines.extend(["", "## Strict directional prompt examples", ""])
    for group_name, label in (
        ("independent_seeds_43_45", "Available independently trained seed(s)"),
        ("selected_seed42", "Selected seed 42"),
    ):
        if group_name not in examples:
            continue
        if group_name == "independent_seeds_43_45":
            label += ": " + ", ".join(
                str(seed) for seed in examples[group_name]["seeds"]
            )
        lines.extend(
            [
                f"### {label}",
                "",
                "Prompt examples use strict `follow_all_instructions`. Excerpts "
                "are capped at 160 characters; responses are not copied.",
                "",
                *render_representatives(examples[group_name], "aff_only"),
                "",
                *render_representatives(examples[group_name], "vocab_only"),
                "",
                "Directional prompt-event family counts are multi-label "
                "(a multi-constraint prompt may contribute to several families): "
                f"`{json.dumps(examples[group_name]['directional_prompt_event_family_counts_multilabel'], ensure_ascii=False, sort_keys=True)}`.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    selection_path = (
        args.final_selection.resolve()
        if args.final_selection is not None
        else experiment_dir / "final_selection.json"
    )
    ifeval_dir = (
        args.ifeval_dir.resolve()
        if args.ifeval_dir is not None
        else experiment_dir / "ifeval_final"
    )
    seeds = parse_seeds(args.seeds)
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else experiment_dir / "ifeval_diagnostics.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else experiment_dir / "ifeval_diagnostics.md"
    )
    if seeds != ALL_SEEDS and args.output_json is None and args.output_md is None:
        suffix = "_".join(str(seed) for seed in seeds)
        output_json = experiment_dir / f"ifeval_diagnostics_partial_sd{suffix}.json"
        output_md = experiment_dir / f"ifeval_diagnostics_partial_sd{suffix}.md"

    summary = collect(
        experiment_dir=experiment_dir,
        selection_path=selection_path,
        ifeval_dir=ifeval_dir,
        seeds=seeds,
        representatives=args.representatives,
        max_new_tokens=args.max_new_tokens,
    )
    atomic_write(
        output_json,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(output_md, render_markdown(summary))
    print(
        f"Validated {2 * len(seeds)} runs x {EXPECTED_PROMPTS} prompts; "
        f"wrote {output_json} and {output_md}"
    )


if __name__ == "__main__":
    main()
