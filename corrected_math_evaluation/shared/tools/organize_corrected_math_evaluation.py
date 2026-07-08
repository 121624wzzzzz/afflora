#!/usr/bin/env python
"""Historical one-shot organizer for corrected_math_evaluation artifacts.

The script only moves files/directories within corrected_math_evaluation and
writes a migration manifest. It does not delete experiment artifacts.

This file is kept for provenance. The current organized tree has also been
cleaned; see corrected_math_evaluation/CLEANUP_MANIFEST.md.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]


MOVES: list[tuple[str, str, str]] = [
    # Shared reusable code / data.
    ("evaluate_math_full.py", "shared/evaluators/evaluate_math_full.py", "MATH full greedy evaluator"),
    ("evaluate_gsm8k_full.py", "shared/evaluators/evaluate_gsm8k_full.py", "GSM8K full greedy evaluator"),
    ("evaluate_metamath_loss.py", "shared/evaluators/evaluate_metamath_loss.py", "MetaMathQA answer-token CE evaluator"),
    ("merge_shards.py", "shared/merge/merge_math_shards.py", "MATH shard merger"),
    ("merge_gsm8k_shards.py", "shared/merge/merge_gsm8k_shards.py", "GSM8K shard merger"),
    ("compare_results.py", "shared/analysis/compare_results.py", "legacy comparison helper"),
    ("test_evaluator.py", "shared/tests/test_evaluator.py", "evaluator unit tests"),
    ("known_contaminated_indices.json", "shared/data_quality/known_contaminated_math_indices.json", "MATH contamination exclusions"),
    # Qwen3 8B lm-head AffLoRA line.
    ("reproducible_afflora_sweep", "model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep", "Qwen3-8B rank1 scale sweep with paired seeds"),
    ("afflora_small_scale_sweep", "model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep", "Qwen3-8B rank1 small-scale follow-up"),
    ("afflora_rank_scale_sweep", "model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep", "Qwen3-8B rank/scale lm-head AffLoRA sweep"),
    # Qwen3 close-size small-model line.
    ("qwen3_close_size_hidden_then_mergeable_sweep", "model_families/qwen3/close_size/hidden_then_mergeable", "Qwen3-0.6B/1.7B hidden-rank then mergeable sweep"),
    # Qwen2.5 line.
    ("qwen25_05b_hidden_rank_then_mergeable_sweep", "model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable", "Qwen2.5-0.5B hidden-rank then mergeable sweep"),
    ("small_mergeable_rank_sweep", "model_families/qwen25/small_models/hidden_mergeable_rank_sweep", "Qwen2.5 0.5B/1.5B hidden+mergeable rank sweep"),
    ("small_affine_only_rank_sweep", "model_families/qwen25/small_models/affine_only_rank_sweep", "Qwen2.5 0.5B/1.5B affine-only rank sweep"),
    # Mixed-family / legacy.
    ("small_mergeable_math_sweep", "cross_family/small_models/mergeable_math_scale_sweep", "Mixed Qwen2.5/Qwen3 small-model mergeable scale sweep"),
    ("outputs", "legacy_root_evals/outputs", "early root-level evaluation outputs"),
    ("logs", "legacy_root_evals/logs", "early root-level logs"),
]

ROOT_MARKDOWN_MOVES: list[tuple[str, str, str]] = [
    ("RESULTS.md", "legacy_root_evals/summaries/RESULTS.md", "early corrected evaluator results"),
    ("MATH8_RERUN_RESULTS.md", "legacy_root_evals/summaries/MATH8_RERUN_RESULTS.md", "8-shard MATH rerun notes"),
    ("LMHEAD_RANK_SWEEP_RESULTS.md", "legacy_root_evals/summaries/LMHEAD_RANK_SWEEP_RESULTS.md", "legacy lm-head rank sweep summary"),
    ("HIDDEN_PLUS_MERGEABLE_SMALL_MODEL_COMPARISON.md", "cross_family/small_models/HIDDEN_PLUS_MERGEABLE_SMALL_MODEL_COMPARISON.md", "small-model hidden+mergeable comparison"),
    ("README.md", "legacy_root_evals/summaries/README.legacy.md", "previous root README"),
]


def safe_move(src_rel: str, dst_rel: str, description: str) -> dict[str, str]:
    src = BASE / src_rel
    dst = BASE / dst_rel
    record = {"old": src_rel, "new": dst_rel, "description": description, "status": "missing"}
    if not src.exists():
        return record
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    record["status"] = "moved"
    return record


def list_top_level() -> list[str]:
    return sorted(p.name for p in BASE.iterdir())


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for p in path.rglob("*") if p.is_file())


def write_docs(records: list[dict[str, str]]) -> None:
    moved = [r for r in records if r["status"] == "moved"]
    manifest = {
        "schema_version": 1,
        "organized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base": str(BASE),
        "policy": {
            "shared": "Reusable evaluators, merge helpers, tests, tools, and data-quality metadata.",
            "model_families": "Experiments grouped first by model family, then by model size/experiment line.",
            "cross_family": "Experiments intentionally comparing more than one model family.",
            "legacy_root_evals": "Older root-level outputs/logs/summaries preserved without reinterpretation.",
        },
        "moves": records,
        "top_level_after": list_top_level(),
    }
    (BASE / "ORGANIZATION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# corrected_math_evaluation path migration",
        "",
        f"Organized at: {manifest['organized_at']}",
        "",
        "Initial organization moved experiment artifacts without deleting them. A later cleanup removed regenerable caches and other low-value residue; see `CLEANUP_MANIFEST.md`.",
        "",
        "| old path | new path | description |",
        "| --- | --- | --- |",
    ]
    for row in moved:
        lines.append(f"| `{row['old']}` | `{row['new']}` | {row['description']} |")
    missing = [r for r in records if r["status"] != "moved"]
    if missing:
        lines += ["", "Missing at organization time:", ""]
        for row in missing:
            lines.append(f"- `{row['old']}` ({row['description']})")
    (BASE / "PATH_MIGRATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    overview = [
        "# corrected_math_evaluation",
        "",
        "This directory contains repaired MetaMathQA-trained math SFT experiments and their full MATH/GSM8K evaluation artifacts.",
        "",
        "## Directory layout",
        "",
        "| path | contents |",
        "| --- | --- |",
        "| `shared/` | reusable evaluators, shard mergers, data-quality metadata, tests, and maintenance tools |",
        "| `model_families/qwen3/qwen3_8b/lmhead_afflora/` | Qwen3-8B hidden LoRA + lm-head AffLoRA scale/rank experiments |",
        "| `model_families/qwen3/close_size/hidden_then_mergeable/` | Qwen3-0.6B and Qwen3-1.7B hidden-rank, mergeable-rank, targeted multi-seed, and MetaMath eval-loss results |",
        "| `model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/` | Qwen2.5-0.5B hidden-rank and hr8+mergeable follow-up |",
        "| `model_families/qwen25/small_models/` | Qwen2.5-0.5B/1.5B affine-only and hidden+mergeable sweeps |",
        "| `cross_family/small_models/` | mixed Qwen2.5/Qwen3 small-model comparison sweeps |",
        "| `legacy_root_evals/` | early root-level outputs/logs/summaries kept for traceability |",
        "",
        "## Important current result files",
        "",
        "- `model_families/qwen3/close_size/hidden_then_mergeable/TARGETED_MULTISEED_ANALYSIS.md`",
        "- `model_families/qwen3/close_size/hidden_then_mergeable/outputs/metamath_eval_loss/`",
        "- `model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/HR8_AR4_ALL5SEEDS_ANALYSIS.md`",
        "- `model_families/qwen25/small_models/hidden_mergeable_rank_sweep/ANALYSIS.md`",
        "",
        "## Path migration",
        "",
        "See `PATH_MIGRATION.md` for old-to-new path mapping, `ORGANIZATION_MANIFEST.json` for a machine-readable manifest, and `CLEANUP_MANIFEST.md` for the cleanup record.",
        "",
        "Note: historical launcher scripts may still contain the old paths in comments or command examples. Use the new directory layout for future work.",
    ]
    (BASE / "README.md").write_text("\n".join(overview) + "\n", encoding="utf-8")

    index_lines = [
        "# Experiment index",
        "",
        "| experiment directory | files | notes |",
        "| --- | ---: | --- |",
    ]
    candidates = [
        "model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep",
        "model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep",
        "model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep",
        "model_families/qwen3/close_size/hidden_then_mergeable",
        "model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable",
        "model_families/qwen25/small_models/hidden_mergeable_rank_sweep",
        "model_families/qwen25/small_models/affine_only_rank_sweep",
        "cross_family/small_models/mergeable_math_scale_sweep",
        "legacy_root_evals",
    ]
    for rel in candidates:
        readme = BASE / rel / "README.md"
        note = ""
        if readme.exists():
            first = readme.read_text(encoding="utf-8", errors="ignore").splitlines()
            note = first[0].lstrip("# ").strip() if first else ""
        index_lines.append(f"| `{rel}` | {count_files(BASE / rel)} | {note} |")
    (BASE / "EXPERIMENT_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def main() -> None:
    if not BASE.is_dir():
        raise SystemExit(f"Missing directory: {BASE}")
    records = []
    for move in MOVES + ROOT_MARKDOWN_MOVES:
        records.append(safe_move(*move))
    # Keep .gitignore at root and create docs after the legacy README has moved.
    write_docs(records)
    print(json.dumps({"moved": sum(r["status"] == "moved" for r in records), "base": str(BASE)}, indent=2))


if __name__ == "__main__":
    main()
