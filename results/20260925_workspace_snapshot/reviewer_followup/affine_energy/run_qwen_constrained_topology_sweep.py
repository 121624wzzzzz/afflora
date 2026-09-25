#!/usr/bin/env python
"""Run constrained A-LoRA topology sweeps on tied Qwen3 and Qwen2.5 models."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
SOURCE = HERE / "run_constrained_topology_sweep.py"


def load_runner():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("constrained_topology_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def main() -> None:
    runner = load_runner()
    runner.MODEL_INFO = {
        "qwen3_06b": {
            "path": ROOT.parent / "models/Qwen3-0.6B-Base",
            "batch": 16,
            "accum": 1,
        },
        "qwen25_15b": {
            "path": ROOT.parent / "models/Qwen2.5-1.5B-Base",
            "batch": 16,
            "accum": 1,
        },
    }
    runner.JOBS = tuple(
        job
        for alias in runner.MODEL_INFO
        for job in (
            runner.Job(alias, "input", "affine_input_plus_hidden_lora"),
            runner.Job(alias, "output", "affine_lm_head_plus_hidden_lora"),
            runner.Job(
                alias, "shared", "affine_input_lm_head_plus_hidden_lora",
                lm_head_bias=True, tie_adapters=True,
            ),
            runner.Job(alias, "decoupled", "affine_input_lm_head_plus_hidden_lora"),
        )
    )
    runner.CHECKPOINTS = HERE / "qwen_topology_checkpoints"
    runner.OUTPUTS = HERE / "qwen_topology_outputs"
    runner.LOGS = HERE / "qwen_topology_logs"
    runner.STATE = HERE / "qwen_constrained_topology_state.json"
    runner.EVENTS = HERE / "qwen_constrained_topology_events.jsonl"
    runner.RESULTS = HERE / "QWEN_CONSTRAINED_TOPOLOGY_RESULTS.md"

    baseline_roots = {
        "qwen3_06b": (
            ROOT / "corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep",
            "qwen3_06b_hidden_hr4_seed42_full.json",
            "qwen3_06b_mergeable_ar16_s1_hr4_seed42_full.json",
            0.06029779,
        ),
        "qwen25_15b": (
            ROOT / "corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep",
            "qwen25_15b_hidden_hr4_seed42_full.json",
            "qwen25_15b_mergeable_ar16_s1_hr4_seed42_full.json",
            0.05500340,
        ),
    }

    def write_report(rhos):  # noqa: ANN001, ANN202
        lines = [
            "# Qwen constrained A-LoRA topology sweep", "", f"Updated: {runner.now()}", "",
            "All treatment rows use joint-from-scratch hidden LoRA r4 + A-LoRA r16, seed 42, tau=0.00625, lambda=100.", "",
            "| model | topology | input rho | output rho | MATH | delta vs hidden | GSM8K | delta vs hidden |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for alias in runner.MODEL_INFO:
            base_root, hidden_file, existing_file, existing_rho = baseline_roots[alias]
            bm = accuracy(base_root / "outputs/math" / hidden_file, "clean")
            bg = accuracy(base_root / "outputs/gsm8k" / hidden_file, "full")
            em = accuracy(base_root / "outputs/math" / existing_file, "clean")
            eg = accuracy(base_root / "outputs/gsm8k" / existing_file, "full")
            lines.append(f"| {alias} | hidden baseline | - | - | {bm:.4f}% | - | {bg:.4f}% | - |")
            lines.append(
                f"| {alias} | shared unconstrained | {existing_rho:.8f} | shared | "
                f"{em:.4f}% | {em-bm:+.4f} pp | {eg:.4f}% | {eg-bg:+.4f} pp |"
            )
            for job in (item for item in runner.JOBS if item.model_alias == alias):
                math_acc = accuracy(runner.full(job, "math"), "clean")
                gsm_acc = accuracy(runner.full(job, "gsm8k"), "full")
                row = rhos[job.name]
                in_rho = f"{row['input']:.8f}" if "input" in row else "-"
                out_rho = f"{row['output']:.8f}" if "output" in row else ("shared" if job.topology == "shared" else "-")
                lines.append(
                    f"| {alias} | {job.topology} constrained | {in_rho} | {out_rho} | "
                    f"{math_acc:.4f}% | {math_acc-bm:+.4f} pp | "
                    f"{gsm_acc:.4f}% | {gsm_acc-bg:+.4f} pp |"
                )
        runner.RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)

    runner.write_report = write_report
    runner.main()


if __name__ == "__main__":
    main()
