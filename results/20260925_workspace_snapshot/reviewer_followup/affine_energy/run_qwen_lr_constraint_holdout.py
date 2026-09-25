#!/usr/bin/env python
"""Five-seed holdout for the useful Qwen LR/energy candidates.

Seed 42 is reused from the completed factorial.  Seeds 43--46 are newly
trained and receive full 16-shard MATH and GSM8K evaluation.  The selected
set contains paired input LR=0.25 none/hinge controls on both model sizes,
plus the best seed-42 unconstrained input/output candidates.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
SOURCE = HERE / "run_constrained_topology_sweep.py"

CHECKPOINTS = HERE / "qwen_lr_constraint_holdout_checkpoints"
OUTPUTS = HERE / "qwen_lr_constraint_holdout_outputs"
LOGS = HERE / "qwen_lr_constraint_holdout_logs"
STATE = HERE / "qwen_lr_constraint_holdout_state.json"
EVENTS = HERE / "qwen_lr_constraint_holdout_events.jsonl"
RESULTS = HERE / "QWEN_LR_CONSTRAINT_HOLDOUT_RESULTS.md"

SEED42_CHECKPOINTS = HERE / "qwen_lr_constraint_checkpoints"
SEED42_OUTPUTS = HERE / "qwen_lr_constraint_outputs"
SEED42_LOGS = HERE / "qwen_lr_constraint_logs/training"
BASELINE_SEED42_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
BASELINE_HOLDOUT_OUTPUTS = HERE / "qwen_selected_multiseed_outputs"

SEEDS = (42, 43, 44, 45, 46)
NEW_SEEDS = (43, 44, 45, 46)
TAUS = {"input": 0.0125, "output": 0.00625}
ENERGY_LAMBDA = 100.0


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_lr_holdout_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class Candidate:
    model_alias: str
    topology: str
    lr_scale: float
    constrained: bool

    @property
    def key(self) -> str:
        constraint = "hinge" if self.constrained else "none"
        return f"{self.model_alias}:{self.topology}:{constraint}:lr{tag(self.lr_scale)}"

    @property
    def label(self) -> str:
        constraint = "hinge" if self.constrained else "none"
        return f"{self.topology}/{constraint}/lr={self.lr_scale:g}"


@dataclass(frozen=True)
class Job:
    candidate: Candidate
    seed: int

    @property
    def model_alias(self) -> str:
        return self.candidate.model_alias

    @property
    def topology(self) -> str:
        return self.candidate.topology

    @property
    def lr_scale(self) -> float:
        return self.candidate.lr_scale

    @property
    def constrained(self) -> bool:
        return self.candidate.constrained

    @property
    def variant(self) -> str:
        if self.topology == "input":
            return "affine_input_plus_hidden_lora"
        return "affine_lm_head_plus_hidden_lora"

    @property
    def name(self) -> str:
        regularizer = (
            f"energy_tau{tag(TAUS[self.topology])}_l100"
            if self.constrained else "unconstrained"
        )
        return (
            f"{self.model_alias}_{self.topology}_ar16_s1_hr4_"
            f"alr{tag(self.lr_scale)}_{regularizer}_seed{self.seed}"
        )

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINTS / self.name

    @property
    def model_path(self) -> Path:
        return MODEL_INFO[self.model_alias]["path"]


MODEL_INFO = {
    "qwen25_3b": {
        "path": ROOT.parent / "models/Qwen2.5-3B-Base",
        "batch": 8,
        "accum": 2,
    },
    "qwen3_4b": {
        "path": ROOT.parent / "models/Qwen3-4B-Base",
        "batch": 4,
        "accum": 4,
    },
}

# Longer 4B work is placed first.  The paired LR=0.25 rows directly test the
# active input hinge; the remaining rows validate the strongest/safest
# unconstrained placements from seed 42.
CANDIDATES = (
    Candidate("qwen3_4b", "input", 0.25, False),
    Candidate("qwen3_4b", "input", 0.25, True),
    Candidate("qwen3_4b", "input", 1.0, False),
    Candidate("qwen3_4b", "output", 0.5, False),
    Candidate("qwen25_3b", "input", 0.25, False),
    Candidate("qwen25_3b", "input", 0.25, True),
    Candidate("qwen25_3b", "output", 0.5, False),
)
JOBS = tuple(Job(candidate, seed) for candidate in CANDIDATES for seed in NEW_SEEDS)


def checkpoint_complete(job: Job) -> bool:
    return all(
        (job.checkpoint / filename).is_file()
        for filename in (
            "adapter_model.safetensors",
            "affine_vocab_adapter.safetensors",
            "affine_vocab_config.json",
            "run_args.json",
        )
    )


def seed42_job(candidate: Candidate) -> Job:
    return Job(candidate, 42)


def candidate_for(alias: str, topology: str, lr_scale: float, constrained: bool) -> Candidate:
    return next(
        candidate for candidate in CANDIDATES
        if candidate.model_alias == alias
        and candidate.topology == topology
        and candidate.lr_scale == lr_scale
        and candidate.constrained == constrained
    )


def accuracy(path: Path, task: str) -> float:
    key = "clean" if task == "math" else "full"
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half_width = 2.7764451052 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half_width


def fmt_mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):+.4f} +/- {statistics.stdev(values):.4f}"


def main() -> None:
    runner = load_base()
    runner.MODEL_INFO = MODEL_INFO
    runner.JOBS = JOBS
    runner.CHECKPOINTS = CHECKPOINTS
    runner.OUTPUTS = OUTPUTS
    runner.LOGS = LOGS
    runner.STATE = STATE
    runner.EVENTS = EVENTS
    runner.RESULTS = RESULTS

    def train(job: Job, gpu: int) -> bool:
        if checkpoint_complete(job):
            runner.event("training_reused", job=job.name, gpu=gpu)
            return True
        info = MODEL_INFO[job.model_alias]
        command = [
            str(runner.PYTHON), str(runner.TRAIN_SCRIPT),
            "--model-path", str(job.model_path),
            "--train-data", str(runner.TRAIN_DATA),
            "--output-dir", str(job.checkpoint),
            "--variant", job.variant,
            "--hidden-lora-rank", "4",
            "--hidden-lora-alpha", "8",
            "--hidden-lora-dropout", "0.05",
            "--affine-rank", "16",
            "--affine-alpha", "16",
            "--affine-learning-rate-scale", str(job.lr_scale),
            "--max-seq-len", "1024",
            "--per-device-train-batch-size", str(info["batch"]),
            "--gradient-accumulation-steps", str(info["accum"]),
            "--learning-rate", "2e-4",
            "--num-train-epochs", "1",
            "--logging-steps", "10",
            "--bf16",
            "--gradient-checkpointing",
            "--seed", str(job.seed),
            "--master-dtype", "fp32",
            "--base-dtype", "bf16",
            "--save-strategy", "epoch",
            "--save-total-limit", "1",
            "--lr-scheduler-type", "cosine",
            "--warmup-ratio", "0.03",
        ]
        if job.constrained:
            command.extend([
                "--affine-energy-tau", str(TAUS[job.topology]),
                "--affine-energy-lambda", str(ENERGY_LAMBDA),
            ])
        log = LOGS / "training" / f"{job.name}.log"
        runner.event(
            "training_started", job=job.name, gpu=gpu,
            candidate=job.candidate.key, seed=job.seed,
        )
        with log.open("w", encoding="utf-8") as stream:
            returncode = subprocess.run(
                command, cwd=ROOT, env=runner.env(gpu),
                stdout=stream, stderr=subprocess.STDOUT,
            ).returncode
        ok = returncode == 0 and checkpoint_complete(job)
        runner.event(
            "training_finished", job=job.name, gpu=gpu,
            returncode=returncode, success=ok,
        )
        return ok

    def result_path(candidate: Candidate, seed: int, task: str) -> Path:
        job = Job(candidate, seed)
        if seed == 42:
            return SEED42_OUTPUTS / task / f"{job.name}_full.json"
        return runner.full(job, task)

    def checkpoint_path(candidate: Candidate, seed: int) -> Path:
        job = Job(candidate, seed)
        return SEED42_CHECKPOINTS / job.name if seed == 42 else job.checkpoint

    def baseline(alias: str, seed: int, task: str) -> float:
        root = BASELINE_SEED42_OUTPUTS if seed == 42 else BASELINE_HOLDOUT_OUTPUTS
        path = root / task / f"{alias}_hidden_hr4_seed{seed}_full.json"
        return accuracy(path, task)

    def exact_rhos():  # noqa: ANN202
        values: dict[str, dict[str, float]] = {}
        for alias in MODEL_INFO:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_INFO[alias]["path"], dtype=torch.bfloat16,
            ).cuda().eval()
            input_stats = runner.matrix_stats(
                model.get_input_embeddings().weight.detach(), centered=False,
            )
            output_stats = runner.matrix_stats(
                model.get_output_embeddings().weight.detach(), centered=True,
            )
            for candidate in (item for item in CANDIDATES if item.model_alias == alias):
                for seed in SEEDS:
                    state = load_file(str(
                        checkpoint_path(candidate, seed) / "affine_vocab_adapter.safetensors"
                    ))
                    if candidate.topology == "input":
                        rho = runner.raw_rho(
                            state, "model.embed_tokens.affine", input_stats,
                        )
                    else:
                        rho = runner.raw_rho(state, "lm_head.affine", output_stats)
                    values[f"{candidate.key}:seed{seed}"] = {"rho": rho}
            del model, input_stats, output_stats
            torch.cuda.empty_cache()
        return values

    def energy_trace(candidate: Candidate, seed: int) -> dict[str, float | int]:
        if not candidate.constrained:
            return {"max_rho": math.nan, "max_excess": 0.0, "active_logs": 0}
        job = Job(candidate, seed)
        log = (
            SEED42_LOGS / f"{job.name}.log"
            if seed == 42 else LOGS / "training" / f"{job.name}.log"
        )
        text = log.read_text(encoding="utf-8", errors="ignore")
        rhos = [float(value) for value in re.findall(r"_rho=([0-9.eE+-]+)", text)]
        excesses = [float(value) for value in re.findall(r"_excess=([0-9.eE+-]+)", text)]
        return {
            "max_rho": max(rhos, default=math.nan),
            "max_excess": max(excesses, default=0.0),
            "active_logs": sum(value > 0 for value in excesses),
        }

    def write_report(rhos):  # noqa: ANN001, ANN202
        deltas: dict[tuple[str, int, str], float] = {}
        scores: dict[tuple[str, int, str], float] = {}
        for candidate in CANDIDATES:
            for seed in SEEDS:
                for task in ("math", "gsm8k"):
                    score = accuracy(result_path(candidate, seed, task), task)
                    scores[(candidate.key, seed, task)] = score
                    deltas[(candidate.key, seed, task)] = (
                        score - baseline(candidate.model_alias, seed, task)
                    )

        lines = [
            "# Qwen A-LoRA LR/constraint five-seed holdout", "",
            f"Updated: {runner.now()}", "",
            "Seeds 42--46. Seed 42 is reused from the factorial; seeds 43--46 "
            "are independent holdouts. Every score uses the full 5,000-example "
            "MATH and 1,319-example GSM8K evaluation.",
            "All rows use hidden LoRA r4 + A-LoRA r16. Deltas are paired against "
            "the hidden-r4 baseline with the same seed.", "",
            "## Aggregate paired deltas", "",
            "| model | candidate | MATH delta mean +/- std | 95% CI | "
            "GSM8K delta mean +/- std | 95% CI | mean two-task delta | "
            "worst mean task delta | both-positive seeds | rho mean +/- std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for candidate in CANDIDATES:
            math_values = [deltas[(candidate.key, seed, "math")] for seed in SEEDS]
            gsm_values = [deltas[(candidate.key, seed, "gsm8k")] for seed in SEEDS]
            mm, mh = mean_ci95(math_values)
            gm, gh = mean_ci95(gsm_values)
            seed_means = [
                (deltas[(candidate.key, seed, "math")] +
                 deltas[(candidate.key, seed, "gsm8k")]) / 2
                for seed in SEEDS
            ]
            both_positive = sum(
                deltas[(candidate.key, seed, "math")] > 0
                and deltas[(candidate.key, seed, "gsm8k")] > 0
                for seed in SEEDS
            )
            rho_values = [
                rhos[f"{candidate.key}:seed{seed}"]["rho"] for seed in SEEDS
            ]
            lines.append(
                f"| {candidate.model_alias} | {candidate.label} | "
                f"{fmt_mean_std(math_values)} pp | [{mm-mh:+.4f}, {mm+mh:+.4f}] | "
                f"{fmt_mean_std(gsm_values)} pp | [{gm-gh:+.4f}, {gm+gh:+.4f}] | "
                f"{statistics.mean(seed_means):+.4f} pp | "
                f"{min(mm, gm):+.4f} pp | {both_positive}/5 | "
                f"{statistics.mean(rho_values):.8f} +/- {statistics.stdev(rho_values):.8f} |"
            )

        lines.extend(["", "## Per-seed paired results", ""])
        for candidate in CANDIDATES:
            lines.extend([
                f"### {candidate.model_alias} {candidate.label}", "",
                "| seed | MATH | delta | GSM8K | delta | two-task mean delta | rho |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for seed in SEEDS:
                dm = deltas[(candidate.key, seed, "math")]
                dg = deltas[(candidate.key, seed, "gsm8k")]
                lines.append(
                    f"| {seed} | {scores[(candidate.key, seed, 'math')]:.4f}% | "
                    f"{dm:+.4f} pp | {scores[(candidate.key, seed, 'gsm8k')]:.4f}% | "
                    f"{dg:+.4f} pp | {(dm+dg)/2:+.4f} pp | "
                    f"{rhos[f'{candidate.key}:seed{seed}']['rho']:.8f} |"
                )
            lines.append("")

        lines.extend([
            "## Paired causal contrast: active input hinge minus none at LR=0.25", "",
            "These contrasts use the same model, seed, topology, and A-LoRA LR. "
            "A positive value favors the hinge run. CUDA training is not bitwise "
            "deterministic, so the five-seed paired distribution is the relevant evidence.", "",
            "| model | task | paired difference mean +/- std | 95% CI | positive seeds |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for alias in ("qwen25_3b", "qwen3_4b"):
            none = candidate_for(alias, "input", 0.25, False)
            hinge = candidate_for(alias, "input", 0.25, True)
            for task in ("math", "gsm8k"):
                values = [
                    scores[(hinge.key, seed, task)] - scores[(none.key, seed, task)]
                    for seed in SEEDS
                ]
                mean, half = mean_ci95(values)
                lines.append(
                    f"| {alias} | {task} | {fmt_mean_std(values)} pp | "
                    f"[{mean-half:+.4f}, {mean+half:+.4f}] | "
                    f"{sum(value > 0 for value in values)}/5 |"
                )

        lines.extend([
            "", "## Hinge activation audit", "",
            "| model | seed | max logged rho | max excess | active logging steps |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for candidate in (item for item in CANDIDATES if item.constrained):
            for seed in SEEDS:
                trace = energy_trace(candidate, seed)
                lines.append(
                    f"| {candidate.model_alias} | {seed} | "
                    f"{trace['max_rho']:.8f} | {trace['max_excess']:.8f} | "
                    f"{trace['active_logs']} |"
                )

        RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)

    runner.checkpoint_complete = checkpoint_complete
    runner.train = train
    runner.exact_rhos = exact_rhos
    runner.write_report = write_report
    runner.main()


if __name__ == "__main__":
    main()
