#!/usr/bin/env python
"""Second-dose multi-seed energy-constraint experiment for Qwen.

Adds one tighter active tau for output-only and tied/mergeable shared A-LoRA.
The seed-42 shared checkpoints already exist and are reused; all other jobs
are trained and fully evaluated on MATH and GSM8K with a non-blocking 8-GPU
queue.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
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

CHECKPOINTS = HERE / "qwen_second_tau_checkpoints"
OUTPUTS = HERE / "qwen_second_tau_outputs"
LOGS = HERE / "qwen_second_tau_logs"
STATE = HERE / "qwen_second_tau_state.json"
EVENTS = HERE / "qwen_second_tau_events.jsonl"
RESULTS = HERE / "QWEN_SECOND_TAU_MULTISEED_RESULTS.md"

TAU_CHECKPOINTS = HERE / "qwen_tau_scale_checkpoints"
TAU_OUTPUTS = HERE / "qwen_tau_scale_outputs"
OUTPUT_TAU_OUTPUTS = HERE / "qwen_output_tau_outputs"
SELECTED_OUTPUTS = HERE / "qwen_selected_multiseed_outputs"
MATCHED_OUTPUTS = HERE / "qwen_matched_control_outputs"
SEED42_UNCON_OUTPUTS = HERE / "qwen_lr_constraint_outputs"
SEED42_SHARED_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
BASELINE_SEED42_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
BASELINE_HOLDOUT_OUTPUTS = HERE / "qwen_selected_multiseed_outputs"

SEEDS = (42, 43, 44, 45, 46)
NEW_SEEDS = (43, 44, 45, 46)
ALT_TAUS = {"shared": 0.0125, "output": 0.003125}
CURRENT_TAUS = {"shared": 0.025, "output": 0.00625}
ENERGY_LAMBDA = 100.0


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_second_tau_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class Job:
    model_alias: str
    topology: str
    seed: int

    @property
    def tau(self) -> float:
        return ALT_TAUS[self.topology]

    @property
    def name(self) -> str:
        return (
            f"{self.model_alias}_{self.topology}_ar16_s1_hr4_energy_"
            f"tau{tag(self.tau)}_l100_seed{self.seed}"
        )

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINTS / self.name

    @property
    def model_path(self) -> Path:
        return MODEL_INFO[self.model_alias]["path"]

    @property
    def variant(self) -> str:
        return {
            "shared": "affine_input_lm_head_plus_hidden_lora",
            "output": "affine_lm_head_plus_hidden_lora",
        }[self.topology]


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

# The slower 4B jobs enter first. Shared seed42 is reused from the original
# tau sweep; output tau=0.003125 is new for all five seeds.
JOBS = tuple(
    [Job("qwen3_4b", "output", seed) for seed in SEEDS]
    + [Job("qwen3_4b", "shared", seed) for seed in NEW_SEEDS]
    + [Job("qwen25_3b", "output", seed) for seed in SEEDS]
    + [Job("qwen25_3b", "shared", seed) for seed in NEW_SEEDS]
)


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


def accuracy(path: Path, task: str) -> float:
    key = "clean" if task == "math" else "full"
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    half = 2.7764451052 * std / math.sqrt(len(values))
    return mean, std, half


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
            "--hidden-lora-rank", "4", "--hidden-lora-alpha", "8",
            "--hidden-lora-dropout", "0.05",
            "--affine-rank", "16", "--affine-alpha", "16",
            "--affine-learning-rate-scale", "1",
            "--affine-energy-tau", str(job.tau),
            "--affine-energy-lambda", str(ENERGY_LAMBDA),
            "--max-seq-len", "1024",
            "--per-device-train-batch-size", str(info["batch"]),
            "--gradient-accumulation-steps", str(info["accum"]),
            "--learning-rate", "2e-4", "--num-train-epochs", "1",
            "--logging-steps", "10", "--bf16", "--gradient-checkpointing",
            "--seed", str(job.seed), "--master-dtype", "fp32",
            "--base-dtype", "bf16", "--save-strategy", "epoch",
            "--save-total-limit", "1", "--lr-scheduler-type", "cosine",
            "--warmup-ratio", "0.03",
        ]
        if job.topology == "shared":
            command.extend([
                "--affine-lm-head-bias",
                "--tie-affine-input-lm-head-adapters",
            ])
        # Output-only intentionally has no beta; the beta term is invariant
        # under softmax and should not be counted as output capacity.
        log = LOGS / "training" / f"{job.name}.log"
        runner.event(
            "training_started", job=job.name, gpu=gpu,
            model=job.model_alias, topology=job.topology,
            tau=job.tau, seed=job.seed,
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

    runner.train = train

    def name(alias: str, topology: str, tau: float, seed: int) -> str:
        return (
            f"{alias}_{topology}_ar16_s1_hr4_energy_"
            f"tau{tag(tau)}_l100_seed{seed}"
        )

    def alt_output(alias: str, topology: str, seed: int, task: str) -> Path:
        run_name = name(alias, topology, ALT_TAUS[topology], seed)
        if topology == "shared" and seed == 42:
            return TAU_OUTPUTS / task / f"{run_name}_full.json"
        return OUTPUTS / task / f"{run_name}_full.json"

    def alt_checkpoint(alias: str, topology: str, seed: int) -> Path:
        run_name = name(alias, topology, ALT_TAUS[topology], seed)
        if topology == "shared" and seed == 42:
            return TAU_CHECKPOINTS / run_name
        return CHECKPOINTS / run_name

    def current_output(alias: str, topology: str, seed: int, task: str) -> Path:
        run_name = name(alias, topology, CURRENT_TAUS[topology], seed)
        if seed != 42:
            root = SELECTED_OUTPUTS
        elif topology == "shared":
            root = TAU_OUTPUTS
        else:
            root = OUTPUT_TAU_OUTPUTS
        return root / task / f"{run_name}_full.json"

    def unconstrained_output(alias: str, topology: str, seed: int, task: str) -> Path:
        if seed == 42 and topology == "shared":
            run_name = f"{alias}_shared_ar16_s1_hr4_unconstrained_seed42"
            root = SEED42_SHARED_OUTPUTS
        elif seed == 42:
            run_name = f"{alias}_output_ar16_s1_hr4_alr1_unconstrained_seed42"
            root = SEED42_UNCON_OUTPUTS
        else:
            run_name = f"{alias}_{topology}_ar16_s1_hr4_alr1_unconstrained_seed{seed}"
            root = MATCHED_OUTPUTS
        return root / task / f"{run_name}_full.json"

    def baseline_output(alias: str, seed: int, task: str) -> Path:
        root = BASELINE_SEED42_OUTPUTS if seed == 42 else BASELINE_HOLDOUT_OUTPUTS
        return root / task / f"{alias}_hidden_hr4_seed{seed}_full.json"

    def exact_alt_rhos() -> dict[str, float]:
        values: dict[str, float] = {}
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
            for topology in ("shared", "output"):
                for seed in SEEDS:
                    state = load_file(str(
                        alt_checkpoint(alias, topology, seed)
                        / "affine_vocab_adapter.safetensors"
                    ))
                    if topology == "shared":
                        rho = runner.raw_rho(
                            state, "model.embed_tokens.affine", input_stats,
                        )
                    else:
                        if any(key.endswith(".bias") for key in state):
                            raise RuntimeError(
                                f"Output-only beta unexpectedly present: {alias} seed{seed}"
                            )
                        rho = runner.raw_rho(
                            state, "lm_head.affine", output_stats,
                        )
                    values[f"{alias}:{topology}:seed{seed}"] = rho
            del model, input_stats, output_stats
            torch.cuda.empty_cache()
        return values

    def write_report(rhos: dict[str, float]) -> None:
        lines = [
            "# Qwen second-tau multi-seed dose results", "",
            f"Updated: {runner.now()}", "",
            "Training: MetaMathQA-40K, one epoch, hidden LoRA r4 + A-LoRA r16, A-LoRA LR scale 1, lambda=100.",
            "Shared is tied/mergeable. Output-only has no beta and is mergeable.",
            "All comparisons are paired by model, seed, topology, rank, and optimizer settings.", "",
        ]
        for alias in MODEL_INFO:
            for topology in ("shared", "output"):
                task_values = {
                    condition: {"math": [], "gsm8k": []}
                    for condition in ("hidden", "unconstrained", "current", "alternate")
                }
                rows = []
                for seed in SEEDS:
                    row: dict[str, tuple[float, float]] = {}
                    for task in ("math", "gsm8k"):
                        hidden = accuracy(baseline_output(alias, seed, task), task)
                        unconstrained = accuracy(
                            unconstrained_output(alias, topology, seed, task), task,
                        )
                        current = accuracy(current_output(alias, topology, seed, task), task)
                        alternate = accuracy(alt_output(alias, topology, seed, task), task)
                        for condition, value in (
                            ("hidden", hidden), ("unconstrained", unconstrained),
                            ("current", current), ("alternate", alternate),
                        ):
                            task_values[condition][task].append(value)
                        row[task] = (current, alternate)
                    rows.append((seed, row))

                def task_means(condition: str) -> list[float]:
                    return [
                        (task_values[condition]["math"][i]
                         + task_values[condition]["gsm8k"][i]) / 2
                        for i in range(len(SEEDS))
                    ]

                hidden_means = task_means("hidden")
                uncon_means = task_means("unconstrained")
                current_means = task_means("current")
                alt_means = task_means("alternate")
                alt_current = [a - c for a, c in zip(alt_means, current_means)]
                alt_uncon = [a - u for a, u in zip(alt_means, uncon_means)]
                alt_hidden = [a - h for a, h in zip(alt_means, hidden_means)]
                current_uncon = [c - u for c, u in zip(current_means, uncon_means)]
                acm, acs, ach = mean_std_ci(alt_current)
                aum, aus, auh = mean_std_ci(alt_uncon)
                ahm, ahs, ahh = mean_std_ci(alt_hidden)
                cum, cus, cuh = mean_std_ci(current_uncon)

                lines.extend([
                    f"## {alias} / {topology}", "",
                    f"Current tau={CURRENT_TAUS[topology]:g}; alternate tau={ALT_TAUS[topology]:g}.", "",
                    "| seed | current MATH | alternate MATH | delta | current GSM8K | alternate GSM8K | delta | alternate rho |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ])
                for seed, row in rows:
                    cm, am = row["math"]
                    cg, ag = row["gsm8k"]
                    lines.append(
                        f"| {seed} | {cm:.4f}% | {am:.4f}% | {am-cm:+.4f} pp | "
                        f"{cg:.4f}% | {ag:.4f}% | {ag-cg:+.4f} pp | "
                        f"{rhos[f'{alias}:{topology}:seed{seed}']:.8f} |"
                    )
                lines.extend([
                    "",
                    f"- alternate-current task mean: {acm:+.4f} +/- {acs:.4f} pp; 95% CI [{acm-ach:+.4f}, {acm+ach:+.4f}]; positive seeds {sum(x > 0 for x in alt_current)}/5",
                    f"- current-unconstrained task mean: {cum:+.4f} +/- {cus:.4f} pp; 95% CI [{cum-cuh:+.4f}, {cum+cuh:+.4f}]",
                    f"- alternate-unconstrained task mean: {aum:+.4f} +/- {aus:.4f} pp; 95% CI [{aum-auh:+.4f}, {aum+auh:+.4f}]",
                    f"- alternate-hidden task mean: {ahm:+.4f} +/- {ahs:.4f} pp; 95% CI [{ahm-ahh:+.4f}, {ahm+ahh:+.4f}]",
                    "",
                    "| dose | two-task mean accuracy | std across seeds |",
                    "| --- | ---: | ---: |",
                    f"| hidden | {statistics.mean(hidden_means):.4f}% | {statistics.stdev(hidden_means):.4f} |",
                    f"| unconstrained | {statistics.mean(uncon_means):.4f}% | {statistics.stdev(uncon_means):.4f} |",
                    f"| tau={CURRENT_TAUS[topology]:g} | {statistics.mean(current_means):.4f}% | {statistics.stdev(current_means):.4f} |",
                    f"| tau={ALT_TAUS[topology]:g} | {statistics.mean(alt_means):.4f}% | {statistics.stdev(alt_means):.4f} |",
                    "",
                ])
        RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    required = []
    for alias in MODEL_INFO:
        for topology in ("shared", "output"):
            for seed in SEEDS:
                for task in ("math", "gsm8k"):
                    required.extend([
                        current_output(alias, topology, seed, task),
                        unconstrained_output(alias, topology, seed, task),
                        baseline_output(alias, seed, task),
                    ])
                    if topology == "shared" and seed == 42:
                        required.append(alt_output(alias, topology, seed, task))
            if topology == "shared":
                required.append(
                    alt_checkpoint(alias, topology, 42)
                    / "affine_vocab_adapter.safetensors"
                )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(map(str, missing)))
    if os.environ.get("SECOND_TAU_CHECK_ONLY") == "1":
        print(f"prerequisites_ok jobs={len(JOBS)} unique={len({job.name for job in JOBS})}")
        for job in JOBS:
            print(job.name)
        return

    for directory in (
        CHECKPOINTS,
        OUTPUTS / "math", OUTPUTS / "gsm8k",
        LOGS / "training", LOGS / "math", LOGS / "gsm8k",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if EVENTS.exists():
        EVENTS.unlink()
    runner.write_state(
        "dynamic_training_and_evaluation", jobs=[job.name for job in JOBS],
    )
    runner.dynamic_run()
    runner.write_state("merging")
    runner.merge_outputs()
    runner.write_state("energy_measurement")
    rhos = exact_alt_rhos()
    write_report(rhos)
    runner.write_state("complete", rhos=rhos, result=str(RESULTS))
    runner.event("complete", result=str(RESULTS))


if __name__ == "__main__":
    main()
