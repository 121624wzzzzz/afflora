#!/usr/bin/env python
"""Matched unconstrained controls for the existing constrained Qwen runs.

The constrained input/shared/output treatments already exist for five seeds.
This runner trains only the missing seed-43--46 unconstrained controls needed
for same-seed, same-topology, same-learning-rate causal contrasts.  Seed 42
controls and all constrained treatments are reused.
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

CHECKPOINTS = HERE / "qwen_matched_control_checkpoints"
OUTPUTS = HERE / "qwen_matched_control_outputs"
LOGS = HERE / "qwen_matched_control_logs"
STATE = HERE / "qwen_matched_control_state.json"
EVENTS = HERE / "qwen_matched_control_events.jsonl"
RESULTS = HERE / "QWEN_MATCHED_CONSTRAINT_CONTROL_RESULTS.md"

SEED42_FACTORIAL_CHECKPOINTS = HERE / "qwen_lr_constraint_checkpoints"
SEED42_FACTORIAL_OUTPUTS = HERE / "qwen_lr_constraint_outputs"
SEED42_SHARED_CHECKPOINTS = HERE / "qwen_multiseed_scale_checkpoints"
SEED42_SHARED_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
SEED42_TAU_OUTPUTS = HERE / "qwen_tau_scale_outputs"
SEED42_OUTPUT_TAU_OUTPUTS = HERE / "qwen_output_tau_outputs"
LR_HOLDOUT_CHECKPOINTS = HERE / "qwen_lr_constraint_holdout_checkpoints"
LR_HOLDOUT_OUTPUTS = HERE / "qwen_lr_constraint_holdout_outputs"
SELECTED_OUTPUTS = HERE / "qwen_selected_multiseed_outputs"
BASELINE_SEED42_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
BASELINE_HOLDOUT_OUTPUTS = HERE / "qwen_selected_multiseed_outputs"

SEEDS = (42, 43, 44, 45, 46)
NEW_SEEDS = (43, 44, 45, 46)
TAUS = {"input": 0.0125, "shared": 0.025, "output": 0.00625}


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_matched_control_base", SOURCE)
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
    def name(self) -> str:
        return (
            f"{self.model_alias}_{self.topology}_ar16_s1_hr4_"
            f"alr1_unconstrained_seed{self.seed}"
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
            "input": "affine_input_plus_hidden_lora",
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

# Qwen3-4B first because it is slower.  Qwen3 input controls are omitted: the
# active holdout runner already trains/evaluates exactly those four controls.
JOBS = tuple(
    [
        Job("qwen3_4b", topology, seed)
        for topology in ("shared", "output")
        for seed in NEW_SEEDS
    ]
    + [
        Job("qwen25_3b", topology, seed)
        for topology in ("input", "shared", "output")
        for seed in NEW_SEEDS
    ]
)


def checkpoint_complete(job: Job) -> bool:
    return all(
        (job.checkpoint / name).is_file()
        for name in (
            "adapter_model.safetensors",
            "affine_vocab_adapter.safetensors",
            "affine_vocab_config.json",
            "run_args.json",
        )
    )


def accuracy(path: Path, task: str) -> float:
    key = "clean" if task == "math" else "full"
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
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
        # Output-only intentionally has no beta; its multiplicative map is
        # mergeable and the omitted beta would be softmax-invariant.
        log = LOGS / "training" / f"{job.name}.log"
        runner.event(
            "training_started", job=job.name, gpu=gpu,
            model=job.model_alias, topology=job.topology, seed=job.seed,
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

    def control_name(alias: str, topology: str, seed: int) -> str:
        if seed == 42 and topology == "shared":
            return f"{alias}_shared_ar16_s1_hr4_unconstrained_seed42"
        return f"{alias}_{topology}_ar16_s1_hr4_alr1_unconstrained_seed{seed}"

    def control_output(alias: str, topology: str, seed: int, task: str) -> Path:
        name = control_name(alias, topology, seed)
        if seed == 42:
            root = (
                SEED42_SHARED_OUTPUTS
                if topology == "shared"
                else SEED42_FACTORIAL_OUTPUTS
            )
            return root / task / f"{name}_full.json"
        if alias == "qwen3_4b" and topology == "input":
            return LR_HOLDOUT_OUTPUTS / task / f"{name}_full.json"
        job = next(
            item for item in JOBS
            if item.model_alias == alias
            and item.topology == topology
            and item.seed == seed
        )
        return runner.full(job, task)

    def control_checkpoint(alias: str, topology: str, seed: int) -> Path:
        name = control_name(alias, topology, seed)
        if seed == 42:
            root = (
                SEED42_SHARED_CHECKPOINTS
                if topology == "shared"
                else SEED42_FACTORIAL_CHECKPOINTS
            )
        elif alias == "qwen3_4b" and topology == "input":
            root = LR_HOLDOUT_CHECKPOINTS
        else:
            root = CHECKPOINTS
        return root / name

    def treatment_name(alias: str, topology: str, seed: int) -> str:
        return (
            f"{alias}_{topology}_ar16_s1_hr4_energy_"
            f"tau{tag(TAUS[topology])}_l100_seed{seed}"
        )

    def treatment_output(alias: str, topology: str, seed: int, task: str) -> Path:
        name = treatment_name(alias, topology, seed)
        if seed != 42:
            root = SELECTED_OUTPUTS
        elif topology == "output":
            root = SEED42_OUTPUT_TAU_OUTPUTS
        else:
            root = SEED42_TAU_OUTPUTS
        return root / task / f"{name}_full.json"

    def baseline_output(alias: str, seed: int, task: str) -> Path:
        root = BASELINE_SEED42_OUTPUTS if seed == 42 else BASELINE_HOLDOUT_OUTPUTS
        return root / task / f"{alias}_hidden_hr4_seed{seed}_full.json"

    def exact_control_rhos():  # noqa: ANN202
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
            topologies = ("input", "shared", "output")
            for topology in topologies:
                for seed in SEEDS:
                    name = control_name(alias, topology, seed)
                    state = load_file(str(
                        control_checkpoint(alias, topology, seed)
                        / "affine_vocab_adapter.safetensors"
                    ))
                    if topology in ("input", "shared"):
                        rho = runner.raw_rho(state, "model.embed_tokens.affine", input_stats)
                    else:
                        if any(key.endswith(".bias") for key in state):
                            raise RuntimeError(f"Output-only beta unexpectedly present: {name}")
                        rho = runner.raw_rho(state, "lm_head.affine", output_stats)
                    values[f"{alias}:{topology}:seed{seed}"] = rho
            del model, input_stats, output_stats
            torch.cuda.empty_cache()
        return values

    def write_report(rhos: dict[str, float]) -> None:
        lines = [
            "# Qwen matched constraint-control results", "",
            f"Updated: {runner.now()}", "",
            "All contrasts use the same model, seed, topology, rank, hidden LoRA, and A-LoRA LR. ",
            "The only intervention is the existing energy constraint versus the newly completed unconstrained control.", "",
        ]
        comparisons = [
            ("qwen25_3b", "input"),
            ("qwen25_3b", "shared"),
            ("qwen25_3b", "output"),
            ("qwen3_4b", "input"),
            ("qwen3_4b", "shared"),
            ("qwen3_4b", "output"),
        ]
        for alias, topology in comparisons:
            causal: dict[str, list[float]] = {"math": [], "gsm8k": []}
            treatment_vs_hidden: dict[str, list[float]] = {"math": [], "gsm8k": []}
            control_vs_hidden: dict[str, list[float]] = {"math": [], "gsm8k": []}
            rows = []
            for seed in SEEDS:
                values: dict[str, tuple[float, float, float]] = {}
                for task in ("math", "gsm8k"):
                    control = accuracy(control_output(alias, topology, seed, task), task)
                    treatment = accuracy(treatment_output(alias, topology, seed, task), task)
                    baseline = accuracy(baseline_output(alias, seed, task), task)
                    causal[task].append(treatment - control)
                    treatment_vs_hidden[task].append(treatment - baseline)
                    control_vs_hidden[task].append(control - baseline)
                    values[task] = (control, treatment, treatment - control)
                rows.append((seed, values))
            mean_task = [
                (causal["math"][i] + causal["gsm8k"][i]) / 2
                for i in range(len(SEEDS))
            ]
            mm, ms, mh = mean_ci95(causal["math"])
            gm, gs, gh = mean_ci95(causal["gsm8k"])
            am, ass, ah = mean_ci95(mean_task)
            lines.extend([
                f"## {alias} / {topology}", "",
                f"Constraint: tau={TAUS[topology]:g}, lambda=100. Shared is tied/mergeable; output has no beta and is mergeable.", "",
                "| seed | control MATH | constrained MATH | causal delta | control GSM8K | constrained GSM8K | causal delta | control rho |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for seed, values in rows:
                cm, tm, dm = values["math"]
                cg, tg, dg = values["gsm8k"]
                lines.append(
                    f"| {seed} | {cm:.4f}% | {tm:.4f}% | {dm:+.4f} pp | "
                    f"{cg:.4f}% | {tg:.4f}% | {dg:+.4f} pp | "
                    f"{rhos[f'{alias}:{topology}:seed{seed}']:.8f} |"
                )
            lines.extend([
                "",
                f"- constrained-control MATH: {mm:+.4f} +/- {ms:.4f} pp; 95% CI [{mm-mh:+.4f}, {mm+mh:+.4f}]",
                f"- constrained-control GSM8K: {gm:+.4f} +/- {gs:.4f} pp; 95% CI [{gm-gh:+.4f}, {gm+gh:+.4f}]",
                f"- constrained-control task mean: {am:+.4f} +/- {ass:.4f} pp; 95% CI [{am-ah:+.4f}, {am+ah:+.4f}]; positive seeds {sum(x > 0 for x in mean_task)}/5",
                f"- constrained-hidden task mean: {statistics.mean([(treatment_vs_hidden['math'][i] + treatment_vs_hidden['gsm8k'][i]) / 2 for i in range(5)]):+.4f} pp",
                f"- control-hidden task mean: {statistics.mean([(control_vs_hidden['math'][i] + control_vs_hidden['gsm8k'][i]) / 2 for i in range(5)]):+.4f} pp",
                "",
            ])
        RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    required = []
    for alias, topology in (
        ("qwen25_3b", "input"),
        ("qwen25_3b", "shared"),
        ("qwen25_3b", "output"),
        ("qwen3_4b", "input"),
        ("qwen3_4b", "shared"),
        ("qwen3_4b", "output"),
    ):
        for seed in SEEDS:
            for task in ("math", "gsm8k"):
                required.extend([
                    treatment_output(alias, topology, seed, task),
                    baseline_output(alias, seed, task),
                ])
                if seed == 42:
                    required.append(control_output(alias, topology, seed, task))
        required.append(
            control_checkpoint(alias, topology, 42)
            / "affine_vocab_adapter.safetensors"
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(map(str, missing)))
    if os.environ.get("MATCHED_CONTROL_CHECK_ONLY") == "1":
        print(f"prerequisites_ok jobs={len(JOBS)} unique={len({job.name for job in JOBS})}")
        for job in JOBS:
            print(job.name)
        return

    if os.environ.get("MATCHED_CONTROL_REPORT_ONLY") == "1":
        rhos = exact_control_rhos()
        write_report(rhos)
        print(f"report_complete={RESULTS}")
        return

    for directory in (
        CHECKPOINTS,
        OUTPUTS / "math", OUTPUTS / "gsm8k",
        LOGS / "training", LOGS / "math", LOGS / "gsm8k",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if EVENTS.exists():
        EVENTS.unlink()
    runner.write_state("dynamic_training_and_evaluation", jobs=[job.name for job in JOBS])
    runner.dynamic_run()
    runner.write_state("merging")
    runner.merge_outputs()
    runner.write_state("energy_measurement")
    rhos = exact_control_rhos()
    write_report(rhos)
    runner.write_state("complete", rhos=rhos, result=str(RESULTS))
    runner.event("complete", result=str(RESULTS))


if __name__ == "__main__":
    main()
