#!/usr/bin/env python
"""Factor A-LoRA learning rate against the energy constraint on Qwen 3B/4B.

The selection seed is 42.  Every new checkpoint receives full 16-shard MATH
and GSM8K evaluation.  Existing constrained LR-scale-1 checkpoints are reused
only in the final comparison table, so the dynamic queue contains 20 new jobs.
"""

from __future__ import annotations

import importlib.util
import json
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

CHECKPOINTS = HERE / "qwen_lr_constraint_checkpoints"
OUTPUTS = HERE / "qwen_lr_constraint_outputs"
LOGS = HERE / "qwen_lr_constraint_logs"
STATE = HERE / "qwen_lr_constraint_state.json"
EVENTS = HERE / "qwen_lr_constraint_events.jsonl"
RESULTS = HERE / "QWEN_LR_CONSTRAINT_FACTORIAL_RESULTS.md"

BASELINE_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
INPUT_EXISTING_CHECKPOINTS = HERE / "qwen_tau_scale_checkpoints"
INPUT_EXISTING_OUTPUTS = HERE / "qwen_tau_scale_outputs"
OUTPUT_EXISTING_CHECKPOINTS = HERE / "qwen_output_tau_checkpoints"
OUTPUT_EXISTING_OUTPUTS = HERE / "qwen_output_tau_outputs"

SEED = 42
LR_SCALES = (0.25, 0.5)
TAUS = {"input": 0.0125, "output": 0.00625}
ENERGY_LAMBDA = 100.0


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_lr_constraint_base", SOURCE)
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
    lr_scale: float
    constrained: bool

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
            f"alr{tag(self.lr_scale)}_{regularizer}_seed{SEED}"
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

# Put the longer 4B jobs first.  The shared queue is non-blocking: a GPU takes
# the next train/eval item immediately when its current item finishes.
JOBS = tuple(
    Job(alias, topology, lr_scale, constrained)
    for alias in ("qwen3_4b", "qwen25_3b")
    for topology in ("input", "output")
    for constrained, scales in (
        (False, (0.25, 0.5, 1.0)),
        (True, LR_SCALES),
    )
    for lr_scale in scales
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


def existing_name(alias: str, topology: str) -> str:
    return (
        f"{alias}_{topology}_ar16_s1_hr4_energy_"
        f"tau{tag(TAUS[topology])}_l100_seed{SEED}"
    )


def existing_checkpoint(alias: str, topology: str) -> Path:
    root = INPUT_EXISTING_CHECKPOINTS if topology == "input" else OUTPUT_EXISTING_CHECKPOINTS
    return root / existing_name(alias, topology)


def existing_full(alias: str, topology: str, task: str) -> Path:
    root = INPUT_EXISTING_OUTPUTS if topology == "input" else OUTPUT_EXISTING_OUTPUTS
    return root / task / f"{existing_name(alias, topology)}_full.json"


def accuracy(path: Path, task: str) -> float:
    key = "clean" if task == "math" else "full"
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def baseline(alias: str, task: str) -> float:
    path = BASELINE_OUTPUTS / task / f"{alias}_hidden_hr4_seed{SEED}_full.json"
    return accuracy(path, task)


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
            "--seed", str(SEED),
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
        # Output-only beta is intentionally disabled (the parser default).
        log = LOGS / "training" / f"{job.name}.log"
        runner.event(
            "training_started", job=job.name, gpu=gpu,
            topology=job.topology, lr_scale=job.lr_scale,
            constrained=job.constrained,
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
            for job in (item for item in JOBS if item.model_alias == alias):
                state = load_file(str(job.checkpoint / "affine_vocab_adapter.safetensors"))
                if job.topology == "input":
                    rho = runner.raw_rho(state, "model.embed_tokens.affine", input_stats)
                else:
                    rho = runner.raw_rho(state, "lm_head.affine", output_stats)
                values[job.name] = {"rho": rho}
            for topology in ("input", "output"):
                state = load_file(str(
                    existing_checkpoint(alias, topology) / "affine_vocab_adapter.safetensors"
                ))
                if topology == "input":
                    rho = runner.raw_rho(state, "model.embed_tokens.affine", input_stats)
                else:
                    rho = runner.raw_rho(state, "lm_head.affine", output_stats)
                values[f"existing:{alias}:{topology}"] = {"rho": rho}
            del model, input_stats, output_stats
            torch.cuda.empty_cache()
        return values

    def write_report(rhos):  # noqa: ANN001, ANN202
        lines = [
            "# Qwen A-LoRA LR x energy-constraint factorial", "",
            f"Updated: {runner.now()}", "",
            "Selection seed: 42. Training uses MetaMathQA-40K for one epoch, "
            "effective batch 16, hidden LoRA r4, and A-LoRA r16 scale 1.",
            "Every new row has full 5,000-example MATH and 1,319-example GSM8K evaluation.",
            "Constrained input uses tau=0.0125; constrained output uses tau=0.00625; "
            "both use lambda=100. Output-only beta is disabled.", "",
        ]
        for alias in ("qwen25_3b", "qwen3_4b"):
            bm, bg = baseline(alias, "math"), baseline(alias, "gsm8k")
            lines.extend([
                f"## {alias}", "",
                "| topology | constraint | A LR scale | rho | MATH | delta | GSM8K | delta | mean delta | worst delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| - | hidden r4 | - | - | {bm:.4f}% | - | {bg:.4f}% | - | - | - |",
            ])
            for topology in ("input", "output"):
                for constrained in (False, True):
                    for lr_scale in (0.25, 0.5, 1.0):
                        if constrained and lr_scale == 1.0:
                            math_score = accuracy(existing_full(alias, topology, "math"), "math")
                            gsm_score = accuracy(existing_full(alias, topology, "gsm8k"), "gsm8k")
                            rho = rhos[f"existing:{alias}:{topology}"]["rho"]
                        else:
                            job = next(
                                item for item in JOBS
                                if item.model_alias == alias
                                and item.topology == topology
                                and item.constrained == constrained
                                and item.lr_scale == lr_scale
                            )
                            math_score = accuracy(runner.full(job, "math"), "math")
                            gsm_score = accuracy(runner.full(job, "gsm8k"), "gsm8k")
                            rho = rhos[job.name]["rho"]
                        dm, dg = math_score - bm, gsm_score - bg
                        label = "hinge" if constrained else "none"
                        lines.append(
                            f"| {topology} | {label} | {lr_scale:g} | {rho:.8f} | "
                            f"{math_score:.4f}% | {dm:+.4f} pp | "
                            f"{gsm_score:.4f}% | {dg:+.4f} pp | "
                            f"{(dm + dg) / 2:+.4f} pp | {min(dm, dg):+.4f} pp |"
                        )
                lines.append("")
        RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)

    runner.checkpoint_complete = checkpoint_complete
    runner.train = train
    runner.exact_rhos = exact_rhos
    runner.write_report = write_report
    runner.main()


if __name__ == "__main__":
    main()
