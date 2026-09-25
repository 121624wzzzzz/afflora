#!/usr/bin/env python
"""Sweep mergeable output-only A-LoRA energy thresholds on Qwen 3B/4B."""

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
BASELINE_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"

CHECKPOINTS = HERE / "qwen_output_tau_checkpoints"
OUTPUTS = HERE / "qwen_output_tau_outputs"
LOGS = HERE / "qwen_output_tau_logs"
STATE = HERE / "qwen_output_tau_state.json"
EVENTS = HERE / "qwen_output_tau_events.jsonl"
RESULTS = HERE / "QWEN_OUTPUT_TAU_RESULTS.md"

ENERGY_LAMBDA = 100.0
TAUS = (0.00625, 0.0125, 0.025, 0.05)


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_output_tau_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tau_slug(tau: float) -> str:
    return f"{tau:.6f}".rstrip("0").replace(".", "p")


@dataclass(frozen=True)
class Job:
    model_alias: str
    tau: float
    topology: str = "output"
    variant: str = "affine_lm_head_plus_hidden_lora"

    @property
    def name(self) -> str:
        return (
            f"{self.model_alias}_output_ar16_s1_hr4_energy_"
            f"tau{tau_slug(self.tau)}_l100_seed42"
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

# Put the four slower 4B jobs first, followed by the four 3B jobs.  The shared
# scheduler lets each GPU immediately take an eval shard when its training ends.
JOBS = tuple(
    [Job("qwen3_4b", tau) for tau in TAUS]
    + [Job("qwen25_3b", tau) for tau in TAUS]
)


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


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

    def checkpoint_complete(job: Job) -> bool:
        return all(
            (job.checkpoint / filename).is_file()
            for filename in (
                "adapter_model.safetensors",
                "affine_vocab_adapter.safetensors",
                "run_args.json",
            )
        )

    def train(job: Job, gpu: int) -> bool:
        if checkpoint_complete(job):
            runner.event("training_reused", job=job.name, gpu=gpu)
            return True
        info = MODEL_INFO[job.model_alias]
        cmd = [
            str(runner.PYTHON), str(runner.TRAIN_SCRIPT),
            "--model-path", str(job.model_path),
            "--train-data", str(runner.TRAIN_DATA),
            "--output-dir", str(job.checkpoint),
            "--variant", job.variant,
            "--hidden-lora-rank", "4", "--hidden-lora-alpha", "8",
            "--hidden-lora-dropout", "0.05",
            "--affine-rank", "16", "--affine-alpha", "16",
            "--affine-energy-tau", str(job.tau),
            "--affine-energy-lambda", str(ENERGY_LAMBDA),
            "--max-seq-len", "1024",
            "--per-device-train-batch-size", str(info["batch"]),
            "--gradient-accumulation-steps", str(info["accum"]),
            "--learning-rate", "2e-4", "--num-train-epochs", "1",
            "--logging-steps", "10", "--bf16", "--gradient-checkpointing",
            "--seed", "42", "--master-dtype", "fp32", "--base-dtype", "bf16",
            "--save-strategy", "epoch", "--save-total-limit", "1",
            "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
        ]
        # Deliberately do not pass --affine-lm-head-bias.  The independent
        # output beta is softmax-invariant in the paper's parameterization;
        # the multiplicative map alone is exactly mergeable into lm_head.weight.
        log = LOGS / "training" / f"{job.name}.log"
        runner.event(
            "training_started", job=job.name, gpu=gpu,
            topology=job.topology, tau=job.tau,
        )
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(
                cmd, cwd=ROOT, env=runner.env(gpu),
                stdout=stream, stderr=subprocess.STDOUT,
            ).returncode
        ok = rc == 0 and checkpoint_complete(job)
        runner.event(
            "training_finished", job=job.name, gpu=gpu,
            returncode=rc, success=ok,
        )
        return ok

    def exact_rhos():  # noqa: ANN202
        values: dict[str, dict[str, float]] = {}
        for alias in MODEL_INFO:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_INFO[alias]["path"], torch_dtype=torch.bfloat16,
            ).cuda().eval()
            stats = runner.matrix_stats(
                model.get_output_embeddings().weight.detach(), centered=True,
            )
            for job in (item for item in JOBS if item.model_alias == alias):
                affine_state = load_file(
                    str(job.checkpoint / "affine_vocab_adapter.safetensors")
                )
                if any(key.endswith(".bias") for key in affine_state):
                    raise RuntimeError(f"Output-only beta unexpectedly present: {job.name}")
                values[job.name] = {
                    "output": runner.raw_rho(
                        affine_state, "lm_head.affine", stats,
                    )
                }
            del model, stats
            torch.cuda.empty_cache()
        return values

    def baseline(alias: str, task: str) -> float:
        key = "clean" if task == "math" else "full"
        name = f"{alias}_hidden_hr4_seed42_full.json"
        return accuracy(BASELINE_OUTPUTS / task / name, key)

    def score(job: Job, task: str) -> float:
        key = "clean" if task == "math" else "full"
        return accuracy(runner.full(job, task), key)

    def write_report(rhos):  # noqa: ANN001, ANN202
        lines = [
            "# Qwen 3B/4B output-only affine-energy tau sweep", "",
            f"Updated: {runner.now()}", "",
            "All treatment rows use jointly trained hidden LoRA r4 + output-only "
            "A-LoRA r16 scale 1, seed 42, lambda=100, and no output beta. "
            "The multiplicative output adapter is mergeable into lm_head.weight.", "",
        ]
        for alias in ("qwen25_3b", "qwen3_4b"):
            bm, bg = baseline(alias, "math"), baseline(alias, "gsm8k")
            lines.extend([
                f"## {alias}", "",
                "| topology | tau | measured output rho | MATH | delta vs hidden | GSM8K | delta vs hidden |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| hidden r4 | - | - | {bm:.4f}% | - | {bg:.4f}% | - |",
            ])
            for tau in (0.05, 0.025, 0.0125, 0.00625):
                job = next(
                    item for item in JOBS
                    if item.model_alias == alias and item.tau == tau
                )
                m, g = score(job, "math"), score(job, "gsm8k")
                lines.append(
                    f"| output | {tau:.5g} | {rhos[job.name]['output']:.8f} | "
                    f"{m:.4f}% | {m-bm:+.4f} pp | "
                    f"{g:.4f}% | {g-bg:+.4f} pp |"
                )
            lines.append("")
        RESULTS.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines), flush=True)

    runner.checkpoint_complete = checkpoint_complete
    runner.train = train
    runner.exact_rhos = exact_rhos
    runner.write_report = write_report
    runner.main()


if __name__ == "__main__":
    main()
