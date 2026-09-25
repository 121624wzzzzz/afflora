#!/usr/bin/env python
"""Paired multi-seed confirmation for frozen Qwen 3B/4B A-LoRA choices.

Seed 42 was used for topology-specific tau selection.  This runner trains only
the missing seeds 43--46 for hidden LoRA and the frozen input/shared/output
configurations, evaluates full MATH and GSM8K, and reports paired deltas.  Work
is scheduled at single-GPU task granularity, so a GPU takes the next task as
soon as it becomes free.
"""

from __future__ import annotations

import importlib.util
import json
import math
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

CHECKPOINTS = HERE / "qwen_selected_multiseed_checkpoints"
OUTPUTS = HERE / "qwen_selected_multiseed_outputs"
LOGS = HERE / "qwen_selected_multiseed_logs"
STATE = HERE / "qwen_selected_multiseed_state.json"
EVENTS = HERE / "qwen_selected_multiseed_events.jsonl"
RESULTS = HERE / "QWEN_SELECTED_MULTISEED_RESULTS.md"

SEEDS = (42, 43, 44, 45, 46)
NEW_SEEDS = (43, 44, 45, 46)
ENERGY_LAMBDA = 100.0
# One frozen threshold per topology, shared across both model sizes.  Shared
# tau=0.025 is the only shared threshold with nonnegative seed-42 deltas on
# both tasks and both models.  Output tau=0.00625 is active and positive on
# both tasks/models.  Input tau=0.0125 has the strongest cross-size aggregate.
SELECTED_TAU = {"input": 0.0125, "shared": 0.025, "output": 0.00625}


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_selected_multiseed_base", SOURCE)
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
    topology: str
    seed: int

    @property
    def tau(self) -> float | None:
        return SELECTED_TAU.get(self.topology)

    @property
    def has_affine(self) -> bool:
        return self.topology != "hidden"

    @property
    def variant(self) -> str:
        return {
            "hidden": "hidden_lora",
            "input": "affine_input_plus_hidden_lora",
            "shared": "affine_input_lm_head_plus_hidden_lora",
            "output": "affine_lm_head_plus_hidden_lora",
        }[self.topology]

    @property
    def name(self) -> str:
        if not self.has_affine:
            return f"{self.model_alias}_hidden_hr4_seed{self.seed}"
        assert self.tau is not None
        return (
            f"{self.model_alias}_{self.topology}_ar16_s1_hr4_energy_"
            f"tau{tau_slug(self.tau)}_l100_seed{self.seed}"
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

# Put all slower 4B jobs first.  Once one finishes, its GPU immediately takes
# the next queued job; no batch barrier is used.
JOBS = tuple(
    Job(alias, topology, seed)
    for alias in ("qwen3_4b", "qwen25_3b")
    for topology in ("hidden", "input", "shared", "output")
    for seed in NEW_SEEDS
)


def result_key(alias: str, topology: str, seed: int) -> str:
    return f"{alias}:{topology}:seed{seed}"


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"


def mean_ci95(values: list[float]) -> tuple[float, float]:
    # Two-sided Student-t 95% CI for n=5 (df=4).
    mean = statistics.mean(values)
    half_width = 2.7764451052 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half_width


def seed42_name(alias: str, topology: str) -> str:
    if topology == "hidden":
        return f"{alias}_hidden_hr4_seed42"
    tau = SELECTED_TAU[topology]
    return (
        f"{alias}_{topology}_ar16_s1_hr4_energy_"
        f"tau{tau_slug(tau)}_l100_seed42"
    )


def seed42_checkpoint(alias: str, topology: str) -> Path:
    name = seed42_name(alias, topology)
    root = {
        "hidden": HERE / "qwen_multiseed_scale_checkpoints",
        "input": HERE / "qwen_tau_scale_checkpoints",
        "shared": HERE / "qwen_tau_scale_checkpoints",
        "output": HERE / "qwen_output_tau_checkpoints",
    }[topology]
    return root / name


def seed42_full(alias: str, topology: str, task: str) -> Path:
    name = seed42_name(alias, topology)
    root = {
        "hidden": HERE / "qwen_multiseed_scale_outputs",
        "input": HERE / "qwen_tau_scale_outputs",
        "shared": HERE / "qwen_tau_scale_outputs",
        "output": HERE / "qwen_output_tau_outputs",
    }[topology]
    return root / task / f"{name}_full.json"


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
        required = ["adapter_model.safetensors", "run_args.json"]
        if job.has_affine:
            required.append("affine_vocab_adapter.safetensors")
        return all((job.checkpoint / filename).is_file() for filename in required)

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
        if job.has_affine:
            assert job.tau is not None
            cmd.extend([
                "--affine-rank", "16", "--affine-alpha", "16",
                "--affine-energy-tau", str(job.tau),
                "--affine-energy-lambda", str(ENERGY_LAMBDA),
            ])
        if job.topology == "shared":
            cmd.extend(["--affine-lm-head-bias", "--tie-affine-input-lm-head-adapters"])
        # Output-only intentionally has no beta: the independent output beta is
        # softmax-invariant, while the multiplicative adapter is mergeable.
        log = LOGS / "training" / f"{job.name}.log"
        runner.event(
            "training_started", job=job.name, gpu=gpu,
            model=job.model_alias, topology=job.topology,
            tau=job.tau, seed=job.seed,
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

    def rho_for(
        state: dict[str, torch.Tensor], topology: str,
        input_stats, output_stats,  # noqa: ANN001
    ) -> float:
        if topology in ("input", "shared"):
            return runner.raw_rho(state, "model.embed_tokens.affine", input_stats)
        if topology == "output":
            if any(key.endswith(".bias") for key in state):
                raise RuntimeError("Output-only beta unexpectedly present")
            return runner.raw_rho(state, "lm_head.affine", output_stats)
        raise ValueError(topology)

    def exact_rhos():  # noqa: ANN202
        values: dict[str, dict[str, float]] = {}
        for alias in MODEL_INFO:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_INFO[alias]["path"], torch_dtype=torch.bfloat16,
            ).cuda().eval()
            input_stats = runner.matrix_stats(
                model.get_input_embeddings().weight.detach(), centered=False,
            )
            output_stats = runner.matrix_stats(
                model.get_output_embeddings().weight.detach(), centered=True,
            )
            for topology in ("input", "shared", "output"):
                seed42_state = load_file(
                    str(seed42_checkpoint(alias, topology) / "affine_vocab_adapter.safetensors")
                )
                values[result_key(alias, topology, 42)] = {
                    "rho": rho_for(seed42_state, topology, input_stats, output_stats)
                }
            for job in (
                item for item in JOBS
                if item.model_alias == alias and item.has_affine
            ):
                state = load_file(str(job.checkpoint / "affine_vocab_adapter.safetensors"))
                values[result_key(alias, job.topology, job.seed)] = {
                    "rho": rho_for(state, job.topology, input_stats, output_stats)
                }
            del model, input_stats, output_stats
            torch.cuda.empty_cache()
        return values

    def score(alias: str, topology: str, seed: int, task: str) -> float:
        key = "clean" if task == "math" else "full"
        if seed == 42:
            path = seed42_full(alias, topology, task)
        else:
            job = next(
                item for item in JOBS
                if item.model_alias == alias
                and item.topology == topology
                and item.seed == seed
            )
            path = runner.full(job, task)
        return accuracy(path, key)

    def write_report(rhos):  # noqa: ANN001, ANN202
        lines = [
            "# Qwen 3B/4B selected-configuration multi-seed confirmation", "",
            f"Updated: {runner.now()}", "",
            "Training: MetaMathQA-40K, one epoch, effective batch 16, hidden LoRA r4. "
            "A-LoRA uses rank 16, scale 1, lambda=100, and is jointly trained with hidden LoRA.",
            "Seed 42 is the tau-selection seed; seeds 43--46 are holdout confirmation seeds. "
            "Thresholds were frozen across model sizes: input=0.0125, shared=0.025, output=0.00625. "
            "Shared is tied/mergeable; output-only has no beta and is mergeable.", "",
        ]
        for alias in ("qwen25_3b", "qwen3_4b"):
            lines.extend([f"## {alias}", ""])
            for topology in ("input", "shared", "output"):
                lines.extend([
                    f"### {topology} (tau={SELECTED_TAU[topology]:.5g})", "",
                    "| seed | hidden MATH | treatment MATH | paired delta | hidden GSM8K | treatment GSM8K | paired delta | rho |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ])
                hm_values: list[float] = []
                tm_values: list[float] = []
                dm_values: list[float] = []
                hg_values: list[float] = []
                tg_values: list[float] = []
                dg_values: list[float] = []
                for seed in SEEDS:
                    hm = score(alias, "hidden", seed, "math")
                    hg = score(alias, "hidden", seed, "gsm8k")
                    tm = score(alias, topology, seed, "math")
                    tg = score(alias, topology, seed, "gsm8k")
                    dm, dg = tm - hm, tg - hg
                    hm_values.append(hm); tm_values.append(tm); dm_values.append(dm)
                    hg_values.append(hg); tg_values.append(tg); dg_values.append(dg)
                    rho = rhos[result_key(alias, topology, seed)]["rho"]
                    seed_label = f"{seed} (selection)" if seed == 42 else f"{seed} (holdout)"
                    lines.append(
                        f"| {seed_label} | {hm:.4f}% | {tm:.4f}% | {dm:+.4f} pp | "
                        f"{hg:.4f}% | {tg:.4f}% | {dg:+.4f} pp | {rho:.8f} |"
                    )
                dm_mean, dm_ci = mean_ci95(dm_values)
                dg_mean, dg_ci = mean_ci95(dg_values)
                lines.extend([
                    f"| mean +/- std | {mean_std(hm_values)} | {mean_std(tm_values)} | "
                    f"{mean_std(dm_values)} pp | {mean_std(hg_values)} | "
                    f"{mean_std(tg_values)} | {mean_std(dg_values)} pp | - |",
                    "",
                    f"Paired delta 95% CI (n=5): MATH {dm_mean:+.4f} +/- {dm_ci:.4f} pp; "
                    f"GSM8K {dg_mean:+.4f} +/- {dg_ci:.4f} pp.",
                    f"Holdout-only mean delta (seeds 43--46): MATH "
                    f"{statistics.mean(dm_values[1:]):+.4f} pp; GSM8K "
                    f"{statistics.mean(dg_values[1:]):+.4f} pp.", "",
                ])
        RESULTS.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines), flush=True)

    # Validate all reused seed-42 artifacts before allocating GPUs.
    for alias in MODEL_INFO:
        for topology in ("hidden", "input", "shared", "output"):
            checkpoint = seed42_checkpoint(alias, topology)
            if not (checkpoint / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(checkpoint / "adapter_model.safetensors")
            if topology != "hidden" and not (
                checkpoint / "affine_vocab_adapter.safetensors"
            ).is_file():
                raise FileNotFoundError(checkpoint / "affine_vocab_adapter.safetensors")
            for task in ("math", "gsm8k"):
                if not seed42_full(alias, topology, task).is_file():
                    raise FileNotFoundError(seed42_full(alias, topology, task))

    runner.checkpoint_complete = checkpoint_complete
    runner.train = train
    runner.exact_rhos = exact_rhos
    runner.write_report = write_report
    runner.main()


if __name__ == "__main__":
    main()
