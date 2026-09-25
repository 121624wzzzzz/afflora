#!/usr/bin/env python
"""Paired Qwen multi-seed confirmation and Qwen 3B/4B scale extension.

The runner intentionally uses the dynamic single-GPU queue from the topology
sweep: a GPU takes the next training/evaluation item as soon as it becomes
free.  Checkpoints and evaluation shards are reusable after interruption.
"""

from __future__ import annotations

import importlib.util
import json
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


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_scale_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Job:
    model_alias: str
    topology: str
    variant: str
    seed: int = 42
    constrained: bool = False
    lm_head_bias: bool = False
    tie_adapters: bool = False

    @property
    def name(self) -> str:
        if self.model_alias == "qwen25_15b":
            return (
                f"{self.model_alias}_input_ar16_s1_hr4_energy_"
                f"tau0p00625_l100_seed{self.seed}"
            )
        if self.topology == "hidden":
            return f"{self.model_alias}_hidden_hr4_seed{self.seed}"
        suffix = "energy_tau0p00625_l100" if self.constrained else "unconstrained"
        return f"{self.model_alias}_{self.topology}_ar16_s1_hr4_{suffix}_seed{self.seed}"

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINTS / self.name

    @property
    def model_path(self) -> Path:
        return MODEL_INFO[self.model_alias]["path"]

    @property
    def has_affine(self) -> bool:
        return self.topology != "hidden"


MODEL_INFO = {
    "qwen25_15b": {
        "path": ROOT.parent / "models/Qwen2.5-1.5B-Base",
        "batch": 16,
        "accum": 1,
    },
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

# Existing seed-42 treatment and all three hidden baselines are reused only in
# reporting.  The ten jobs below are genuinely new training runs.
JOBS = (
    Job("qwen25_15b", "input", "affine_input_plus_hidden_lora", seed=43, constrained=True),
    Job("qwen25_15b", "input", "affine_input_plus_hidden_lora", seed=44, constrained=True),
    Job("qwen25_3b", "hidden", "hidden_lora"),
    Job(
        "qwen25_3b", "shared", "affine_input_lm_head_plus_hidden_lora",
        lm_head_bias=True, tie_adapters=True,
    ),
    Job("qwen25_3b", "input", "affine_input_plus_hidden_lora", constrained=True),
    Job(
        "qwen25_3b", "shared", "affine_input_lm_head_plus_hidden_lora",
        constrained=True, lm_head_bias=True, tie_adapters=True,
    ),
    Job("qwen3_4b", "hidden", "hidden_lora"),
    Job(
        "qwen3_4b", "shared", "affine_input_lm_head_plus_hidden_lora",
        lm_head_bias=True, tie_adapters=True,
    ),
    Job("qwen3_4b", "input", "affine_input_plus_hidden_lora", constrained=True),
    Job(
        "qwen3_4b", "shared", "affine_input_lm_head_plus_hidden_lora",
        constrained=True, lm_head_bias=True, tie_adapters=True,
    ),
)

CHECKPOINTS = HERE / "qwen_multiseed_scale_checkpoints"
OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
LOGS = HERE / "qwen_multiseed_scale_logs"
STATE = HERE / "qwen_multiseed_scale_state.json"
EVENTS = HERE / "qwen_multiseed_scale_events.jsonl"
RESULTS = HERE / "QWEN_MULTISEED_SCALE_RESULTS.md"
TAU = 0.00625
ENERGY_LAMBDA = 100.0


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"


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
            cmd.extend(["--affine-rank", "16", "--affine-alpha", "16"])
        if job.constrained:
            cmd.extend([
                "--affine-energy-tau", str(TAU),
                "--affine-energy-lambda", str(ENERGY_LAMBDA),
            ])
        if job.lm_head_bias:
            cmd.append("--affine-lm-head-bias")
        if job.tie_adapters:
            cmd.append("--tie-affine-input-lm-head-adapters")
        log = LOGS / "training" / f"{job.name}.log"
        runner.event(
            "training_started", job=job.name, gpu=gpu,
            topology=job.topology, constrained=job.constrained, seed=job.seed,
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
            affine_jobs = [job for job in JOBS if job.model_alias == alias and job.has_affine]
            if not affine_jobs:
                continue
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_INFO[alias]["path"], torch_dtype=torch.bfloat16,
            ).cuda().eval()
            input_stats = runner.matrix_stats(
                model.get_input_embeddings().weight.detach(), centered=False,
            )
            for job in affine_jobs:
                state = load_file(str(job.checkpoint / "affine_vocab_adapter.safetensors"))
                values[job.name] = {
                    "input": runner.raw_rho(
                        state, "model.embed_tokens.affine", input_stats,
                    )
                }
            del model, input_stats
            torch.cuda.empty_cache()
        return values

    hidden_root = (
        ROOT / "corrected_math_evaluation/model_families/qwen25/"
        "small_models/hidden_mergeable_rank_sweep/outputs"
    )
    old_qwen_outputs = HERE / "qwen_topology_outputs"

    def existing_hidden(seed: int, task: str) -> float:
        key = "clean" if task == "math" else "full"
        return accuracy(
            hidden_root / task / f"qwen25_15b_hidden_hr4_seed{seed}_full.json", key,
        )

    def existing_treatment_seed42(task: str) -> float:
        key = "clean" if task == "math" else "full"
        return accuracy(
            old_qwen_outputs / task /
            "qwen25_15b_input_ar16_s1_hr4_energy_tau0p00625_l100_seed42_full.json",
            key,
        )

    def job_score(job: Job, task: str) -> float:
        key = "clean" if task == "math" else "full"
        return accuracy(runner.full(job, task), key)

    def write_report(rhos):  # noqa: ANN001, ANN202
        lines = [
            "# Qwen paired multi-seed and scale extension", "",
            f"Updated: {runner.now()}", "",
            "Training: MetaMathQA-40K, one epoch, effective batch 16, hidden LoRA r4. "
            "A-LoRA uses rank 16 and scale 1. Constrained rows use tau=0.00625 and lambda=100.",
            "", "## Paired Qwen2.5-1.5B input-constrained confirmation", "",
            "| seed | hidden MATH | constrained MATH | paired delta | hidden GSM8K | constrained GSM8K | paired delta | rho |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        seed42_rho = 0.006252847920878559
        hidden_math: list[float] = []
        treatment_math: list[float] = []
        delta_math: list[float] = []
        hidden_gsm: list[float] = []
        treatment_gsm: list[float] = []
        delta_gsm: list[float] = []
        for seed in (42, 43, 44):
            hm, hg = existing_hidden(seed, "math"), existing_hidden(seed, "gsm8k")
            if seed == 42:
                tm, tg, rho = (
                    existing_treatment_seed42("math"),
                    existing_treatment_seed42("gsm8k"), seed42_rho,
                )
            else:
                job = next(item for item in JOBS if item.model_alias == "qwen25_15b" and item.seed == seed)
                tm, tg, rho = job_score(job, "math"), job_score(job, "gsm8k"), rhos[job.name]["input"]
            hidden_math.append(hm); treatment_math.append(tm); delta_math.append(tm - hm)
            hidden_gsm.append(hg); treatment_gsm.append(tg); delta_gsm.append(tg - hg)
            lines.append(
                f"| {seed} | {hm:.4f}% | {tm:.4f}% | {tm-hm:+.4f} pp | "
                f"{hg:.4f}% | {tg:.4f}% | {tg-hg:+.4f} pp | {rho:.8f} |"
            )
        lines.extend([
            f"| mean +/- std | {mean_std(hidden_math)} | {mean_std(treatment_math)} | "
            f"{mean_std(delta_math)} pp | {mean_std(hidden_gsm)} | {mean_std(treatment_gsm)} | "
            f"{mean_std(delta_gsm)} pp | - |",
            "", "## Qwen 3B/4B scale extension (seed 42)", "",
            "| model | configuration | rho | MATH | delta vs hidden | GSM8K | delta vs hidden |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for alias in ("qwen25_3b", "qwen3_4b"):
            model_jobs = [job for job in JOBS if job.model_alias == alias]
            hidden = next(job for job in model_jobs if job.topology == "hidden")
            bm, bg = job_score(hidden, "math"), job_score(hidden, "gsm8k")
            lines.append(f"| {alias} | hidden r4 | - | {bm:.4f}% | - | {bg:.4f}% | - |")
            for job in model_jobs:
                if job.topology == "hidden":
                    continue
                m, g = job_score(job, "math"), job_score(job, "gsm8k")
                label = f"{job.topology} {'constrained' if job.constrained else 'unconstrained'}"
                lines.append(
                    f"| {alias} | {label} | {rhos[job.name]['input']:.8f} | "
                    f"{m:.4f}% | {m-bm:+.4f} pp | {g:.4f}% | {g-bg:+.4f} pp |"
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
