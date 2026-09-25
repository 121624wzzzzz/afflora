#!/usr/bin/env python
"""Sweep weaker affine-energy thresholds on Qwen2.5-3B and Qwen3-4B."""

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
OLD_OUTPUTS = HERE / "qwen_multiseed_scale_outputs"
OLD_STATE = HERE / "qwen_multiseed_scale_state.json"

CHECKPOINTS = HERE / "qwen_tau_scale_checkpoints"
OUTPUTS = HERE / "qwen_tau_scale_outputs"
LOGS = HERE / "qwen_tau_scale_logs"
STATE = HERE / "qwen_tau_scale_state.json"
EVENTS = HERE / "qwen_tau_scale_events.jsonl"
RESULTS = HERE / "QWEN_TAU_SCALE_RESULTS.md"

ENERGY_LAMBDA = 100.0
TAUS = (0.0125, 0.025, 0.05)


def load_base():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("qwen_tau_base", SOURCE)
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
    tau: float

    @property
    def variant(self) -> str:
        if self.topology == "input":
            return "affine_input_plus_hidden_lora"
        return "affine_input_lm_head_plus_hidden_lora"

    @property
    def name(self) -> str:
        return (
            f"{self.model_alias}_{self.topology}_ar16_s1_hr4_energy_"
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

# Longest-processing-time order keeps the initial six 4B jobs resident, then
# fills the remaining two GPUs with 3B jobs.  The shared queue remains fully
# non-blocking as jobs finish.
JOBS = tuple(
    [Job("qwen3_4b", topology, tau) for topology in ("input", "shared") for tau in TAUS]
    + [Job("qwen25_3b", topology, tau) for topology in ("input", "shared") for tau in TAUS]
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
        if job.topology == "shared":
            cmd.extend(["--affine-lm-head-bias", "--tie-affine-input-lm-head-adapters"])
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
                model.get_input_embeddings().weight.detach(), centered=False,
            )
            for job in (item for item in JOBS if item.model_alias == alias):
                affine_state = load_file(
                    str(job.checkpoint / "affine_vocab_adapter.safetensors")
                )
                values[job.name] = {
                    "input": runner.raw_rho(
                        affine_state, "model.embed_tokens.affine", stats,
                    )
                }
            del model, stats
            torch.cuda.empty_cache()
        return values

    old_rhos = json.loads(OLD_STATE.read_text(encoding="utf-8"))["rhos"]

    def old_name(alias: str, configuration: str) -> str:
        if configuration == "hidden":
            return f"{alias}_hidden_hr4_seed42"
        if configuration == "shared_unconstrained":
            return f"{alias}_shared_ar16_s1_hr4_unconstrained_seed42"
        topology = configuration.split("_", 1)[0]
        return f"{alias}_{topology}_ar16_s1_hr4_energy_tau0p00625_l100_seed42"

    def old_score(alias: str, configuration: str, task: str) -> float:
        key = "clean" if task == "math" else "full"
        return accuracy(OLD_OUTPUTS / task / f"{old_name(alias, configuration)}_full.json", key)

    def score(job: Job, task: str) -> float:
        key = "clean" if task == "math" else "full"
        return accuracy(runner.full(job, task), key)

    def write_report(rhos):  # noqa: ANN001, ANN202
        lines = [
            "# Qwen 3B/4B affine-energy tau sweep", "",
            f"Updated: {runner.now()}", "",
            "All new rows use MetaMathQA-40K, one epoch, effective batch 16, "
            "hidden LoRA r4, A-LoRA r16 scale 1, seed 42, and lambda=100.", "",
        ]
        for alias in ("qwen25_3b", "qwen3_4b"):
            bm, bg = old_score(alias, "hidden", "math"), old_score(alias, "hidden", "gsm8k")
            lines.extend([
                f"## {alias}", "",
                "| topology | tau | measured rho | MATH | delta vs hidden | GSM8K | delta vs hidden |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| hidden r4 | - | - | {bm:.4f}% | - | {bg:.4f}% | - |",
            ])
            for topology in ("input", "shared"):
                if topology == "shared":
                    name = old_name(alias, "shared_unconstrained")
                    m = old_score(alias, "shared_unconstrained", "math")
                    g = old_score(alias, "shared_unconstrained", "gsm8k")
                    lines.append(
                        f"| shared | unconstrained | {old_rhos[name]['input']:.8f} | "
                        f"{m:.4f}% | {m-bm:+.4f} pp | {g:.4f}% | {g-bg:+.4f} pp |"
                    )
                for tau in (0.05, 0.025, 0.0125):
                    job = next(
                        item for item in JOBS
                        if item.model_alias == alias and item.topology == topology and item.tau == tau
                    )
                    m, g = score(job, "math"), score(job, "gsm8k")
                    lines.append(
                        f"| {topology} | {tau:.4f} | {rhos[job.name]['input']:.8f} | "
                        f"{m:.4f}% | {m-bm:+.4f} pp | {g:.4f}% | {g-bg:+.4f} pp |"
                    )
                name = old_name(alias, f"{topology}_tau00625")
                m = old_score(alias, f"{topology}_tau00625", "math")
                g = old_score(alias, f"{topology}_tau00625", "gsm8k")
                lines.append(
                    f"| {topology} | 0.00625 | {old_rhos[name]['input']:.8f} | "
                    f"{m:.4f}% | {m-bm:+.4f} pp | {g:.4f}% | {g-bg:+.4f} pp |"
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
