#!/usr/bin/env python
"""Sweep A-LoRA energy radius and penalty strength on Llama-3.1-8B.

Training and full MATH/GSM8K evaluation share one dynamic queue.  As soon as a
GPU finishes its training job it starts evaluating completed checkpoints; it
does not wait for the other training jobs in the batch.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
MODEL = ROOT.parent / "models/Llama-3.1-8B-Base"
TRAIN = ROOT / "data/metamathqa_40k/train.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"
CROSS = ROOT / "reviewer_followup/llama_cross_version"
CHECKPOINTS = HERE / "checkpoints"
OUTPUTS = HERE / "strength_sweep_outputs"
LOGS = HERE / "strength_sweep_logs"
STATE = HERE / "energy_strength_sweep_state.json"
EVENTS = HERE / "energy_strength_sweep_events.jsonl"
RESULTS = HERE / "ENERGY_STRENGTH_SWEEP_RESULTS.md"
NUM_SHARDS = 16
NUM_GPUS = 8
LOCK = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Job:
    axis: str
    tau: float
    lam: float

    @property
    def name(self) -> str:
        return (
            "llama31_8b_lmhead_ar16_s1_hr4_energy_"
            f"tau{safe_float(self.tau)}_l{safe_float(self.lam)}_seed42"
        )

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINTS / self.name


# One-factor-at-a-time cross: lambda controls enforcement stiffness; tau controls
# the actual allowed update radius.  The existing tau=0.00729014, lambda=100 run
# and the unconstrained run are reused in the final report.
JOBS = (
    Job("lambda", 0.00729014, 10.0),
    Job("lambda", 0.00729014, 30.0),
    Job("lambda", 0.00729014, 300.0),
    Job("lambda", 0.00729014, 1000.0),
    Job("tau", 0.00519388, 100.0),
    Job("tau", 0.00625, 100.0),
    Job("tau", 0.0085, 100.0),
    Job("tau", 0.0095, 100.0),
)


def event(kind: str, **payload: Any) -> None:
    row = {"time": now(), "kind": kind, **payload}
    with LOCK:
        with EVENTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_state(phase: str, **payload: Any) -> None:
    row = {"phase": phase, "updated_at": now(), **payload}
    with LOCK:
        STATE.write_text(json.dumps(row, indent=2), encoding="utf-8")


def env(gpu: int) -> dict[str, str]:
    value = os.environ.copy()
    value["CUDA_VISIBLE_DEVICES"] = str(gpu)
    value["DS_IGNORE_CUDA_DETECTION"] = "1"
    return value


def checkpoint_complete(job: Job) -> bool:
    files = ("adapter_model.safetensors", "affine_vocab_adapter.safetensors", "run_args.json")
    return all((job.checkpoint / filename).is_file() for filename in files)


def train(job: Job, gpu: int) -> bool:
    if checkpoint_complete(job):
        event("training_reused", job=job.name, gpu=gpu)
        return True
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT), "--model-path", str(MODEL),
        "--train-data", str(TRAIN), "--output-dir", str(job.checkpoint),
        "--variant", "affine_lm_head_plus_hidden_lora",
        "--hidden-lora-rank", "4", "--hidden-lora-alpha", "8",
        "--hidden-lora-dropout", "0.05", "--affine-rank", "16",
        "--affine-alpha", "16", "--affine-energy-tau", str(job.tau),
        "--affine-energy-lambda", str(job.lam), "--max-seq-len", "1024",
        "--per-device-train-batch-size", "4", "--gradient-accumulation-steps", "4",
        "--learning-rate", "2e-4", "--num-train-epochs", "1",
        "--logging-steps", "10", "--bf16", "--gradient-checkpointing",
        "--seed", "42", "--master-dtype", "fp32", "--base-dtype", "bf16",
        "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]
    log = LOGS / "training" / f"{job.name}.log"
    event("training_started", job=job.name, gpu=gpu, tau=job.tau, lam=job.lam)
    with log.open("w", encoding="utf-8") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and checkpoint_complete(job)
    event("training_finished", job=job.name, gpu=gpu, returncode=rc, success=ok)
    return ok


def shard(job: Job, task: str, index: int) -> Path:
    return OUTPUTS / task / f"{job.name}_shard{index}.json"


def full(job: Job, task: str) -> Path:
    return OUTPUTS / task / f"{job.name}_full.json"


def eval_one(job: Job, task: str, index: int, gpu: int) -> bool:
    target = shard(job, task, index)
    if target.is_file():
        return True
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    for attempt, batch in enumerate((64, 32, 16), 1):
        cmd = [
            str(PYTHON), str(script), "--model-path", str(MODEL),
            "--run-dir", str(job.checkpoint), "--eval-data", str(data),
            "--output-file", str(target), "--batch-size", str(batch),
            "--max-new-tokens", "512", "--num-shards", str(NUM_SHARDS),
            "--shard-index", str(index),
        ]
        log = LOGS / task / f"{job.name}_shard{index}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(cmd, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc == 0 and target.is_file():
            event("eval_shard_finished", job=job.name, task=task, shard=index, gpu=gpu)
            return True
        event(
            "eval_shard_retry", job=job.name, task=task, shard=index,
            gpu=gpu, attempt=attempt, batch=batch, returncode=rc,
        )
    return False


def run_dynamic_queue() -> None:
    work: queue.Queue[tuple[str, Job, str | None, int | None] | None] = queue.Queue()
    failures: list[str] = []
    for job in JOBS:
        work.put(("train", job, None, None))

    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            kind, job, task, index = item
            try:
                if kind == "train":
                    if train(job, gpu):
                        # Enqueue before marking training done so queue.join() cannot
                        # observe a transient zero and return early.
                        for eval_task in ("math", "gsm8k"):
                            for shard_index in range(NUM_SHARDS):
                                work.put(("eval", job, eval_task, shard_index))
                    else:
                        with LOCK:
                            failures.append(f"train:{job.name}")
                else:
                    assert task is not None and index is not None
                    if not eval_one(job, task, index, gpu):
                        with LOCK:
                            failures.append(f"eval:{job.name}:{task}:{index}")
            except Exception as exc:  # keep other GPUs productive and record failure
                event("worker_exception", gpu=gpu, job=job.name, task=task, error=repr(exc))
                with LOCK:
                    failures.append(f"exception:{job.name}:{task}:{index}:{exc!r}")
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in range(NUM_GPUS)]
    for thread in threads:
        thread.start()
    work.join()
    for _ in threads:
        work.put(None)
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Sweep failures: {failures}")


def merge_outputs() -> None:
    for job in JOBS:
        for task in ("math", "gsm8k"):
            merge = MATH_MERGE if task == "math" else GSM_MERGE
            expected = "5000" if task == "math" else "1319"
            inputs = [str(shard(job, task, i)) for i in range(NUM_SHARDS)]
            rc = subprocess.run(
                [str(PYTHON), str(merge), "--inputs", *inputs,
                 "--output", str(full(job, task)), "--expected-samples", expected],
                cwd=ROOT,
            ).returncode
            if rc != 0:
                raise RuntimeError(f"Merge failed: {job.name}:{task}")


def exact_rhos() -> dict[str, float]:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    weight = model.lm_head.weight.detach()
    vocab = weight.shape[0]
    mean = weight.float().mean(dim=0)
    covariance = torch.zeros((weight.shape[1], weight.shape[1]), dtype=torch.float32, device="cuda")
    for begin in range(0, vocab, 4096):
        centered = weight[begin : begin + 4096].float() - mean
        covariance.addmm_(centered.T, centered)
    denominator = float(covariance.diagonal().sum().item())
    values: dict[str, float] = {}
    for job in JOBS:
        state = load_file(str(job.checkpoint / "affine_vocab_adapter.safetensors"))
        up = state["lm_head.affine.up.weight"].float().cuda()
        down = state["lm_head.affine.down.weight"].float().cuda()
        numerator = float(((up.T @ covariance @ up) * (down @ down.T)).sum().item())
        values[job.name] = math.sqrt(max(numerator / denominator, 0.0))
    del model, covariance, weight
    torch.cuda.empty_cache()
    return values


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def write_report(rhos: dict[str, float]) -> None:
    base_m = accuracy(CROSS / "outputs/math/llama31_8b_hidden_hr4_seed42_full.json", "clean")
    base_g = accuracy(CROSS / "outputs/gsm8k/llama31_8b_hidden_hr4_seed42_full.json", "full")
    unc_m = accuracy(CROSS / "outputs/math/llama31_8b_lmhead_ar16_s1_hr4_seed42_full.json", "clean")
    unc_g = accuracy(CROSS / "outputs/gsm8k/llama31_8b_lmhead_ar16_s1_hr4_seed42_full.json", "full")
    existing_m = accuracy(HERE / "outputs/math/llama31_8b_lmhead_ar16_s1_hr4_energy_tau0p00729_l100_seed42_full.json", "clean")
    existing_g = accuracy(HERE / "outputs/gsm8k/llama31_8b_lmhead_ar16_s1_hr4_energy_tau0p00729_l100_seed42_full.json", "full")
    lines = [
        "# Llama-3.1 A-LoRA energy-constraint strength sweep", "",
        f"Updated: {now()}", "",
        "All A-LoRA rows use hidden LoRA r4 + output A-LoRA r16, seed 42. ",
        "The sweep changes only the energy trust-region hyperparameters.", "",
        "| axis | tau | lambda | final rho | MATH | delta vs hidden | GSM8K | delta vs hidden |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| baseline | - | - | - | {base_m:.4f}% | - | {base_g:.4f}% | - |",
        f"| unconstrained | - | 0 | 0.01006077 | {unc_m:.4f}% | {unc_m-base_m:+.4f} pp | {unc_g:.4f}% | {unc_g-base_g:+.4f} pp |",
        f"| existing | 0.00729014 | 100 | 0.00724376 | {existing_m:.4f}% | {existing_m-base_m:+.4f} pp | {existing_g:.4f}% | {existing_g-base_g:+.4f} pp |",
    ]
    for job in sorted(JOBS, key=lambda value: (value.axis, value.tau, value.lam)):
        math_acc = accuracy(full(job, "math"), "clean")
        gsm_acc = accuracy(full(job, "gsm8k"), "full")
        lines.append(
            f"| {job.axis} | {job.tau:.8f} | {job.lam:g} | {rhos[job.name]:.8f} | "
            f"{math_acc:.4f}% | {math_acc-base_m:+.4f} pp | "
            f"{gsm_acc:.4f}% | {gsm_acc-base_g:+.4f} pp |"
        )
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (
        CHECKPOINTS, OUTPUTS / "math", OUTPUTS / "gsm8k",
        LOGS / "training", LOGS / "math", LOGS / "gsm8k",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if EVENTS.exists():
        EVENTS.unlink()
    write_state("dynamic_training_and_evaluation", jobs=[job.name for job in JOBS])
    run_dynamic_queue()
    write_state("merging")
    merge_outputs()
    write_state("energy_measurement")
    rhos = exact_rhos()
    write_report(rhos)
    write_state("complete", rhos=rhos, result=str(RESULTS))
    event("sweep_complete", result=str(RESULTS))


if __name__ == "__main__":
    main()
