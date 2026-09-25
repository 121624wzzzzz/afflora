#!/usr/bin/env python
"""Train and fully evaluate pure output A-LoRA with/without an energy constraint."""

from __future__ import annotations

import argparse
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
CHECKPOINTS = HERE / "pure_alora_checkpoints"
OUTPUTS = HERE / "pure_alora_outputs"
LOGS = HERE / "pure_alora_logs"
STATE = HERE / "pure_alora_state.json"
EVENTS = HERE / "pure_alora_events.jsonl"
RESULTS = HERE / "PURE_ALORA_ENERGY_RESULTS.md"
FROZEN_OUTPUTS = HERE / "frozen_baseline_outputs"
NUM_SHARDS = 16
NUM_GPUS = 8
LOCK = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Job:
    label: str
    tau: float
    lam: float

    @property
    def name(self) -> str:
        if self.lam == 0:
            suffix = "unconstrained"
        else:
            suffix = f"energy_tau{safe_float(self.tau)}_l{safe_float(self.lam)}"
        return f"llama31_8b_pure_lmhead_ar16_s1_{suffix}_seed42"

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINTS / self.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--energy-lambda", type=float, required=True)
    return parser.parse_args()


def event(kind: str, **payload: Any) -> None:
    row = {"time": now(), "kind": kind, **payload}
    with LOCK:
        with EVENTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_state(phase: str, **payload: Any) -> None:
    with LOCK:
        STATE.write_text(
            json.dumps({"phase": phase, "updated_at": now(), **payload}, indent=2),
            encoding="utf-8",
        )


def env(gpu: int) -> dict[str, str]:
    value = os.environ.copy()
    value["CUDA_VISIBLE_DEVICES"] = str(gpu)
    value["DS_IGNORE_CUDA_DETECTION"] = "1"
    return value


def checkpoint_complete(job: Job) -> bool:
    files = ("affine_vocab_adapter.safetensors", "affine_vocab_config.json", "run_args.json")
    return all((job.checkpoint / filename).is_file() for filename in files)


def train(job: Job, gpu: int) -> bool:
    if checkpoint_complete(job):
        event("training_reused", job=job.name, gpu=gpu)
        return True
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT), "--model-path", str(MODEL),
        "--train-data", str(TRAIN), "--output-dir", str(job.checkpoint),
        "--variant", "affine_lm_head", "--affine-rank", "16",
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


def dynamic_train_and_evaluate(jobs: tuple[Job, ...]) -> None:
    work: queue.Queue[tuple[str, Job, str | None, int | None] | None] = queue.Queue()
    failures: list[str] = []
    for job in jobs:
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
            except Exception as exc:
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
        raise RuntimeError(f"Pure A-LoRA failures: {failures}")


def merge_outputs(jobs: tuple[Job, ...]) -> None:
    for job in jobs:
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


def exact_rhos(jobs: tuple[Job, ...]) -> dict[str, float]:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    weight = model.lm_head.weight.detach()
    mean = weight.float().mean(dim=0)
    covariance = torch.zeros((weight.shape[1], weight.shape[1]), dtype=torch.float32, device="cuda")
    for begin in range(0, weight.shape[0], 4096):
        centered = weight[begin : begin + 4096].float() - mean
        covariance.addmm_(centered.T, centered)
    denominator = float(covariance.diagonal().sum().item())
    values: dict[str, float] = {}
    for job in jobs:
        state = load_file(str(job.checkpoint / "affine_vocab_adapter.safetensors"))
        up = state["lm_head.affine.up.weight"].float().cuda()
        down = state["lm_head.affine.down.weight"].float().cuda()
        numerator = float(((up.T @ covariance @ up) * (down @ down.T)).sum().item())
        values[job.name] = math.sqrt(max(numerator / denominator, 0.0))
    return values


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def write_report(jobs: tuple[Job, ...], rhos: dict[str, float]) -> None:
    frozen_m = accuracy(FROZEN_OUTPUTS / "math/llama31_8b_frozen_base_full.json", "clean")
    frozen_g = accuracy(FROZEN_OUTPUTS / "gsm8k/llama31_8b_frozen_base_full.json", "full")
    lines = [
        "# Pure Llama-3.1 output A-LoRA energy experiment", "",
        f"Updated: {now()}", "",
        "Both rows are standalone output A-LoRA r16, seed 42, with no hidden LoRA.", "",
        "| config | tau | lambda | final rho | MATH | GSM8K |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| frozen Llama-3.1-8B Base | - | - | - | {frozen_m:.4f}% | {frozen_g:.4f}% |",
    ]
    for job in jobs:
        math_acc = accuracy(full(job, "math"), "clean")
        gsm_acc = accuracy(full(job, "gsm8k"), "full")
        lines.append(
            f"| {job.label} | {job.tau:.8f} | {job.lam:g} | {rhos[job.name]:.8f} | "
            f"{math_acc:.4f}% | {gsm_acc:.4f}% |"
        )
    unconstrained, constrained = jobs
    unc_m = accuracy(full(unconstrained, "math"), "clean")
    unc_g = accuracy(full(unconstrained, "gsm8k"), "full")
    con_m = accuracy(full(constrained, "math"), "clean")
    con_g = accuracy(full(constrained, "gsm8k"), "full")
    lines.extend([
        "", "Constrained minus unconstrained:", "",
        f"- MATH: {con_m-unc_m:+.4f} pp",
        f"- GSM8K: {con_g-unc_g:+.4f} pp",
        "", "A-LoRA minus frozen Base:", "",
        f"- Unconstrained: MATH {unc_m-frozen_m:+.4f} pp; GSM8K {unc_g-frozen_g:+.4f} pp",
        f"- Constrained: MATH {con_m-frozen_m:+.4f} pp; GSM8K {con_g-frozen_g:+.4f} pp",
    ])
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    jobs = (
        Job("pure A-LoRA unconstrained", 0.0, 0.0),
        Job("pure A-LoRA constrained", args.tau, args.energy_lambda),
    )
    for directory in (
        CHECKPOINTS, OUTPUTS / "math", OUTPUTS / "gsm8k",
        LOGS / "training", LOGS / "math", LOGS / "gsm8k",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if EVENTS.exists():
        EVENTS.unlink()
    write_state(
        "dynamic_training_and_evaluation", tau=args.tau,
        energy_lambda=args.energy_lambda, jobs=[job.name for job in jobs],
    )
    dynamic_train_and_evaluate(jobs)
    write_state("merging", tau=args.tau, energy_lambda=args.energy_lambda)
    merge_outputs(jobs)
    write_state("energy_measurement", tau=args.tau, energy_lambda=args.energy_lambda)
    rhos = exact_rhos(jobs)
    write_report(jobs, rhos)
    write_state("complete", tau=args.tau, energy_lambda=args.energy_lambda, rhos=rhos, result=str(RESULTS))
    event("pure_alora_complete", result=str(RESULTS))


if __name__ == "__main__":
    main()
