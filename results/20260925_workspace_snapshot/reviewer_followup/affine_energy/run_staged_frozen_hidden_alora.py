#!/usr/bin/env python
"""Add output A-LoRA to a trained, frozen hidden-LoRA checkpoint."""

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
HIDDEN = ROOT / "reviewer_followup/llama_cross_version/checkpoints/llama31_8b_hidden_hr4_seed42"
CROSS = ROOT / "reviewer_followup/llama_cross_version"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"
CHECKPOINTS = HERE / "staged_checkpoints"
OUTPUTS = HERE / "staged_outputs"
LOGS = HERE / "staged_logs"
STATE = HERE / "staged_frozen_hidden_state.json"
EVENTS = HERE / "staged_frozen_hidden_events.jsonl"
RESULTS = HERE / "STAGED_FROZEN_HIDDEN_ALORA_RESULTS.md"
NUM_SHARDS = 16
LOCK = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Job:
    label: str
    tau: float
    lam: float

    @property
    def name(self) -> str:
        suffix = "unconstrained" if self.lam == 0 else "energy_tau0p00625_l100"
        return f"llama31_8b_frozen_hr4_then_lmhead_ar16_s1_{suffix}_seed42"

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINTS / self.name


JOBS = (
    Job("staged A unconstrained", 0.0, 0.0),
    Job("staged A constrained", 0.00625, 100.0),
)


def event(kind: str, **payload: Any) -> None:
    with LOCK:
        with EVENTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "kind": kind, **payload}) + "\n")


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
    files = (
        "adapter_model.safetensors", "adapter_config.json",
        "affine_vocab_adapter.safetensors", "affine_vocab_config.json", "run_args.json",
    )
    return all((job.checkpoint / filename).is_file() for filename in files)


def frozen_hidden_identical(job: Job) -> bool:
    source = load_file(str(HIDDEN / "adapter_model.safetensors"))
    result = load_file(str(job.checkpoint / "adapter_model.safetensors"))
    return source.keys() == result.keys() and all(torch.equal(source[key], result[key]) for key in source)


def train(job: Job, gpu: int) -> bool:
    if checkpoint_complete(job):
        if not frozen_hidden_identical(job):
            raise RuntimeError(f"Frozen hidden adapter changed in reused checkpoint: {job.name}")
        event("training_reused", job=job.name, gpu=gpu)
        return True
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT), "--model-path", str(MODEL),
        "--train-data", str(TRAIN), "--output-dir", str(job.checkpoint),
        "--variant", "affine_lm_head_plus_hidden_lora",
        "--initial-hidden-lora-adapter", str(HIDDEN), "--freeze-initial-hidden-lora",
        "--affine-rank", "16", "--affine-alpha", "16",
        "--affine-energy-tau", str(job.tau), "--affine-energy-lambda", str(job.lam),
        "--max-seq-len", "1024", "--per-device-train-batch-size", "4",
        "--gradient-accumulation-steps", "4", "--learning-rate", "2e-4",
        "--num-train-epochs", "1", "--logging-steps", "10", "--bf16",
        "--seed", "42", "--master-dtype", "fp32", "--base-dtype", "bf16",
        "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]
    log = LOGS / "training" / f"{job.name}.log"
    event("training_started", job=job.name, gpu=gpu, tau=job.tau, lam=job.lam)
    with log.open("w", encoding="utf-8") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and checkpoint_complete(job)
    if ok:
        ok = frozen_hidden_identical(job)
    event("training_finished", job=job.name, gpu=gpu, returncode=rc, success=ok, hidden_identical=ok)
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
        event("eval_retry", job=job.name, task=task, shard=index, gpu=gpu, attempt=attempt, returncode=rc)
    return False


def dynamic_run() -> None:
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
                event("exception", job=job.name, task=task, index=index, gpu=gpu, error=repr(exc))
                with LOCK:
                    failures.append(f"exception:{job.name}:{task}:{index}:{exc!r}")
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in range(8)]
    for thread in threads:
        thread.start()
    work.join()
    for _ in threads:
        work.put(None)
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Staged run failures: {failures}")


def merge_outputs() -> None:
    for job in JOBS:
        for task in ("math", "gsm8k"):
            script = MATH_MERGE if task == "math" else GSM_MERGE
            expected = "5000" if task == "math" else "1319"
            inputs = [str(shard(job, task, index)) for index in range(NUM_SHARDS)]
            rc = subprocess.run(
                [str(PYTHON), str(script), "--inputs", *inputs,
                 "--output", str(full(job, task)), "--expected-samples", expected],
                cwd=ROOT,
            ).returncode
            if rc != 0:
                raise RuntimeError(f"Merge failed: {job.name}:{task}")


def exact_rhos() -> dict[str, float]:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    weight = model.lm_head.weight.detach()
    mean = weight.float().mean(dim=0)
    covariance = torch.zeros((weight.shape[1], weight.shape[1]), dtype=torch.float32, device="cuda")
    for begin in range(0, weight.shape[0], 4096):
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
    return values


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def report(rhos: dict[str, float]) -> None:
    base_m = accuracy(CROSS / "outputs/math/llama31_8b_hidden_hr4_seed42_full.json", "clean")
    base_g = accuracy(CROSS / "outputs/gsm8k/llama31_8b_hidden_hr4_seed42_full.json", "full")
    joint_m = accuracy(CROSS / "outputs/math/llama31_8b_lmhead_ar16_s1_hr4_seed42_full.json", "clean")
    joint_g = accuracy(CROSS / "outputs/gsm8k/llama31_8b_lmhead_ar16_s1_hr4_seed42_full.json", "full")
    lines = [
        "# Frozen-hidden staged A-LoRA result", "", f"Updated: {now()}", "",
        "The hidden LoRA r4 checkpoint is loaded and frozen exactly; only output A-LoRA r16 trains.", "",
        "| config | final rho | MATH | delta vs hidden | GSM8K | delta vs hidden |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| hidden r4 baseline | - | {base_m:.4f}% | - | {base_g:.4f}% | - |",
        f"| joint-from-scratch hidden+A | 0.01006077 | {joint_m:.4f}% | {joint_m-base_m:+.4f} pp | {joint_g:.4f}% | {joint_g-base_g:+.4f} pp |",
    ]
    for job in JOBS:
        math_acc = accuracy(full(job, "math"), "clean")
        gsm_acc = accuracy(full(job, "gsm8k"), "full")
        lines.append(
            f"| {job.label} | {rhos[job.name]:.8f} | {math_acc:.4f}% | {math_acc-base_m:+.4f} pp | "
            f"{gsm_acc:.4f}% | {gsm_acc-base_g:+.4f} pp |"
        )
    lines.extend(["", "Frozen-hidden integrity:", ""])
    for job in JOBS:
        lines.append(f"- {job.label}: hidden adapter tensor-exact = {frozen_hidden_identical(job)}")
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    for directory in (
        CHECKPOINTS, OUTPUTS / "math", OUTPUTS / "gsm8k",
        LOGS / "training", LOGS / "math", LOGS / "gsm8k",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if EVENTS.exists():
        EVENTS.unlink()
    write_state("dynamic_training_and_evaluation", jobs=[job.name for job in JOBS])
    dynamic_run()
    write_state("merging")
    merge_outputs()
    write_state("energy_measurement")
    rhos = exact_rhos()
    report(rhos)
    write_state("complete", rhos=rhos, result=str(RESULTS))
    event("complete", result=str(RESULTS))


if __name__ == "__main__":
    main()
