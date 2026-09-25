#!/usr/bin/env python
"""Three-seed full MATH/GSM8K confirmation for tuned Llama-3.2 A-LoRA."""

from __future__ import annotations

import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reviewer_followup/llama32_tuned_confirmation"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
MODEL = ROOT.parent / "models/Llama-3.2-3B-Base"
TRAIN = ROOT / "data/metamathqa_40k/train.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"
EXISTING_TUNED = ROOT / "reviewer_followup/llama_affine_lr_sweep/checkpoints"
BASE_OUTPUTS = ROOT / "reviewer_followup/llama_cross_version/outputs"
CHECKPOINTS = HERE / "checkpoints"
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
STATE_FILE = HERE / "state.json"
EVENTS_FILE = HERE / "events.jsonl"
RESULTS_FILE = HERE / "RESULTS.md"
SEEDS = (42, 43, 44)
NUM_SHARDS = 16
GPUS = tuple(range(8))
LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    seed: int

    @property
    def name(self) -> str:
        return f"llama32_3b_mergeable_ar32_s1_hr4_alr0p5_seed{self.seed}"

    @property
    def checkpoint_dir(self) -> Path:
        return (EXISTING_TUNED if self.seed == 42 else CHECKPOINTS) / self.name


JOBS = [Job(seed) for seed in SEEDS]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def event(kind: str, **fields: Any) -> None:
    with LOCK:
        with EVENTS_FILE.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "kind": kind, **fields}) + "\n")


def run_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DS_IGNORE_CUDA_DETECTION"] = "1"
    env["HF_HOME"] = str(HERE / "hf_cache/home")
    env["HF_DATASETS_CACHE"] = str(HERE / "hf_cache/datasets")
    env["XDG_CACHE_HOME"] = str(HERE / "hf_cache/xdg")
    return env


def checkpoint_complete(job: Job) -> bool:
    files = (
        "adapter_model.safetensors", "adapter_config.json",
        "affine_vocab_adapter.safetensors", "affine_vocab_config.json", "run_args.json",
    )
    return all((job.checkpoint_dir / f).is_file() for f in files)


def full_path(job: Job, task: str) -> Path:
    return OUTPUTS / task / f"{job.name}_full.json"


def full_complete(job: Job, task: str) -> bool:
    expected = 5000 if task == "math" else 1319
    path = full_path(job, task)
    if not path.is_file():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8")); rows = d["results"]
        return d["full"]["num_samples"] == expected and len(rows) == expected
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def train_command(job: Job) -> list[str]:
    return [
        str(PYTHON), str(TRAIN_SCRIPT), "--model-path", str(MODEL),
        "--train-data", str(TRAIN), "--output-dir", str(job.checkpoint_dir),
        "--variant", "affine_input_lm_head_plus_hidden_lora",
        "--hidden-lora-rank", "4", "--hidden-lora-alpha", "8",
        "--hidden-lora-dropout", "0.05", "--affine-rank", "32",
        "--affine-alpha", "32", "--affine-learning-rate-scale", "0.5",
        "--tie-affine-input-lm-head-adapters", "--affine-lm-head-bias",
        "--max-seq-len", "1024", "--per-device-train-batch-size", "8",
        "--gradient-accumulation-steps", "2", "--learning-rate", "2e-4",
        "--num-train-epochs", "1", "--logging-steps", "10", "--bf16",
        "--gradient-checkpointing", "--seed", str(job.seed),
        "--master-dtype", "fp32", "--base-dtype", "bf16",
        "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]


def train_one(job: Job, gpu: int) -> bool:
    if checkpoint_complete(job):
        event("training_skip_complete", job=job.name, reused=job.seed == 42)
        return True
    for attempt in (1, 2):
        event("training_start", job=job.name, gpu=gpu, attempt=attempt)
        log = LOGS / "train" / f"{job.name}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(
                train_command(job), cwd=ROOT, env=run_env(gpu),
                stdout=stream, stderr=subprocess.STDOUT,
            ).returncode
        if rc == 0 and checkpoint_complete(job):
            event("training_finish", job=job.name, gpu=gpu, attempt=attempt)
            return True
        event("training_failure", job=job.name, gpu=gpu, attempt=attempt, returncode=rc)
    return False


def shard_path(job: Job, task: str, shard: int) -> Path:
    return OUTPUTS / task / f"{job.name}_shard{shard}.json"


def eval_shard(job: Job, task: str, shard: int, gpu: int) -> bool:
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    batches = (128, 64, 32)
    for attempt, batch in enumerate(batches, 1):
        event(
            "eval_shard_start", job=job.name, task=task, shard=shard,
            gpu=gpu, attempt=attempt, batch_size=batch,
        )
        cmd = [
            str(PYTHON), str(script), "--model-path", str(MODEL),
            "--run-dir", str(job.checkpoint_dir), "--eval-data", str(data),
            "--output-file", str(shard_path(job, task, shard)),
            "--batch-size", str(batch), "--max-new-tokens", "512",
            "--num-shards", str(NUM_SHARDS), "--shard-index", str(shard),
        ]
        log = LOGS / task / f"{job.name}_shard{shard}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(
                cmd, cwd=ROOT, env=run_env(gpu), stdout=stream, stderr=subprocess.STDOUT
            ).returncode
        out = shard_path(job, task, shard)
        if rc == 0 and out.is_file() and out.stat().st_size > 0:
            event(
                "eval_shard_finish", job=job.name, task=task, shard=shard,
                gpu=gpu, attempt=attempt, batch_size=batch,
            )
            return True
        event(
            "eval_shard_failure", job=job.name, task=task, shard=shard,
            gpu=gpu, attempt=attempt, batch_size=batch, returncode=rc,
        )
    return False


def enqueue_eval(work: queue.Queue[Any], job: Job) -> None:
    for task in ("math", "gsm8k"):
        if not full_complete(job, task):
            for shard in range(NUM_SHARDS):
                work.put(("eval", job, task, shard))


def metric(job: Job, task: str) -> float | None:
    if not full_complete(job, task):
        return None
    d = json.loads(full_path(job, task).read_text(encoding="utf-8"))
    return float(d["clean" if task == "math" else "full"]["accuracy_pct"])


def baseline(seed: int, task: str) -> float:
    path = BASE_OUTPUTS / task / f"llama32_3b_hidden_hr4_seed{seed}_full.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return float(d["clean" if task == "math" else "full"]["accuracy_pct"])


def write_results() -> None:
    lines = [
        "# Llama-3.2 tuned A-LoRA full downstream confirmation", "",
        f"Updated: {now()}", "", "| seed | hidden MATH | tuned MATH | Δ MATH | hidden GSM8K | tuned GSM8K | Δ GSM8K |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    deltas = {"math": [], "gsm8k": []}
    for job in JOBS:
        values = []
        for task in ("math", "gsm8k"):
            b = baseline(job.seed, task); t = metric(job, task)
            values += [b, t, None if t is None else t-b]
            if t is not None: deltas[task].append(t-b)
        fmt = lambda x: "-" if x is None else f"{x:.4f}%"
        lines.append(f"| {job.seed} | " + " | ".join(fmt(x) for x in values) + " |")
    lines += ["", "## Paired aggregates", ""]
    for task, xs in deltas.items():
        if len(xs) == 3:
            mean = statistics.mean(xs); sd = statistics.stdev(xs)
            half = 4.3026527299 * sd / math.sqrt(3)
            lines.append(
                f"- {task}: {mean:+.6f} ± {sd:.6f} pp; 95% t-CI "
                f"[{mean-half:+.6f}, {mean+half:+.6f}]; positive {sum(x>0 for x in xs)}/3."
            )
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for d in (
        HERE, CHECKPOINTS, OUTPUTS / "math", OUTPUTS / "gsm8k",
        LOGS / "train", LOGS / "math", LOGS / "gsm8k", HERE / "hf_cache/home",
    ):
        d.mkdir(parents=True, exist_ok=True)
    state = {"started_at": now(), "phase": "mixed_training_evaluation", "jobs": [j.name for j in JOBS]}
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    event("phase", phase=state["phase"])
    work: queue.Queue[Any] = queue.Queue()
    # New training first; existing seed-42 evaluation fills the other GPUs.
    for job in JOBS:
        if not checkpoint_complete(job): work.put(("train", job, None, None))
    for job in JOBS:
        if checkpoint_complete(job): enqueue_eval(work, job)
    failures: list[str] = []
    failure_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done(); return
            kind, job, task, shard = item
            try:
                if kind == "train":
                    ok = train_one(job, gpu)
                    if ok: enqueue_eval(work, job)
                else:
                    ok = eval_shard(job, task, shard, gpu)
                if not ok:
                    with failure_lock: failures.append(f"{kind}:{job.name}:{task}:{shard}")
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in GPUS]
    for thread in threads: thread.start()
    work.join()
    for _ in threads: work.put(None)
    for thread in threads: thread.join()
    if failures:
        state["phase"] = "failed"; state["failures"] = failures
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        event("experiment_failure", failures=failures)
        raise SystemExit(failures)

    for job in JOBS:
        for task in ("math", "gsm8k"):
            if full_complete(job, task): continue
            merge = MATH_MERGE if task == "math" else GSM_MERGE
            expected = "5000" if task == "math" else "1319"
            inputs = [str(shard_path(job, task, i)) for i in range(NUM_SHARDS)]
            rc = subprocess.run(
                [str(PYTHON), str(merge), "--inputs", *inputs, "--output", str(full_path(job, task)), "--expected-samples", expected],
                cwd=ROOT,
            ).returncode
            if rc != 0 or not full_complete(job, task):
                raise SystemExit(f"Merge failed: {job.name} {task}")
    write_results()
    state["phase"] = "complete"; state["finished_at"] = now()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    event("experiment_complete")


if __name__ == "__main__":
    main()
