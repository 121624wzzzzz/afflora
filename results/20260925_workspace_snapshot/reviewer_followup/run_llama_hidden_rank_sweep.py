#!/usr/bin/env python
"""Work-conserving hidden-LoRA rank screening for Llama-3.1/3.2."""

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


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reviewer_followup/llama_hidden_rank_sweep"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
MODELS = {
    "llama31_8b": ROOT.parent / "models/Llama-3.1-8B-Base",
    "llama32_3b": ROOT.parent / "models/Llama-3.2-3B-Base",
}
TRAIN = ROOT / "data/metamathqa_40k/train.jsonl"
DEV = ROOT / "data/metamathqa_40k/eval.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
LOSS_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_metamath_loss.py"
CHECKPOINTS = HERE / "checkpoints"
EXISTING = ROOT / "reviewer_followup/llama_cross_version/checkpoints"
OUTPUTS = HERE / "outputs/metamath_loss"
LOGS = HERE / "logs"
STATE_FILE = HERE / "state.json"
EVENTS_FILE = HERE / "events.jsonl"
RESULTS_FILE = HERE / "RESULTS.md"
RANKS = (1, 2, 4, 8, 16)
GPUS = tuple(range(8))
SEED = 42
STATE_LOCK = threading.Lock()
QUEUE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    model_alias: str
    rank: int

    @property
    def name(self) -> str:
        return f"{self.model_alias}_hidden_hr{self.rank}_seed{SEED}"

    @property
    def model_path(self) -> Path:
        return MODELS[self.model_alias]

    @property
    def checkpoint_dir(self) -> Path:
        if self.rank == 4:
            return EXISTING / self.name
        return CHECKPOINTS / self.name


JOBS = [Job(alias, rank) for alias in MODELS for rank in RANKS]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def initial_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        for job in JOBS:
            state.setdefault("jobs", {}).setdefault(
                job.name, {"status": "pending", "attempts": 0}
            )
        return state
    return {
        "schema_version": 1,
        "started_at": now(),
        "updated_at": now(),
        "phase": "screening",
        "jobs": {job.name: {"status": "pending", "attempts": 0} for job in JOBS},
    }


STATE = initial_state()


def event(kind: str, **fields: Any) -> None:
    row = {"time": now(), "kind": kind, **fields}
    with STATE_LOCK:
        with EVENTS_FILE.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_job(name: str, **fields: Any) -> None:
    with STATE_LOCK:
        STATE["jobs"][name].update(fields)
        STATE["updated_at"] = now()
        atomic_json(STATE_FILE, STATE)
        write_summary_locked()


def checkpoint_complete(job: Job) -> bool:
    required = (
        job.checkpoint_dir / "adapter_model.safetensors",
        job.checkpoint_dir / "adapter_config.json",
        job.checkpoint_dir / "run_args.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def eval_path(job: Job) -> Path:
    return OUTPUTS / f"{job.name}.json"


def eval_complete(job: Job) -> bool:
    path = eval_path(job)
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return (
            report["num_samples"] == 499
            and report["answer_tokens"] > 0
            and math.isfinite(float(report["loss"]))
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DS_IGNORE_CUDA_DETECTION"] = "1"
    env["HF_HOME"] = str(HERE / "hf_cache/home")
    env["HF_DATASETS_CACHE"] = str(HERE / "hf_cache/datasets")
    env["XDG_CACHE_HOME"] = str(HERE / "hf_cache/xdg")
    return env


def train_command(job: Job) -> list[str]:
    if job.model_alias == "llama31_8b":
        batch, accum = 4, 4
    else:
        batch, accum = 8, 2
    return [
        str(PYTHON), str(TRAIN_SCRIPT),
        "--model-path", str(job.model_path),
        "--train-data", str(TRAIN),
        "--output-dir", str(job.checkpoint_dir),
        "--variant", "hidden_lora",
        "--hidden-lora-rank", str(job.rank),
        "--hidden-lora-alpha", str(2 * job.rank),
        "--hidden-lora-dropout", "0.05",
        "--max-seq-len", "1024",
        "--per-device-train-batch-size", str(batch),
        "--gradient-accumulation-steps", str(accum),
        "--learning-rate", "2e-4",
        "--num-train-epochs", "1",
        "--logging-steps", "10",
        "--bf16", "--gradient-checkpointing",
        "--seed", str(SEED),
        "--master-dtype", "fp32", "--base-dtype", "bf16",
        "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]


def train_one(job: Job, gpu: int) -> bool:
    if checkpoint_complete(job):
        update_job(job.name, status="trained", gpu=None, reused=(job.rank == 4))
        event("training_skip_complete", job=job.name, reused=(job.rank == 4))
        return True
    for attempt in (1, 2):
        log = LOGS / "train" / f"{job.name}_attempt{attempt}.log"
        update_job(
            job.name, status="training", gpu=gpu, attempts=attempt,
            started_at=now(),
        )
        event("training_start", job=job.name, gpu=gpu, attempt=attempt)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                train_command(job), cwd=ROOT, env=run_env(gpu),
                stdout=stream, stderr=subprocess.STDOUT,
            )
        if result.returncode == 0 and checkpoint_complete(job):
            update_job(
                job.name, status="trained", gpu=None, returncode=0,
                finished_at=now(),
            )
            event("training_finish", job=job.name, gpu=gpu, attempt=attempt)
            return True
        update_job(
            job.name, status="retrying" if attempt == 1 else "failed",
            gpu=None, returncode=result.returncode,
        )
        event(
            "training_failure", job=job.name, gpu=gpu,
            attempt=attempt, returncode=result.returncode,
        )
    return False


def eval_one(job: Job, gpu: int) -> bool:
    if eval_complete(job):
        update_job(job.name, status="complete", gpu=None)
        event("eval_skip_complete", job=job.name)
        return True
    batches = (8, 4, 2) if job.model_alias == "llama31_8b" else (16, 8, 4)
    for attempt, batch in enumerate(batches, 1):
        log = LOGS / "eval" / f"{job.name}_attempt{attempt}.log"
        update_job(
            job.name, status="evaluating_dev", gpu=gpu,
            eval_attempt=attempt, eval_batch_size=batch,
        )
        event(
            "eval_start", job=job.name, gpu=gpu,
            attempt=attempt, batch_size=batch,
        )
        cmd = [
            str(PYTHON), str(LOSS_EVAL),
            "--model-path", str(job.model_path),
            "--run-dir", str(job.checkpoint_dir),
            "--eval-data", str(DEV),
            "--output-file", str(eval_path(job)),
            "--batch-size", str(batch),
            "--max-seq-len", "1024", "--dtype", "bf16",
        ]
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                cmd, cwd=ROOT, env=run_env(gpu),
                stdout=stream, stderr=subprocess.STDOUT,
            )
        if result.returncode == 0 and eval_complete(job):
            update_job(job.name, status="complete", gpu=None, finished_at=now())
            event(
                "eval_finish", job=job.name, gpu=gpu,
                attempt=attempt, batch_size=batch,
            )
            return True
        event(
            "eval_failure", job=job.name, gpu=gpu, attempt=attempt,
            batch_size=batch, returncode=result.returncode,
        )
    update_job(job.name, status="eval_failed", gpu=None)
    return False


def result_for(job: Job) -> tuple[float | None, float | None]:
    if not eval_complete(job):
        return None, None
    report = json.loads(eval_path(job).read_text(encoding="utf-8"))
    return float(report["loss"]), float(report["perplexity"])


def write_summary_locked() -> None:
    counts: dict[str, int] = {}
    for item in STATE["jobs"].values():
        status = item.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# Llama hidden-LoRA rank screening", "",
        f"Updated: {now()}", "", f"Phase: `{STATE['phase']}`", "",
        "## Runtime status", "", "| status | jobs |", "| --- | ---: |",
    ]
    lines += [f"| {key} | {value} |" for key, value in sorted(counts.items())]
    lines += [
        "", "## Held-out MetaMathQA answer-token loss", "",
        "| model | hidden rank | seed | dev CE | dev PPL | checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for job in JOBS:
        loss, ppl = result_for(job)
        reused = "reused" if job.rank == 4 else "new"
        lines.append(
            f"| {job.model_alias} | {job.rank} | {SEED} | "
            f"{'-' if loss is None else f'{loss:.8f}'} | "
            f"{'-' if ppl is None else f'{ppl:.8f}'} | {reused} |"
        )
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (
        HERE, CHECKPOINTS, OUTPUTS, LOGS / "train", LOGS / "eval",
        HERE / "hf_cache/home", HERE / "hf_cache/datasets", HERE / "hf_cache/xdg",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE_FILE, STATE)
    event("phase", phase="mixed_training_and_dev_evaluation")

    work: queue.Queue[tuple[str, Job] | None] = queue.Queue()
    eval_enqueued: set[str] = set()

    def enqueue_eval(job: Job) -> None:
        with QUEUE_LOCK:
            if job.name in eval_enqueued:
                return
            eval_enqueued.add(job.name)
            work.put(("eval", job))

    # Put all new training jobs first so all eight GPUs begin with training.
    for job in JOBS:
        if not checkpoint_complete(job):
            work.put(("train", job))
        elif eval_complete(job):
            update_job(job.name, status="complete", gpu=None, reused=(job.rank == 4))
        else:
            update_job(job.name, status="trained", gpu=None, reused=(job.rank == 4))
    # Existing checkpoints enter behind training jobs and fill GPUs as soon as
    # shorter training jobs finish.
    for job in JOBS:
        if checkpoint_complete(job) and not eval_complete(job):
            enqueue_eval(job)

    failures: list[str] = []
    failure_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            kind, job = item
            try:
                if kind == "train":
                    if train_one(job, gpu):
                        enqueue_eval(job)
                    else:
                        with failure_lock:
                            failures.append(f"train:{job.name}")
                elif not eval_one(job, gpu):
                    with failure_lock:
                        failures.append(f"eval:{job.name}")
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in GPUS]
    for thread in threads:
        thread.start()
    work.join()
    for _ in threads:
        work.put(None)
    for thread in threads:
        thread.join()

    with STATE_LOCK:
        STATE["phase"] = "failed" if failures else "complete"
        STATE["updated_at"] = now()
        atomic_json(STATE_FILE, STATE)
        write_summary_locked()
    if failures:
        event("experiment_failure", failures=failures)
        raise SystemExit(f"Failures: {failures}")
    event("experiment_complete")


if __name__ == "__main__":
    main()
