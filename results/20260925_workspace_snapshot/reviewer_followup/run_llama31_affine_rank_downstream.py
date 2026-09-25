#!/usr/bin/env python
"""Full MATH/GSM8K diagnostic for all Llama-3.1 A-LoRA rank checkpoints."""

from __future__ import annotations

import json
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
HERE = ROOT / "reviewer_followup/llama31_affine_rank_downstream"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
MODEL = ROOT.parent / "models/Llama-3.1-8B-Base"
RANK_SWEEP = ROOT / "reviewer_followup/llama_affine_rank_sweep"
CROSS = ROOT / "reviewer_followup/llama_cross_version"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
STATE_FILE = HERE / "state.json"
EVENTS_FILE = HERE / "events.jsonl"
RESULTS_FILE = HERE / "RESULTS.md"
RANKS = (4, 8, 16, 32)
NUM_SHARDS = 16
GPUS = tuple(range(8))
SEED = 42
LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    rank: int

    @property
    def name(self) -> str:
        return f"llama31_8b_lmhead_ar{self.rank}_s1_hr4_seed{SEED}"

    @property
    def checkpoint_dir(self) -> Path:
        root = CROSS / "checkpoints" if self.rank == 16 else RANK_SWEEP / "checkpoints"
        return root / self.name


JOBS = [Job(rank) for rank in RANKS]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def event(kind: str, **fields: Any) -> None:
    with LOCK:
        with EVENTS_FILE.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "kind": kind, **fields}) + "\n")


def run_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DS_IGNORE_CUDA_DETECTION"] = "1"
    return env


def full_path(job: Job, task: str) -> Path:
    if job.rank == 16:
        return CROSS / "outputs" / task / f"{job.name}_full.json"
    return OUTPUTS / task / f"{job.name}_full.json"


def full_complete(job: Job, task: str) -> bool:
    expected = 5000 if task == "math" else 1319
    path = full_path(job, task)
    if not path.is_file(): return False
    try:
        d = json.loads(path.read_text()); rows = d["results"]
        return d["full"]["num_samples"] == expected and len(rows) == expected
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def shard_path(job: Job, task: str, shard: int) -> Path:
    return OUTPUTS / task / f"{job.name}_shard{shard}.json"


def eval_shard(job: Job, task: str, shard: int, gpu: int) -> bool:
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    for attempt, batch in enumerate((64, 32, 16), 1):
        event("eval_start", job=job.name, task=task, shard=shard, gpu=gpu, attempt=attempt, batch_size=batch)
        cmd = [
            str(PYTHON), str(script), "--model-path", str(MODEL),
            "--run-dir", str(job.checkpoint_dir), "--eval-data", str(data),
            "--output-file", str(shard_path(job, task, shard)),
            "--batch-size", str(batch), "--max-new-tokens", "512",
            "--num-shards", str(NUM_SHARDS), "--shard-index", str(shard),
        ]
        log = LOGS / task / f"{job.name}_shard{shard}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(cmd, cwd=ROOT, env=run_env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
        out = shard_path(job, task, shard)
        if rc == 0 and out.is_file() and out.stat().st_size > 0:
            event("eval_finish", job=job.name, task=task, shard=shard, gpu=gpu, attempt=attempt, batch_size=batch)
            return True
        event("eval_failure", job=job.name, task=task, shard=shard, gpu=gpu, attempt=attempt, returncode=rc)
    return False


def metric(job: Job, task: str) -> float:
    d = json.loads(full_path(job, task).read_text())
    return float(d["clean" if task == "math" else "full"]["accuracy_pct"])


def baseline(task: str) -> float:
    path = CROSS / "outputs" / task / f"llama31_8b_hidden_hr4_seed{SEED}_full.json"
    d = json.loads(path.read_text())
    return float(d["clean" if task == "math" else "full"]["accuracy_pct"])


def ce(job: Job) -> float:
    path = RANK_SWEEP / "outputs/metamath_loss" / f"{job.name}.json"
    return float(json.loads(path.read_text())["loss"])


def write_results() -> None:
    bm, bg = baseline("math"), baseline("gsm8k")
    lines = [
        "# Llama-3.1 A-LoRA rank downstream diagnostic", "", f"Updated: {now()}", "",
        f"Hidden-only baseline: MATH {bm:.4f}%, GSM8K {bg:.4f}%.", "",
        "| A rank | dev CE | MATH | Δ MATH | GSM8K | Δ GSM8K |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for job in JOBS:
        m, g = metric(job, "math"), metric(job, "gsm8k")
        lines.append(f"| {job.rank} | {ce(job):.8f} | {m:.4f}% | {m-bm:+.4f} pp | {g:.4f}% | {g-bg:+.4f} pp |")
    RESULTS_FILE.write_text("\n".join(lines) + "\n")


def main() -> None:
    for d in (HERE, OUTPUTS / "math", OUTPUTS / "gsm8k", LOGS / "math", LOGS / "gsm8k"):
        d.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        needed = ("adapter_model.safetensors", "affine_vocab_adapter.safetensors", "run_args.json")
        if not all((job.checkpoint_dir / f).is_file() for f in needed):
            raise SystemExit(f"Incomplete checkpoint: {job.checkpoint_dir}")
    state = {"started_at": now(), "phase": "dynamic_evaluation", "ranks": list(RANKS)}
    STATE_FILE.write_text(json.dumps(state, indent=2)); event("phase", phase=state["phase"])
    work: queue.Queue[Any] = queue.Queue()
    for job in JOBS:
        for task in ("math", "gsm8k"):
            if not full_complete(job, task):
                for shard in range(NUM_SHARDS): work.put((job, task, shard))
    failures: list[str] = []; failure_lock = threading.Lock()
    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None: work.task_done(); return
            job, task, shard = item
            try:
                if not eval_shard(job, task, shard, gpu):
                    with failure_lock: failures.append(f"{job.name}:{task}:{shard}")
            finally: work.task_done()
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in GPUS]
    for t in threads: t.start()
    work.join()
    for _ in threads: work.put(None)
    for t in threads: t.join()
    if failures:
        state["phase"] = "failed"; state["failures"] = failures
        STATE_FILE.write_text(json.dumps(state, indent=2)); raise SystemExit(failures)
    for job in JOBS:
        for task in ("math", "gsm8k"):
            if full_complete(job, task): continue
            merge = MATH_MERGE if task == "math" else GSM_MERGE
            expected = "5000" if task == "math" else "1319"
            inputs = [str(shard_path(job, task, i)) for i in range(NUM_SHARDS)]
            rc = subprocess.run([
                str(PYTHON), str(merge), "--inputs", *inputs,
                "--output", str(full_path(job, task)), "--expected-samples", expected,
            ], cwd=ROOT).returncode
            if rc != 0 or not full_complete(job, task): raise SystemExit(f"merge failed: {job.name} {task}")
    write_results(); state["phase"] = "complete"; state["finished_at"] = now()
    STATE_FILE.write_text(json.dumps(state, indent=2)); event("experiment_complete")


if __name__ == "__main__":
    main()
