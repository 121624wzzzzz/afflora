#!/usr/bin/env python
"""Confirm selected Qwen2.5-1.5B Math configurations on untouched seeds."""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reviewer_followup" / "qwen25_math_confirmation"
SOURCE = (
    ROOT
    / "corrected_math_evaluation/model_families/qwen25/small_models"
    / "hidden_mergeable_rank_sweep/run_experiment.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("qwen25_confirmation_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import experiment runner from {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    runner.CHECKPOINTS = HERE / "checkpoints"
    runner.OUTPUTS = HERE / "outputs"
    runner.LOGS = HERE / "logs"
    runner.STATE_FILE = HERE / "state.json"
    runner.EVENTS_FILE = HERE / "events.jsonl"
    runner.RESULTS_FILE = HERE / "RESULTS.md"
    runner.JOBS = []
    for seed in (45, 46, 47):
        runner.JOBS.append(
            runner.Job(f"qwen25_15b_hidden_hr4_seed{seed}", "qwen25_15b", 4, seed, 0, None)
        )
        for rank in (8, 16):
            runner.JOBS.append(
                runner.Job(
                    f"qwen25_15b_mergeable_ar{rank}_s1_hr4_seed{seed}",
                    "qwen25_15b",
                    4,
                    seed,
                    rank,
                    1.0,
                )
            )

    for directory in (
        runner.CHECKPOINTS,
        runner.OUTPUTS / "math",
        runner.OUTPUTS / "gsm8k",
        runner.LOGS / "train",
        runner.LOGS / "math",
        runner.LOGS / "gsm8k",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if runner.STATE_FILE.exists():
        runner.STATE = json.loads(runner.STATE_FILE.read_text(encoding="utf-8"))
        for job in runner.JOBS:
            runner.STATE.setdefault("jobs", {}).setdefault(
                job.name, {"status": "pending", "attempts": 0}
            )
    else:
        runner.STATE = runner.initial_state()
    runner.atomic_json(runner.STATE_FILE, runner.STATE)

    runner.set_phase("training_confirmation")
    work = queue.Queue()
    for job in runner.JOBS:
        if runner.checkpoint_complete(job):
            runner.update_job(job.name, status="trained", gpu=None)
        else:
            work.put(job)
    # Work-conserving single-GPU training pool: every available GPU pulls the
    # next pending job immediately.
    threads = [
        threading.Thread(target=runner.training_worker, args=(gpu, work), daemon=False)
        for gpu in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [job.name for job in runner.JOBS if not runner.checkpoint_complete(job)]
    if failed:
        runner.set_phase("training_failed")
        runner.event("experiment_failure", failed_jobs=failed)
        raise SystemExit(f"Training failed: {failed}")

    runner.GPUS = tuple(range(8))
    # Put every (checkpoint, dataset, shard) in one queue. A GPU that finishes
    # a short/low-length shard immediately takes the next pending shard instead
    # of waiting at a same-checkpoint barrier. Eight shards preserve the
    # existing evaluator layout while providing 9 x 2 x 8 independent tasks.
    num_shards = 8
    eval_work = queue.Queue()
    for job in runner.JOBS:
        for task in ("math", "gsm8k"):
            if not runner.eval_complete(job, task):
                for shard in range(num_shards):
                    eval_work.put((job, task, shard))

    eval_failures = []
    failure_lock = threading.Lock()

    def eval_worker(gpu: int) -> None:
        while True:
            try:
                job, task, shard = eval_work.get_nowait()
            except queue.Empty:
                return
            try:
                script = runner.MATH_EVAL if task == "math" else runner.GSM_EVAL
                data = ROOT / (
                    "data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl"
                )
                output = runner.OUTPUTS / task / f"{job.name}_shard{shard}.json"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["DS_IGNORE_CUDA_DETECTION"] = "1"
                succeeded = False
                for attempt, batch in enumerate((256, 128, 64), 1):
                    cmd = [
                        str(runner.PYTHON), str(script),
                        "--model-path", str(job.model_path),
                        "--run-dir", str(runner.CHECKPOINTS / job.name),
                        "--eval-data", str(data),
                        "--output-file", str(output),
                        "--batch-size", str(batch),
                        "--max-new-tokens", "512",
                        "--num-shards", str(num_shards),
                        "--shard-index", str(shard),
                    ]
                    log = (
                        runner.LOGS / task /
                        f"{job.name}_shard{shard}_dynamic_attempt{attempt}.log"
                    )
                    runner.event(
                        "eval_shard_start", job=job.name, task=task,
                        shard=shard, gpu=gpu, attempt=attempt, batch_size=batch,
                    )
                    with log.open("w", encoding="utf-8") as stream:
                        result = subprocess.run(
                            cmd, cwd=ROOT, env=env,
                            stdout=stream, stderr=subprocess.STDOUT,
                        )
                    if result.returncode == 0 and output.is_file() and output.stat().st_size > 0:
                        succeeded = True
                        runner.event(
                            "eval_shard_finish", job=job.name, task=task,
                            shard=shard, gpu=gpu, attempt=attempt, batch_size=batch,
                        )
                        break
                    runner.event(
                        "eval_shard_failure", job=job.name, task=task,
                        shard=shard, gpu=gpu, attempt=attempt,
                        batch_size=batch, returncode=result.returncode,
                    )
                if not succeeded:
                    with failure_lock:
                        eval_failures.append((job.name, task, shard))
            finally:
                eval_work.task_done()

    runner.set_phase("evaluation_confirmation_dynamic")
    threads = [
        threading.Thread(target=eval_worker, args=(gpu,), daemon=False)
        for gpu in runner.GPUS
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if eval_failures:
        runner.set_phase("evaluation_failed")
        raise SystemExit(f"Evaluation shard failures: {eval_failures}")

    for job in runner.JOBS:
        for task in ("math", "gsm8k"):
            if runner.eval_complete(job, task):
                continue
            merge = runner.MATH_MERGE if task == "math" else runner.GSM_MERGE
            expected = "5000" if task == "math" else "1319"
            inputs = [
                str(runner.OUTPUTS / task / f"{job.name}_shard{shard}.json")
                for shard in range(num_shards)
            ]
            merged = subprocess.run(
                [
                    str(runner.PYTHON), str(merge),
                    "--inputs", *inputs,
                    "--output", str(
                        runner.OUTPUTS / task / f"{job.name}_full.json"
                    ),
                    "--expected-samples", expected,
                ],
                cwd=ROOT,
            )
            if merged.returncode != 0 or not runner.eval_complete(job, task):
                runner.set_phase("evaluation_failed")
                raise SystemExit(f"Evaluation merge failed: {job.name} {task}")
        runner.update_job(job.name, status="complete", gpu=None, finished_at=runner.now())
    runner.set_phase("complete")
    runner.event("experiment_complete")


if __name__ == "__main__":
    main()
