#!/usr/bin/env python
"""Work-conserving Llama-3.1/3.2 reviewer follow-up experiment."""

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
HERE = ROOT / "reviewer_followup" / "llama_cross_version"
SOURCE = (
    ROOT
    / "corrected_math_evaluation/model_families/qwen25/small_models"
    / "hidden_mergeable_rank_sweep/run_experiment.py"
)
MODELS = {
    "llama31_8b": ROOT.parent / "models/Llama-3.1-8B-Base",
    "llama32_3b": ROOT.parent / "models/Llama-3.2-3B-Base",
}
MIN_WEIGHT_BYTES = {"llama31_8b": 14_000_000_000, "llama32_3b": 5_000_000_000}


def load_runner():
    spec = importlib.util.spec_from_file_location("llama_cross_version_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import experiment runner from {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def wait_for_prerequisites() -> None:
    # Model downloads are the only hard prerequisite. Do not wait for the
    # preceding experiment's CPU-only merge/report phase: once its GPU shard
    # workers have drained, Llama can safely begin loading on the GPUs.
    sessions = (
        "alora_dl_llama31_8b",
        "alora_dl_llama32_3b",
    )
    while any(tmux_exists(name) for name in sessions):
        time.sleep(5)
    incomplete = []
    for alias, model_dir in MODELS.items():
        total = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
        if total < MIN_WEIGHT_BYTES[alias]:
            incomplete.append(f"{alias}: {total} bytes")
    if incomplete:
        raise RuntimeError(f"Model download incomplete: {incomplete}")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    wait_for_prerequisites()
    runner = load_runner()
    runner.MODELS = MODELS
    runner.CHECKPOINTS = HERE / "checkpoints"
    runner.OUTPUTS = HERE / "outputs"
    runner.LOGS = HERE / "logs"
    runner.STATE_FILE = HERE / "state.json"
    runner.EVENTS_FILE = HERE / "events.jsonl"
    runner.RESULTS_FILE = HERE / "RESULTS.md"
    runner.JOBS = []
    for alias in MODELS:
        for seed in (42, 43, 44):
            runner.JOBS.append(
                runner.Job(f"{alias}_hidden_hr4_seed{seed}", alias, 4, seed, 0, None)
            )
            placement = "lmhead" if alias == "llama31_8b" else "mergeable"
            runner.JOBS.append(
                runner.Job(
                    f"{alias}_{placement}_ar16_s1_hr4_seed{seed}",
                    alias,
                    4,
                    seed,
                    16,
                    1.0,
                )
            )

    def train_command(job):  # noqa: ANN001, ANN202
        if job.model_alias == "llama31_8b":
            batch, accum = 4, 4
        else:
            batch, accum = 8, 2
        variant = "hidden_lora"
        if job.affine_scale is not None:
            variant = (
                "affine_lm_head_plus_hidden_lora"
                if job.model_alias == "llama31_8b"
                else "affine_input_lm_head_plus_hidden_lora"
            )
        cmd = [
            str(runner.PYTHON), str(runner.TRAIN_SCRIPT),
            "--model-path", str(job.model_path),
            "--train-data", str(runner.TRAIN),
            "--output-dir", str(runner.CHECKPOINTS / job.name),
            "--variant", variant,
            "--hidden-lora-rank", "4", "--hidden-lora-alpha", "8",
            "--hidden-lora-dropout", "0.05",
            "--max-seq-len", "1024",
            "--per-device-train-batch-size", str(batch),
            "--gradient-accumulation-steps", str(accum),
            "--learning-rate", "2e-4", "--num-train-epochs", "1",
            "--logging-steps", "10", "--bf16", "--gradient-checkpointing",
            "--seed", str(job.seed), "--master-dtype", "fp32",
            "--base-dtype", "bf16", "--save-strategy", "epoch",
            "--save-total-limit", "1", "--lr-scheduler-type", "cosine",
            "--warmup-ratio", "0.03",
        ]
        if job.affine_scale is not None:
            cmd += ["--affine-rank", "16", "--affine-alpha", "16"]
            if job.model_alias == "llama32_3b":
                cmd += ["--tie-affine-input-lm-head-adapters", "--affine-lm-head-bias"]
        return cmd

    runner.train_command = train_command
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

    runner.GPUS = tuple(range(8))
    runner.set_phase("training_llama_cross_version")
    work = queue.Queue()
    for job in runner.JOBS:
        if runner.checkpoint_complete(job):
            runner.update_job(job.name, status="trained", gpu=None)
        else:
            work.put(job)
    threads = [
        threading.Thread(target=runner.training_worker, args=(gpu, work), daemon=False)
        for gpu in runner.GPUS
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

    # Dynamic evaluation queue. Sixteen shards per dataset provide enough
    # granularity for a GPU that finishes early to immediately pull more work.
    # This replaces the legacy fixed one-shard-per-GPU barrier.
    num_shards = 16
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
                data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
                output = runner.OUTPUTS / task / f"{job.name}_shard{shard}.json"
                log_dir = runner.LOGS / task
                log_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["DS_IGNORE_CUDA_DETECTION"] = "1"
                succeeded = False
                batches = (64, 32, 16) if job.model_alias == "llama31_8b" else (128, 64, 32)
                for attempt, batch in enumerate(batches, 1):
                    cmd = [
                        str(runner.PYTHON), str(script), "--model-path", str(job.model_path),
                        "--run-dir", str(runner.CHECKPOINTS / job.name),
                        "--eval-data", str(data), "--output-file", str(output),
                        "--batch-size", str(batch), "--max-new-tokens", "512",
                        "--num-shards", str(num_shards), "--shard-index", str(shard),
                    ]
                    log = log_dir / f"{job.name}_shard{shard}_attempt{attempt}.log"
                    with log.open("w", encoding="utf-8") as stream:
                        result = subprocess.run(
                            cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
                        )
                    if result.returncode == 0 and output.is_file() and output.stat().st_size > 0:
                        succeeded = True
                        break
                if not succeeded:
                    with failure_lock:
                        eval_failures.append((job.name, task, shard))
            finally:
                eval_work.task_done()

    runner.set_phase("evaluation_llama_cross_version_dynamic")
    eval_threads = [
        threading.Thread(target=eval_worker, args=(gpu,), daemon=False)
        for gpu in runner.GPUS
    ]
    for thread in eval_threads:
        thread.start()
    for thread in eval_threads:
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
                    str(runner.PYTHON), str(merge), "--inputs", *inputs,
                    "--output", str(runner.OUTPUTS / task / f"{job.name}_full.json"),
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
