#!/usr/bin/env python
"""High/low affine-friendliness contrast on two tied Gemma base models."""

from __future__ import annotations

import importlib.util
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reviewer_followup" / "gemma_structure_contrast"
SOURCE = (
    ROOT
    / "corrected_math_evaluation/model_families/qwen25/small_models"
    / "hidden_mergeable_rank_sweep/run_experiment.py"
)
MODELS = {
    "gemma2_2b": ROOT.parent / "models/Gemma-2-2B-Base",
    "gemma3_1b": ROOT.parent / "models/Gemma-3-1B-Base",
}
REQUIRED_WEIGHTS = {
    "gemma2_2b": (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ),
    "gemma3_1b": ("model.safetensors",),
}


def load_runner():
    spec = importlib.util.spec_from_file_location("gemma_structure_base", SOURCE)
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
    sessions = (
        "alora_review_qwen25_math_confirm",
        "alora_review_equal_budget",
        "alora_dl_gemma2_2b",
        "alora_dl_gemma3_1b",
    )
    while any(tmux_exists(name) for name in sessions):
        time.sleep(30)
    missing = []
    for alias, filenames in REQUIRED_WEIGHTS.items():
        for filename in filenames:
            path = MODELS[alias] / filename
            if not path.is_file() or path.stat().st_size < 100_000_000:
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"Model download incomplete: {missing}")


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
            runner.JOBS.append(
                runner.Job(
                    f"{alias}_mergeable_ar16_s1_hr4_seed{seed}",
                    alias,
                    4,
                    seed,
                    16,
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

    runner.GPUS = tuple(range(8))
    runner.set_phase("training_structure_contrast")
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

    runner.set_phase("evaluation_structure_contrast")
    for job in runner.JOBS:
        if not runner.run_eval_task(job, "math") or not runner.run_eval_task(job, "gsm8k"):
            runner.set_phase("evaluation_failed")
            raise SystemExit(f"Evaluation failed: {job.name}")
        runner.update_job(job.name, status="complete", gpu=None, finished_at=runner.now())
    runner.set_phase("complete")
    runner.event("experiment_complete")


if __name__ == "__main__":
    main()
