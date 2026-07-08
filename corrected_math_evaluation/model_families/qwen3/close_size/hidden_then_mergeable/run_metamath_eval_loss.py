#!/usr/bin/env python
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data").is_dir() and (p / "scripts").is_dir())
PYTHON = Path("/home/wz/anaconda3/envs/torch24/bin/python")
EVAL_SCRIPT = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_metamath_loss.py"
EVAL_DATA = ROOT / "data/metamathqa_40k/eval.jsonl"
CHECKPOINTS = HERE / "checkpoints"
OUTPUTS = HERE / "outputs/metamath_eval_loss"
LOGS = HERE / "logs/metamath_eval_loss"
STATE_FILE = HERE / "state_metamath_eval_loss.json"
EVENTS_FILE = HERE / "events_metamath_eval_loss.jsonl"
GPUS = tuple(range(8))

BASE_MODELS = {
    "qwen3_06b_base": Path("/home/wz/projects/mypro/im_exp/models/Qwen3-0.6B-Base"),
    "qwen3_17b_base": Path("/home/wz/projects/mypro/im_exp/models/Qwen3-1.7B-Base"),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def event(kind: str, **fields: Any) -> None:
    with EVENTS_FILE.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), "kind": kind, **fields}, ensure_ascii=False) + "\n")


def atomic_state(state: dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def build_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = [
        (name, ["--model-path", str(path)]) for name, path in BASE_MODELS.items()
    ]
    for seed in (42, 43, 44):
        names = [
            f"qwen3_06b_hidden_hr8_seed{seed}",
            f"qwen3_06b_mergeable_ar1_s8_hr8_seed{seed}",
            f"qwen3_06b_mergeable_ar2_s8_hr8_seed{seed}",
            f"qwen3_17b_hidden_hr4_seed{seed}",
            f"qwen3_17b_mergeable_ar2_s1_hr4_seed{seed}",
            f"qwen3_17b_mergeable_ar8_s1_hr4_seed{seed}",
        ]
        jobs.extend((name, ["--run-dir", str(CHECKPOINTS / name)]) for name in names)
    return jobs


JOBS = build_jobs()
STATE_LOCK = threading.Lock()
STATE = {
    "schema_version": 1,
    "started_at": now(),
    "updated_at": now(),
    "phase": "running",
    "jobs": {name: {"status": "pending"} for name, _ in JOBS},
}


def update_job(name: str, **fields: Any) -> None:
    with STATE_LOCK:
        STATE["jobs"][name].update(fields)
        STATE["updated_at"] = now()
        atomic_state(STATE)


def run_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DS_IGNORE_CUDA_DETECTION"] = "1"
    env["HF_HOME"] = str(HERE / "hf_cache/home")
    env["HF_DATASETS_CACHE"] = str(HERE / "hf_cache/datasets")
    env["XDG_CACHE_HOME"] = str(HERE / "hf_cache/xdg")
    return env


def output_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return report.get("num_samples") == 499 and report.get("answer_tokens", 0) > 0 and "loss" in report


def worker(gpu: int, work: queue.Queue[tuple[str, list[str]]]) -> None:
    while True:
        try:
            name, model_args = work.get_nowait()
        except queue.Empty:
            return
        try:
            out = OUTPUTS / f"{name}.json"
            if output_ok(out):
                update_job(name, status="complete", gpu=None, skipped=True)
                event("skip_complete", job=name)
                continue
            log = LOGS / f"{name}.log"
            cmd = [
                str(PYTHON),
                str(EVAL_SCRIPT),
                *model_args,
                "--eval-data",
                str(EVAL_DATA),
                "--output-file",
                str(out),
                "--batch-size",
                "8",
                "--max-seq-len",
                "1024",
                "--dtype",
                "bf16",
            ]
            update_job(name, status="running", gpu=gpu, started_at=now())
            event("start", job=name, gpu=gpu)
            with log.open("w", encoding="utf-8") as stream:
                rc = subprocess.run(cmd, cwd=ROOT, env=run_env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
            if rc == 0 and output_ok(out):
                report = json.loads(out.read_text(encoding="utf-8"))
                update_job(name, status="complete", gpu=None, finished_at=now(), loss=report["loss"])
                event("finish", job=name, gpu=gpu, loss=report["loss"])
            else:
                update_job(name, status="failed", gpu=None, finished_at=now(), returncode=rc)
                event("failure", job=name, gpu=gpu, returncode=rc)
        finally:
            work.task_done()


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    atomic_state(STATE)
    work: queue.Queue[tuple[str, list[str]]] = queue.Queue()
    for job in JOBS:
        work.put(job)
    threads = [threading.Thread(target=worker, args=(gpu, work), daemon=False) for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with STATE_LOCK:
        failed = [name for name, state in STATE["jobs"].items() if state.get("status") != "complete"]
        STATE["phase"] = "failed" if failed else "complete"
        STATE["failed_jobs"] = failed
        STATE["updated_at"] = now()
        atomic_state(STATE)
    event("complete", failed_jobs=failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
