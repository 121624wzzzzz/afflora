#!/usr/bin/env python
"""Full MATH/GSM8K baseline for the frozen Llama-3.1-8B base model."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
MODEL = ROOT.parent / "models/Llama-3.1-8B-Base"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"
OUTPUTS = HERE / "frozen_baseline_outputs"
LOGS = HERE / "frozen_baseline_logs"
STATE = HERE / "frozen_baseline_state.json"
RESULTS = HERE / "LLAMA31_FROZEN_BASELINE_RESULTS.md"
NUM_SHARDS = 16
LOCK = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def env(gpu: int) -> dict[str, str]:
    value = os.environ.copy()
    value["CUDA_VISIBLE_DEVICES"] = str(gpu)
    value["DS_IGNORE_CUDA_DETECTION"] = "1"
    return value


def shard(task: str, index: int) -> Path:
    return OUTPUTS / task / f"llama31_8b_frozen_base_shard{index}.json"


def full(task: str) -> Path:
    return OUTPUTS / task / "llama31_8b_frozen_base_full.json"


def eval_one(task: str, index: int, gpu: int) -> bool:
    target = shard(task, index)
    if target.is_file():
        return True
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    for attempt, batch in enumerate((64, 32, 16), 1):
        cmd = [
            str(PYTHON), str(script), "--model-path", str(MODEL),
            "--eval-data", str(data), "--output-file", str(target),
            "--batch-size", str(batch), "--max-new-tokens", "512",
            "--num-shards", str(NUM_SHARDS), "--shard-index", str(index),
        ]
        log = LOGS / task / f"shard{index}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(cmd, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc == 0 and target.is_file():
            return True
    return False


def evaluate() -> None:
    work: queue.Queue[tuple[str, int] | None] = queue.Queue()
    for task in ("math", "gsm8k"):
        for index in range(NUM_SHARDS):
            work.put((task, index))
    failures: list[str] = []

    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            task, index = item
            try:
                if not eval_one(task, index, gpu):
                    with LOCK:
                        failures.append(f"{task}:{index}")
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
        raise RuntimeError(f"Evaluation failures: {failures}")


def merge() -> None:
    for task in ("math", "gsm8k"):
        script = MATH_MERGE if task == "math" else GSM_MERGE
        expected = "5000" if task == "math" else "1319"
        inputs = [str(shard(task, index)) for index in range(NUM_SHARDS)]
        rc = subprocess.run(
            [str(PYTHON), str(script), "--inputs", *inputs,
             "--output", str(full(task)), "--expected-samples", expected],
            cwd=ROOT,
        ).returncode
        if rc != 0:
            raise RuntimeError(f"Merge failed: {task}")


def accuracy(path: Path, key: str) -> tuple[int, int, float]:
    metric = json.loads(path.read_text(encoding="utf-8"))[key]
    return int(metric["correct"]), int(metric["num_samples"]), float(metric["accuracy_pct"])


def report() -> None:
    math_correct, math_total, math_acc = accuracy(full("math"), "clean")
    gsm_correct, gsm_total, gsm_acc = accuracy(full("gsm8k"), "full")
    text = f"""# Frozen Llama-3.1-8B Base baseline

Updated: {now()}

| model | MATH clean | GSM8K |
| --- | ---: | ---: |
| frozen Llama-3.1-8B Base | {math_acc:.4f}% ({math_correct}/{math_total}) | {gsm_acc:.4f}% ({gsm_correct}/{gsm_total}) |
"""
    RESULTS.write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    for directory in (OUTPUTS / "math", OUTPUTS / "gsm8k", LOGS / "math", LOGS / "gsm8k"):
        directory.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"phase": "evaluation", "updated_at": now()}, indent=2), encoding="utf-8")
    evaluate()
    STATE.write_text(json.dumps({"phase": "merging", "updated_at": now()}, indent=2), encoding="utf-8")
    merge()
    report()
    STATE.write_text(
        json.dumps({"phase": "complete", "updated_at": now(), "result": str(RESULTS)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
