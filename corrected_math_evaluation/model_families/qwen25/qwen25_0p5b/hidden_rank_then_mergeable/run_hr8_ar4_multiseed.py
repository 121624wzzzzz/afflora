#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data").is_dir() and (p / "scripts").is_dir())
PYTHON = Path("/home/wz/anaconda3/envs/torch24/bin/python")
MODEL = Path("/home/wz/projects/mypro/im_exp/models/Qwen2.5-0.5B-Base")
TRAIN = ROOT / "data/metamathqa_40k/train.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"

CHECKPOINTS = HERE / "checkpoints"
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
STATE_FILE = HERE / "state_hr8_ar4_multiseed.json"
EVENTS_FILE = HERE / "events_hr8_ar4_multiseed.jsonl"

HIDDEN_RANK = 8
AFFINE_RANK = 4
AFFINE_SCALE = 8.0
SEEDS_TO_RUN = (43, 44)
ALL_SEEDS_FOR_ANALYSIS = (42, 43, 44)
GPUS = tuple(range(8))
EVAL_BATCH_CANDIDATES = (512, 256, 128, 64)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def scale_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


@dataclass(frozen=True)
class Job:
    seed: int

    @property
    def name(self) -> str:
        return f"qwen25_05b_mergeable_ar{AFFINE_RANK}_s{scale_tag(AFFINE_SCALE)}_hr{HIDDEN_RANK}_seed{self.seed}"

    @property
    def hidden_alpha(self) -> int:
        return 2 * HIDDEN_RANK

    @property
    def affine_alpha(self) -> float:
        return AFFINE_RANK * AFFINE_SCALE


JOBS = [Job(seed) for seed in SEEDS_TO_RUN]
STATE_LOCK = threading.Lock()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def initial_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        for job in JOBS:
            state.setdefault("jobs", {}).setdefault(job.name, {"status": "pending", "attempts": 0})
        return state
    return {
        "schema_version": 1,
        "started_at": now(),
        "updated_at": now(),
        "phase": "training",
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


def set_phase(phase: str) -> None:
    with STATE_LOCK:
        STATE["phase"] = phase
        STATE["updated_at"] = now()
        atomic_json(STATE_FILE, STATE)
    event("phase", phase=phase)


def run_env(gpu: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DS_IGNORE_CUDA_DETECTION"] = "1"
    env["HF_HOME"] = str(HERE / "hf_cache" / "home")
    env["HF_DATASETS_CACHE"] = str(HERE / "hf_cache" / "datasets")
    env["XDG_CACHE_HOME"] = str(HERE / "hf_cache" / "xdg")
    return env


def checkpoint_complete(job: Job) -> bool:
    directory = CHECKPOINTS / job.name
    required = [
        directory / "adapter_model.safetensors",
        directory / "adapter_config.json",
        directory / "affine_vocab_adapter.safetensors",
        directory / "affine_vocab_config.json",
        directory / "run_args.json",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def eval_complete(job: Job, task: str) -> bool:
    expected = 5000 if task == "math" else 1319
    path = OUTPUTS / task / f"{job.name}_full.json"
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["results"]
        return (
            report["full"]["num_samples"] == expected
            and len(rows) == expected
            and [row["dataset_index"] for row in rows] == list(range(expected))
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def train_command(job: Job) -> list[str]:
    return [
        str(PYTHON), str(TRAIN_SCRIPT),
        "--model-path", str(MODEL),
        "--train-data", str(TRAIN),
        "--output-dir", str(CHECKPOINTS / job.name),
        "--variant", "affine_input_lm_head_plus_hidden_lora",
        "--hidden-lora-rank", str(HIDDEN_RANK),
        "--hidden-lora-alpha", str(job.hidden_alpha),
        "--hidden-lora-dropout", "0.05",
        "--affine-rank", str(AFFINE_RANK),
        "--affine-alpha", f"{job.affine_alpha:g}",
        "--tie-affine-input-lm-head-adapters",
        "--affine-lm-head-bias",
        "--max-seq-len", "1024",
        "--per-device-train-batch-size", "16",
        "--gradient-accumulation-steps", "1",
        "--learning-rate", "2e-4",
        "--num-train-epochs", "1",
        "--logging-steps", "10",
        "--bf16", "--gradient-checkpointing",
        "--seed", str(job.seed),
        "--master-dtype", "fp32", "--base-dtype", "bf16",
        "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]


def train_one(job: Job, gpu: int) -> None:
    if checkpoint_complete(job):
        update_job(job.name, status="trained", gpu=None, finished_at=now())
        event("training_skip_complete", job=job.name)
        return
    for attempt in (1, 2):
        log = LOGS / "train" / f"{job.name}_attempt{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        update_job(job.name, status="training", gpu=gpu, attempts=attempt, started_at=now())
        event("training_start", job=job.name, gpu=gpu, attempt=attempt)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(train_command(job), cwd=ROOT, env=run_env(gpu), stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode == 0 and checkpoint_complete(job):
            update_job(job.name, status="trained", gpu=None, finished_at=now(), returncode=0)
            event("training_finish", job=job.name, gpu=gpu, attempt=attempt)
            return
        update_job(job.name, status="retrying" if attempt == 1 else "failed", gpu=None, returncode=result.returncode)
        event("training_failure", job=job.name, gpu=gpu, attempt=attempt, returncode=result.returncode)


def eval_command(job: Job, task: str, shard: int, batch_size: int) -> list[str]:
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    return [
        str(PYTHON), str(script),
        "--model-path", str(MODEL),
        "--run-dir", str(CHECKPOINTS / job.name),
        "--eval-data", str(data),
        "--output-file", str(OUTPUTS / task / f"{job.name}_shard{shard}.json"),
        "--batch-size", str(batch_size),
        "--max-new-tokens", "512",
        "--num-shards", "8",
        "--shard-index", str(shard),
    ]


def run_eval_task(job: Job, task: str) -> bool:
    if eval_complete(job, task):
        event("eval_skip_complete", job=job.name, task=task)
        return True
    (OUTPUTS / task).mkdir(parents=True, exist_ok=True)
    (LOGS / task).mkdir(parents=True, exist_ok=True)
    for attempt, batch_size in enumerate(EVAL_BATCH_CANDIDATES, 1):
        update_job(job.name, status=f"evaluating_{task}", gpu="0-7", eval_attempt=attempt, eval_batch_size=batch_size)
        event("eval_start", job=job.name, task=task, attempt=attempt, batch_size=batch_size)
        processes: list[tuple[subprocess.Popen[Any], Any]] = []
        for shard, gpu in enumerate(GPUS):
            log = (LOGS / task / f"{job.name}_shard{shard}_attempt{attempt}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(eval_command(job, task, shard, batch_size), cwd=ROOT, env=run_env(gpu), stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log))
        codes = []
        for process, log in processes:
            codes.append(process.wait())
            log.close()
        if all(code == 0 for code in codes):
            merge = MATH_MERGE if task == "math" else GSM_MERGE
            expected = "5000" if task == "math" else "1319"
            inputs = [str(OUTPUTS / task / f"{job.name}_shard{i}.json") for i in GPUS]
            merge_log = LOGS / task / f"{job.name}_merge_attempt{attempt}.log"
            with merge_log.open("w", encoding="utf-8") as stream:
                merged = subprocess.run(
                    [str(PYTHON), str(merge), "--inputs", *inputs, "--output", str(OUTPUTS / task / f"{job.name}_full.json"), "--expected-samples", expected],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                )
            if merged.returncode == 0 and eval_complete(job, task):
                event("eval_finish", job=job.name, task=task, attempt=attempt, batch_size=batch_size)
                return True
        event("eval_failure", job=job.name, task=task, attempt=attempt, batch_size=batch_size, shard_returncodes=codes)
    update_job(job.name, status=f"failed_{task}", gpu=None)
    return False


def main() -> None:
    for directory in (CHECKPOINTS, OUTPUTS / "math", OUTPUTS / "gsm8k", LOGS / "train", LOGS / "math", LOGS / "gsm8k", HERE / "hf_cache"):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE_FILE, STATE)
    set_phase("training")
    threads = [threading.Thread(target=train_one, args=(job, gpu), daemon=False) for job, gpu in zip(JOBS, GPUS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [job.name for job in JOBS if not checkpoint_complete(job)]
    if failed:
        set_phase("training_failed")
        event("experiment_failure", failed_jobs=failed)
        raise SystemExit(1)
    set_phase("evaluation")
    for job in JOBS:
        if not run_eval_task(job, "math") or not run_eval_task(job, "gsm8k"):
            set_phase("evaluation_failed")
            raise SystemExit(1)
        update_job(job.name, status="complete", gpu=None, finished_at=now())
    set_phase("complete")
    event("experiment_complete")


if __name__ == "__main__":
    main()
