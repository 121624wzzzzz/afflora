#!/usr/bin/env python
from __future__ import annotations

import json
import os
import queue
import statistics
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data").is_dir() and (p / "scripts").is_dir())
PYTHON = Path("/home/wz/anaconda3/envs/torch24/bin/python")
MODELS = {
    "qwen3_06b": Path("/home/wz/projects/mypro/im_exp/models/Qwen3-0.6B-Base"),
    "qwen3_17b": Path("/home/wz/projects/mypro/im_exp/models/Qwen3-1.7B-Base"),
}
TRAIN = ROOT / "data/metamathqa_40k/train.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"

CHECKPOINTS = HERE / "checkpoints"
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
STATE_FILE = HERE / "state_hidden_rank.json"
EVENTS_FILE = HERE / "events_hidden_rank.jsonl"
RESULTS_FILE = HERE / "RESULTS_HIDDEN_RANK.md"

SEED = 42
HIDDEN_RANKS = (1, 2, 4, 8, 16)
GPUS = tuple(range(8))
EVAL_BATCH_CANDIDATES = (512, 256, 128, 64)


@dataclass(frozen=True)
class Job:
    model_alias: str
    hidden_rank: int
    seed: int = SEED

    @property
    def name(self) -> str:
        return f"{self.model_alias}_hidden_hr{self.hidden_rank}_seed{self.seed}"

    @property
    def hidden_alpha(self) -> int:
        return 2 * self.hidden_rank

    @property
    def model_path(self) -> Path:
        return MODELS[self.model_alias]


JOBS = [Job(model_alias, rank) for model_alias in MODELS for rank in HIDDEN_RANKS]
STATE_LOCK = threading.Lock()


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
        write_summary_locked()


def set_phase(phase: str) -> None:
    with STATE_LOCK:
        STATE["phase"] = phase
        STATE["updated_at"] = now()
        atomic_json(STATE_FILE, STATE)
        write_summary_locked()
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
    required = [directory / "adapter_model.safetensors", directory / "adapter_config.json", directory / "run_args.json"]
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
        "--model-path", str(job.model_path),
        "--train-data", str(TRAIN),
        "--output-dir", str(CHECKPOINTS / job.name),
        "--variant", "hidden_lora",
        "--hidden-lora-rank", str(job.hidden_rank),
        "--hidden-lora-alpha", str(job.hidden_alpha),
        "--hidden-lora-dropout", "0.05",
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


def train_one(job: Job, gpu: int) -> bool:
    if checkpoint_complete(job):
        update_job(job.name, status="trained", gpu=None, finished_at=now())
        event("training_skip_complete", job=job.name)
        return True
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
            return True
        update_job(job.name, status="retrying" if attempt == 1 else "failed", gpu=None, returncode=result.returncode)
        event("training_failure", job=job.name, gpu=gpu, attempt=attempt, returncode=result.returncode)
    return False


def training_worker(gpu: int, work: queue.Queue[Job]) -> None:
    while True:
        try:
            job = work.get_nowait()
        except queue.Empty:
            return
        try:
            train_one(job, gpu)
        finally:
            work.task_done()


def eval_command(job: Job, task: str, shard: int, batch_size: int) -> list[str]:
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    return [
        str(PYTHON), str(script),
        "--model-path", str(job.model_path),
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


def result_for(job: Job, task: str) -> float | None:
    if not eval_complete(job, task):
        return None
    report = json.loads((OUTPUTS / task / f"{job.name}_full.json").read_text(encoding="utf-8"))
    return float(report["clean" if task == "math" else "full"]["accuracy_pct"])


def write_summary_locked() -> None:
    status_counts: dict[str, int] = {}
    for item in STATE["jobs"].values():
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    lines = [
        "# Qwen3 close-size hidden LoRA rank sweep",
        "",
        f"Updated: {now()}",
        "",
        f"Phase: `{STATE['phase']}`",
        "",
        "## Runtime status",
        "",
        "| status | jobs |",
        "| --- | ---: |",
    ]
    lines += [f"| {key} | {value} |" for key, value in sorted(status_counts.items())]
    lines += ["", "## Per-run results", "", "| model | hidden rank | seed | MATH clean | GSM8K |", "| --- | ---: | ---: | ---: | ---: |"]
    for job in JOBS:
        m = result_for(job, "math")
        g = result_for(job, "gsm8k")
        lines.append(
            f"| {job.model_alias} | {job.hidden_rank} | {job.seed} | {m:.4f}% | {g:.4f}% |"
            if m is not None and g is not None
            else f"| {job.model_alias} | {job.hidden_rank} | {job.seed} | {'-' if m is None else f'{m:.4f}%'} | {'-' if g is None else f'{g:.4f}%'} |"
        )
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (CHECKPOINTS, OUTPUTS / "math", OUTPUTS / "gsm8k", LOGS / "train", LOGS / "math", LOGS / "gsm8k", HERE / "hf_cache"):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE_FILE, STATE)
    set_phase("training")
    work: queue.Queue[Job] = queue.Queue()
    for job in JOBS:
        if checkpoint_complete(job):
            update_job(job.name, status="trained", gpu=None)
        else:
            work.put(job)
    threads = [threading.Thread(target=training_worker, args=(gpu, work), daemon=False) for gpu in GPUS]
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
