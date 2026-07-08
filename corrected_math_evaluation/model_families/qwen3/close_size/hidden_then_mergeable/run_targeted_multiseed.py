#!/usr/bin/env python
from __future__ import annotations

import json
import os
import queue
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
MODEL_HIDDEN_RANK = {"qwen3_06b": 8, "qwen3_17b": 4}
MODEL_AFFINE_SCALE = {"qwen3_06b": 8.0, "qwen3_17b": 1.0}
MODEL_AFFINE_RANKS = {"qwen3_06b": (1, 2), "qwen3_17b": (2, 8)}
TRAIN = ROOT / "data/metamathqa_40k/train.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"

CHECKPOINTS = HERE / "checkpoints"
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
STATE_FILE = HERE / "state_targeted_multiseed.json"
EVENTS_FILE = HERE / "events_targeted_multiseed.jsonl"
SEEDS_TO_RUN = (43, 44)
GPUS = tuple(range(8))
EVAL_BATCH_CANDIDATES = (512, 256, 128, 64)


def scale_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


@dataclass(frozen=True)
class Job:
    model_alias: str
    seed: int
    affine_rank: int | None

    @property
    def hidden_rank(self) -> int:
        return MODEL_HIDDEN_RANK[self.model_alias]

    @property
    def affine_scale(self) -> float:
        return MODEL_AFFINE_SCALE[self.model_alias]

    @property
    def is_mergeable(self) -> bool:
        return self.affine_rank is not None

    @property
    def name(self) -> str:
        if self.affine_rank is None:
            return f"{self.model_alias}_hidden_hr{self.hidden_rank}_seed{self.seed}"
        return (
            f"{self.model_alias}_mergeable_ar{self.affine_rank}_s{scale_tag(self.affine_scale)}"
            f"_hr{self.hidden_rank}_seed{self.seed}"
        )

    @property
    def variant(self) -> str:
        return "affine_input_lm_head_plus_hidden_lora" if self.is_mergeable else "hidden_lora"

    @property
    def hidden_alpha(self) -> int:
        return 2 * self.hidden_rank

    @property
    def affine_alpha(self) -> float:
        assert self.affine_rank is not None
        return self.affine_rank * self.affine_scale

    @property
    def model_path(self) -> Path:
        return MODELS[self.model_alias]


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for model_alias in MODELS:
        for seed in SEEDS_TO_RUN:
            jobs.append(Job(model_alias, seed, None))
        for rank in MODEL_AFFINE_RANKS[model_alias]:
            for seed in SEEDS_TO_RUN:
                jobs.append(Job(model_alias, seed, rank))
    return jobs


JOBS = build_jobs()
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
    required = [directory / "adapter_model.safetensors", directory / "adapter_config.json", directory / "run_args.json"]
    if job.is_mergeable:
        required.extend([directory / "affine_vocab_adapter.safetensors", directory / "affine_vocab_config.json"])
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
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT),
        "--model-path", str(job.model_path),
        "--train-data", str(TRAIN),
        "--output-dir", str(CHECKPOINTS / job.name),
        "--variant", job.variant,
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
    if job.is_mergeable:
        cmd += [
            "--affine-rank", str(job.affine_rank),
            "--affine-alpha", f"{job.affine_alpha:g}",
            "--tie-affine-input-lm-head-adapters",
            "--affine-lm-head-bias",
        ]
    return cmd


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
