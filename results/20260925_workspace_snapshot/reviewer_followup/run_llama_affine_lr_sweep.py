#!/usr/bin/env python
"""Select A-LoRA rank, then screen a separate A-LoRA LR multiplier."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reviewer_followup/llama_affine_lr_sweep"
RANK_SWEEP = ROOT / "reviewer_followup/llama_affine_rank_sweep"
CROSS = ROOT / "reviewer_followup/llama_cross_version/checkpoints"
HIDDEN_SWEEP = ROOT / "reviewer_followup/llama_hidden_rank_sweep"
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
OUTPUTS = HERE / "outputs/metamath_loss"
LOGS = HERE / "logs"
STATE_FILE = HERE / "state.json"
EVENTS_FILE = HERE / "events.jsonl"
RESULTS_FILE = HERE / "RESULTS.md"
SELECTION_FILE = HERE / "selection.json"
AFFINE_RANKS = (4, 8, 16, 32)
LR_SCALES = (0.1, 0.25, 0.5, 0.75, 1.0)
GPUS = tuple(range(8))
SEED = 42
STATE_LOCK = threading.Lock()
QUEUE_LOCK = threading.Lock()
STATE: dict[str, Any] = {}
JOBS: list["Job"] = []


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def placement(alias: str) -> str:
    return "lmhead" if alias == "llama31_8b" else "mergeable"


def rank_name(alias: str, rank: int) -> str:
    return f"{alias}_{placement(alias)}_ar{rank}_s1_hr4_seed{SEED}"


def rank_eval_path(alias: str, rank: int) -> Path:
    return RANK_SWEEP / "outputs/metamath_loss" / f"{rank_name(alias, rank)}.json"


def rank_checkpoint(alias: str, rank: int) -> Path:
    name = rank_name(alias, rank)
    return (CROSS if rank == 16 else RANK_SWEEP / "checkpoints") / name


def select_ranks() -> dict[str, int]:
    selected: dict[str, int] = {}
    audit: dict[str, Any] = {"selected_at": now(), "models": {}}
    for alias in MODELS:
        losses: dict[str, float] = {}
        for rank in AFFINE_RANKS:
            path = rank_eval_path(alias, rank)
            report = json.loads(path.read_text(encoding="utf-8"))
            loss = float(report["loss"])
            if report["num_samples"] != 499 or not math.isfinite(loss):
                raise ValueError(f"Invalid rank result: {path}")
            losses[str(rank)] = loss
        best = min(AFFINE_RANKS, key=lambda rank: losses[str(rank)])
        selected[alias] = best
        audit["models"][alias] = {"losses": losses, "selected_rank": best}
    SELECTION_FILE.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return selected


@dataclass(frozen=True)
class Job:
    model_alias: str
    affine_rank: int
    lr_scale: float

    @property
    def name(self) -> str:
        return (
            f"{self.model_alias}_{placement(self.model_alias)}_ar{self.affine_rank}_"
            f"s1_hr4_alr{tag(self.lr_scale)}_seed{SEED}"
        )

    @property
    def model_path(self) -> Path:
        return MODELS[self.model_alias]

    @property
    def reused(self) -> bool:
        return self.lr_scale == 1.0

    @property
    def checkpoint_dir(self) -> Path:
        return (
            rank_checkpoint(self.model_alias, self.affine_rank)
            if self.reused else CHECKPOINTS / self.name
        )

    @property
    def output_path(self) -> Path:
        return (
            rank_eval_path(self.model_alias, self.affine_rank)
            if self.reused else OUTPUTS / f"{self.name}.json"
        )


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def event(kind: str, **fields: Any) -> None:
    with STATE_LOCK:
        with EVENTS_FILE.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "kind": kind, **fields}) + "\n")


def update_job(name: str, **fields: Any) -> None:
    with STATE_LOCK:
        STATE["jobs"][name].update(fields)
        STATE["updated_at"] = now()
        atomic_json(STATE_FILE, STATE)
        write_summary_locked()


def checkpoint_complete(job: Job) -> bool:
    files = (
        "adapter_model.safetensors", "adapter_config.json",
        "affine_vocab_adapter.safetensors", "affine_vocab_config.json", "run_args.json",
    )
    return all((job.checkpoint_dir / f).is_file() for f in files)


def eval_complete(job: Job) -> bool:
    if not job.output_path.is_file():
        return False
    try:
        d = json.loads(job.output_path.read_text(encoding="utf-8"))
        return d["num_samples"] == 499 and math.isfinite(float(d["loss"]))
    except (KeyError, ValueError, json.JSONDecodeError):
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
    batch, accum = (4, 4) if job.model_alias == "llama31_8b" else (8, 2)
    variant = (
        "affine_lm_head_plus_hidden_lora" if job.model_alias == "llama31_8b"
        else "affine_input_lm_head_plus_hidden_lora"
    )
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT), "--model-path", str(job.model_path),
        "--train-data", str(TRAIN), "--output-dir", str(job.checkpoint_dir),
        "--variant", variant, "--hidden-lora-rank", "4", "--hidden-lora-alpha", "8",
        "--hidden-lora-dropout", "0.05", "--affine-rank", str(job.affine_rank),
        "--affine-alpha", str(job.affine_rank),
        "--affine-learning-rate-scale", str(job.lr_scale),
        "--max-seq-len", "1024", "--per-device-train-batch-size", str(batch),
        "--gradient-accumulation-steps", str(accum), "--learning-rate", "2e-4",
        "--num-train-epochs", "1", "--logging-steps", "10", "--bf16",
        "--gradient-checkpointing", "--seed", str(SEED), "--master-dtype", "fp32",
        "--base-dtype", "bf16", "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]
    if job.model_alias == "llama32_3b":
        cmd += ["--tie-affine-input-lm-head-adapters", "--affine-lm-head-bias"]
    return cmd


def train_one(job: Job, gpu: int) -> bool:
    for attempt in (1, 2):
        if checkpoint_complete(job):
            return True
        update_job(job.name, status="training", gpu=gpu, attempts=attempt)
        event("training_start", job=job.name, gpu=gpu, attempt=attempt)
        log = LOGS / "train" / f"{job.name}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(
                train_command(job), cwd=ROOT, env=run_env(gpu),
                stdout=stream, stderr=subprocess.STDOUT,
            ).returncode
        if rc == 0 and checkpoint_complete(job):
            update_job(job.name, status="trained", gpu=None)
            event("training_finish", job=job.name, gpu=gpu, attempt=attempt)
            return True
        event("training_failure", job=job.name, gpu=gpu, attempt=attempt, returncode=rc)
    update_job(job.name, status="failed", gpu=None)
    return False


def eval_one(job: Job, gpu: int) -> bool:
    if eval_complete(job):
        update_job(job.name, status="complete", gpu=None)
        return True
    batches = (8, 4, 2) if job.model_alias == "llama31_8b" else (16, 8, 4)
    for attempt, batch in enumerate(batches, 1):
        update_job(job.name, status="evaluating_dev", gpu=gpu, eval_batch_size=batch)
        event("eval_start", job=job.name, gpu=gpu, attempt=attempt, batch_size=batch)
        cmd = [
            str(PYTHON), str(LOSS_EVAL), "--model-path", str(job.model_path),
            "--run-dir", str(job.checkpoint_dir), "--eval-data", str(DEV),
            "--output-file", str(job.output_path), "--batch-size", str(batch),
            "--max-seq-len", "1024", "--dtype", "bf16",
        ]
        log = LOGS / "eval" / f"{job.name}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(
                cmd, cwd=ROOT, env=run_env(gpu), stdout=stream, stderr=subprocess.STDOUT
            ).returncode
        if rc == 0 and eval_complete(job):
            update_job(job.name, status="complete", gpu=None)
            event("eval_finish", job=job.name, gpu=gpu, attempt=attempt, batch_size=batch)
            return True
    update_job(job.name, status="eval_failed", gpu=None)
    return False


def hidden_loss(alias: str) -> float:
    p = HIDDEN_SWEEP / "outputs/metamath_loss" / f"{alias}_hidden_hr4_seed{SEED}.json"
    return float(json.loads(p.read_text(encoding="utf-8"))["loss"])


def write_summary_locked() -> None:
    counts: dict[str, int] = {}
    for v in STATE.get("jobs", {}).values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    lines = [
        "# Llama A-LoRA learning-rate screening", "", f"Updated: {now()}", "",
        f"Phase: `{STATE.get('phase', 'initializing')}`", "", "| status | jobs |",
        "| --- | ---: |",
    ] + [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += [
        "", "| model | A rank | A LR scale | A LR | hidden CE | treatment CE | ΔCE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for job in JOBS:
        loss = None
        if eval_complete(job):
            loss = float(json.loads(job.output_path.read_text(encoding="utf-8"))["loss"])
        base = hidden_loss(job.model_alias)
        fmt = lambda x: "-" if x is None else f"{x:.8f}"
        lines.append(
            f"| {job.model_alias} | {job.affine_rank} | {job.lr_scale:g} | "
            f"{2e-4 * job.lr_scale:.2e} | {fmt(base)} | {fmt(loss)} | "
            f"{fmt(None if loss is None else loss-base)} |"
        )
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global JOBS, STATE
    for d in (HERE, CHECKPOINTS, OUTPUTS, LOGS / "train", LOGS / "eval", HERE / "hf_cache/home"):
        d.mkdir(parents=True, exist_ok=True)
    while tmux_exists("alora_review_llama_affine_rank"):
        time.sleep(5)
    selected = select_ranks()
    JOBS = [Job(alias, selected[alias], scale) for alias in MODELS for scale in LR_SCALES]
    prior = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    old = prior.get("jobs", {})
    STATE = {
        "schema_version": 1, "started_at": prior.get("started_at", now()),
        "updated_at": now(), "phase": "mixed_training_and_dev_evaluation",
        "selected_ranks": selected,
        "jobs": {j.name: old.get(j.name, {"status": "pending", "attempts": 0}) for j in JOBS},
    }
    atomic_json(STATE_FILE, STATE); write_summary_locked()
    event("phase", phase=STATE["phase"], selected_ranks=selected)

    work: queue.Queue[tuple[str, Job] | None] = queue.Queue()
    queued: set[str] = set()
    def enqueue_eval(job: Job) -> None:
        with QUEUE_LOCK:
            if job.name not in queued:
                queued.add(job.name); work.put(("eval", job))
    for job in JOBS:
        if job.reused:
            update_job(job.name, status="complete", gpu=None, reused=True)
        elif checkpoint_complete(job):
            enqueue_eval(job)
        else:
            work.put(("train", job))
    failures: list[str] = []
    lock = threading.Lock()
    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done(); return
            kind, job = item
            try:
                ok = train_one(job, gpu) if kind == "train" else eval_one(job, gpu)
                if kind == "train" and ok: enqueue_eval(job)
                elif not ok:
                    with lock: failures.append(f"{kind}:{job.name}")
            finally: work.task_done()
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in GPUS]
    for t in threads: t.start()
    work.join()
    for _ in threads: work.put(None)
    for t in threads: t.join()
    STATE["phase"] = "failed" if failures else "complete"; STATE["updated_at"] = now()
    atomic_json(STATE_FILE, STATE); write_summary_locked()
    event("experiment_failure" if failures else "experiment_complete", failures=failures)
    if failures: raise SystemExit(failures)


if __name__ == "__main__":
    main()
