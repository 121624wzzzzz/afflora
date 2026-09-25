#!/usr/bin/env python
"""Constrained A-LoRA topology sweep on untied Llama-3.1 and tied Llama-3.2."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
TRAIN_DATA = ROOT / "data/metamathqa_40k/train.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
CROSS = ROOT / "reviewer_followup/llama_cross_version"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"
CHECKPOINTS = HERE / "topology_checkpoints"
OUTPUTS = HERE / "topology_outputs"
LOGS = HERE / "topology_logs"
STATE = HERE / "constrained_topology_state.json"
EVENTS = HERE / "constrained_topology_events.jsonl"
RESULTS = HERE / "CONSTRAINED_TOPOLOGY_RESULTS.md"
TAU = 0.00625
ENERGY_LAMBDA = 100.0
NUM_SHARDS = 16
LOCK = threading.Lock()


MODEL_INFO = {
    "llama31_8b": {
        "path": ROOT.parent / "models/Llama-3.1-8B-Base",
        "batch": 4,
        "accum": 4,
        "baseline": "llama31_8b_hidden_hr4_seed42_full.json",
    },
    "llama32_3b": {
        "path": ROOT.parent / "models/Llama-3.2-3B-Base",
        "batch": 8,
        "accum": 2,
        "baseline": "llama32_3b_hidden_hr4_seed42_full.json",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Job:
    model_alias: str
    topology: str
    variant: str
    lm_head_bias: bool = False
    tie_adapters: bool = False

    @property
    def name(self) -> str:
        return (
            f"{self.model_alias}_{self.topology}_ar16_s1_hr4_"
            "energy_tau0p00625_l100_seed42"
        )

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINTS / self.name

    @property
    def model_path(self) -> Path:
        return MODEL_INFO[self.model_alias]["path"]


JOBS = (
    Job("llama31_8b", "input", "affine_input_plus_hidden_lora"),
    Job("llama31_8b", "decoupled", "affine_input_lm_head_plus_hidden_lora"),
    Job("llama32_3b", "input", "affine_input_plus_hidden_lora"),
    Job("llama32_3b", "output", "affine_lm_head_plus_hidden_lora"),
    Job("llama32_3b", "decoupled", "affine_input_lm_head_plus_hidden_lora"),
    Job(
        "llama32_3b", "shared", "affine_input_lm_head_plus_hidden_lora",
        lm_head_bias=True, tie_adapters=True,
    ),
)


def event(kind: str, **payload: Any) -> None:
    with LOCK:
        with EVENTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "kind": kind, **payload}) + "\n")


def write_state(phase: str, **payload: Any) -> None:
    with LOCK:
        STATE.write_text(
            json.dumps({"phase": phase, "updated_at": now(), **payload}, indent=2),
            encoding="utf-8",
        )


def env(gpu: int) -> dict[str, str]:
    value = os.environ.copy()
    value["CUDA_VISIBLE_DEVICES"] = str(gpu)
    value["DS_IGNORE_CUDA_DETECTION"] = "1"
    return value


def checkpoint_complete(job: Job) -> bool:
    files = ("adapter_model.safetensors", "affine_vocab_adapter.safetensors", "run_args.json")
    return all((job.checkpoint / filename).is_file() for filename in files)


def train(job: Job, gpu: int) -> bool:
    if checkpoint_complete(job):
        event("training_reused", job=job.name, gpu=gpu)
        return True
    info = MODEL_INFO[job.model_alias]
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT), "--model-path", str(job.model_path),
        "--train-data", str(TRAIN_DATA), "--output-dir", str(job.checkpoint),
        "--variant", job.variant, "--hidden-lora-rank", "4",
        "--hidden-lora-alpha", "8", "--hidden-lora-dropout", "0.05",
        "--affine-rank", "16", "--affine-alpha", "16",
        "--affine-energy-tau", str(TAU), "--affine-energy-lambda", str(ENERGY_LAMBDA),
        "--max-seq-len", "1024", "--per-device-train-batch-size", str(info["batch"]),
        "--gradient-accumulation-steps", str(info["accum"]), "--learning-rate", "2e-4",
        "--num-train-epochs", "1", "--logging-steps", "10", "--bf16",
        "--gradient-checkpointing", "--seed", "42", "--master-dtype", "fp32",
        "--base-dtype", "bf16", "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]
    if job.lm_head_bias:
        cmd.append("--affine-lm-head-bias")
    if job.tie_adapters:
        cmd.append("--tie-affine-input-lm-head-adapters")
    log = LOGS / "training" / f"{job.name}.log"
    event("training_started", job=job.name, gpu=gpu, topology=job.topology)
    with log.open("w", encoding="utf-8") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and checkpoint_complete(job)
    event("training_finished", job=job.name, gpu=gpu, returncode=rc, success=ok)
    return ok


def shard(job: Job, task: str, index: int) -> Path:
    return OUTPUTS / task / f"{job.name}_shard{index}.json"


def full(job: Job, task: str) -> Path:
    return OUTPUTS / task / f"{job.name}_full.json"


def eval_one(job: Job, task: str, index: int, gpu: int) -> bool:
    target = shard(job, task, index)
    if target.is_file():
        return True
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    for attempt, batch in enumerate((64, 32, 16), 1):
        cmd = [
            str(PYTHON), str(script), "--model-path", str(job.model_path),
            "--run-dir", str(job.checkpoint), "--eval-data", str(data),
            "--output-file", str(target), "--batch-size", str(batch),
            "--max-new-tokens", "512", "--num-shards", str(NUM_SHARDS),
            "--shard-index", str(index),
        ]
        log = LOGS / task / f"{job.name}_shard{index}_attempt{attempt}.log"
        with log.open("w", encoding="utf-8") as stream:
            rc = subprocess.run(cmd, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc == 0 and target.is_file():
            event("eval_shard_finished", job=job.name, task=task, shard=index, gpu=gpu)
            return True
        event("eval_retry", job=job.name, task=task, shard=index, gpu=gpu, attempt=attempt, returncode=rc)
    return False


def dynamic_run() -> None:
    work: queue.Queue[tuple[str, Job, str | None, int | None] | None] = queue.Queue()
    failures: list[str] = []
    for job in JOBS:
        work.put(("train", job, None, None))

    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            kind, job, task, index = item
            try:
                if kind == "train":
                    if train(job, gpu):
                        for eval_task in ("math", "gsm8k"):
                            for shard_index in range(NUM_SHARDS):
                                work.put(("eval", job, eval_task, shard_index))
                    else:
                        with LOCK:
                            failures.append(f"train:{job.name}")
                else:
                    assert task is not None and index is not None
                    if not eval_one(job, task, index, gpu):
                        with LOCK:
                            failures.append(f"eval:{job.name}:{task}:{index}")
            except Exception as exc:
                event("exception", job=job.name, task=task, index=index, gpu=gpu, error=repr(exc))
                with LOCK:
                    failures.append(f"exception:{job.name}:{task}:{index}:{exc!r}")
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
        raise RuntimeError(f"Topology sweep failures: {failures}")


def merge_outputs() -> None:
    for job in JOBS:
        for task in ("math", "gsm8k"):
            script = MATH_MERGE if task == "math" else GSM_MERGE
            expected = "5000" if task == "math" else "1319"
            inputs = [str(shard(job, task, index)) for index in range(NUM_SHARDS)]
            rc = subprocess.run(
                [str(PYTHON), str(script), "--inputs", *inputs,
                 "--output", str(full(job, task)), "--expected-samples", expected],
                cwd=ROOT,
            ).returncode
            if rc != 0:
                raise RuntimeError(f"Merge failed: {job.name}:{task}")


def matrix_stats(weight: torch.Tensor, centered: bool) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    vocab, hidden = weight.shape
    covariance = torch.zeros((hidden, hidden), dtype=torch.float32, device=weight.device)
    row_sum = torch.zeros(hidden, dtype=torch.float32, device=weight.device)
    for begin in range(0, vocab, 4096):
        chunk = weight[begin : begin + 4096].float()
        covariance.addmm_(chunk.T, chunk)
        row_sum.add_(chunk.sum(dim=0))
    if centered:
        covariance.add_(torch.outer(row_sum, row_sum), alpha=-1.0 / float(vocab))
        row_sum.zero_()
    return covariance, row_sum, float(covariance.diagonal().sum().item()), vocab


def raw_rho(
    state: dict[str, torch.Tensor], prefix: str,
    stats: tuple[torch.Tensor, torch.Tensor, float, int],
) -> float:
    covariance, row_sum, denominator, vocab = stats
    up = state[f"{prefix}.up.weight"].float().cuda()
    down = state[f"{prefix}.down.weight"].float().cuda()
    numerator = ((up.T @ covariance @ up) * (down @ down.T)).sum()
    bias_key = f"{prefix}.bias"
    if bias_key in state:
        beta = state[bias_key].float().cuda()
        numerator = numerator + 2.0 * (((row_sum @ up) @ down) * beta).sum()
        numerator = numerator + float(vocab) * beta.square().sum()
    return math.sqrt(max(float(numerator.item()) / denominator, 0.0))


def exact_rhos() -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for alias in MODEL_INFO:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_INFO[alias]["path"], torch_dtype=torch.bfloat16
        ).cuda().eval()
        input_stats = matrix_stats(model.get_input_embeddings().weight.detach(), centered=False)
        output_stats = matrix_stats(model.get_output_embeddings().weight.detach(), centered=True)
        for job in (item for item in JOBS if item.model_alias == alias):
            state = load_file(str(job.checkpoint / "affine_vocab_adapter.safetensors"))
            row: dict[str, float] = {}
            if job.topology in ("input", "decoupled", "shared"):
                row["input"] = raw_rho(state, "model.embed_tokens.affine", input_stats)
            if job.topology in ("output", "decoupled"):
                row["output"] = raw_rho(state, "lm_head.affine", output_stats)
            values[job.name] = row
        del model, input_stats, output_stats
        torch.cuda.empty_cache()
    return values


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def baseline(alias: str, task: str) -> float:
    filename = MODEL_INFO[alias]["baseline"]
    key = "clean" if task == "math" else "full"
    return accuracy(CROSS / f"outputs/{task}" / filename, key)


def write_report(rhos: dict[str, dict[str, float]]) -> None:
    lines = [
        "# Constrained A-LoRA topology sweep", "", f"Updated: {now()}", "",
        "All treatment rows use joint-from-scratch hidden LoRA r4 + A-LoRA r16, seed 42, tau=0.00625, lambda=100.", "",
        "| model | topology | input rho | output rho | MATH | delta vs hidden | GSM8K | delta vs hidden |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for alias in MODEL_INFO:
        bm, bg = baseline(alias, "math"), baseline(alias, "gsm8k")
        lines.append(f"| {alias} | hidden baseline | - | - | {bm:.4f}% | - | {bg:.4f}% | - |")
        if alias == "llama31_8b":
            em = accuracy(HERE / "strength_sweep_outputs/math/llama31_8b_lmhead_ar16_s1_hr4_energy_tau0p00625_l100_seed42_full.json", "clean")
            eg = accuracy(HERE / "strength_sweep_outputs/gsm8k/llama31_8b_lmhead_ar16_s1_hr4_energy_tau0p00625_l100_seed42_full.json", "full")
            lines.append(
                f"| {alias} | output | - | 0.00623416 | {em:.4f}% | {em-bm:+.4f} pp | "
                f"{eg:.4f}% | {eg-bg:+.4f} pp |"
            )
        for job in (item for item in JOBS if item.model_alias == alias):
            math_acc = accuracy(full(job, "math"), "clean")
            gsm_acc = accuracy(full(job, "gsm8k"), "full")
            row = rhos[job.name]
            in_rho = f"{row['input']:.8f}" if "input" in row else "-"
            out_rho = f"{row['output']:.8f}" if "output" in row else ("shared" if job.topology == "shared" else "-")
            lines.append(
                f"| {alias} | {job.topology} | {in_rho} | {out_rho} | "
                f"{math_acc:.4f}% | {math_acc-bm:+.4f} pp | "
                f"{gsm_acc:.4f}% | {gsm_acc-bg:+.4f} pp |"
            )
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    for directory in (
        CHECKPOINTS, OUTPUTS / "math", OUTPUTS / "gsm8k",
        LOGS / "training", LOGS / "math", LOGS / "gsm8k",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if EVENTS.exists():
        EVENTS.unlink()
    write_state("dynamic_training_and_evaluation", jobs=[job.name for job in JOBS])
    dynamic_run()
    write_state("merging")
    merge_outputs()
    write_state("energy_measurement")
    rhos = exact_rhos()
    write_report(rhos)
    write_state("complete", rhos=rhos, result=str(RESULTS))
    event("complete", result=str(RESULTS))


if __name__ == "__main__":
    main()
