#!/usr/bin/env python
"""Train/evaluate one centered-output energy-constrained Llama-3.1 A-LoRA run."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
MODEL = ROOT.parent / "models/Llama-3.1-8B-Base"
TRAIN = ROOT / "data/metamathqa_40k/train.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/train_affine_vocab_lora.py"
MATH_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py"
GSM_EVAL = ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py"
MATH_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_math_shards.py"
GSM_MERGE = ROOT / "corrected_math_evaluation/shared/merge/merge_gsm8k_shards.py"
CROSS = ROOT / "reviewer_followup/llama_cross_version"
NAME = "llama31_8b_lmhead_ar16_s1_hr4_energy_tau0p00729_l100_seed42"
CHECKPOINT = HERE / "checkpoints" / NAME
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
STATE = HERE / "energy_run_state.json"
RESULTS = HERE / "ENERGY_CONSTRAINED_RESULTS.md"
TAU = 0.00729014
ENERGY_LAMBDA = 100.0
NUM_SHARDS = 16
LOCK = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def env(gpu: int) -> dict[str, str]:
    value = os.environ.copy(); value["CUDA_VISIBLE_DEVICES"] = str(gpu)
    value["DS_IGNORE_CUDA_DETECTION"] = "1"; return value


def checkpoint_complete() -> bool:
    files = ("adapter_model.safetensors", "affine_vocab_adapter.safetensors", "run_args.json")
    return all((CHECKPOINT / f).is_file() for f in files)


def train() -> None:
    if checkpoint_complete(): return
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT), "--model-path", str(MODEL),
        "--train-data", str(TRAIN), "--output-dir", str(CHECKPOINT),
        "--variant", "affine_lm_head_plus_hidden_lora",
        "--hidden-lora-rank", "4", "--hidden-lora-alpha", "8",
        "--hidden-lora-dropout", "0.05", "--affine-rank", "16",
        "--affine-alpha", "16", "--affine-energy-tau", str(TAU),
        "--affine-energy-lambda", str(ENERGY_LAMBDA), "--max-seq-len", "1024",
        "--per-device-train-batch-size", "4", "--gradient-accumulation-steps", "4",
        "--learning-rate", "2e-4", "--num-train-epochs", "1",
        "--logging-steps", "10", "--bf16", "--gradient-checkpointing",
        "--seed", "42", "--master-dtype", "fp32", "--base-dtype", "bf16",
        "--save-strategy", "epoch", "--save-total-limit", "1",
        "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
    ]
    with (LOGS / "train.log").open("w") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env(0), stdout=stream, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not checkpoint_complete(): raise RuntimeError(f"training failed: {rc}")


def shard(task: str, index: int) -> Path:
    return OUTPUTS / task / f"{NAME}_shard{index}.json"


def full(task: str) -> Path:
    return OUTPUTS / task / f"{NAME}_full.json"


def eval_one(task: str, index: int, gpu: int) -> bool:
    script = MATH_EVAL if task == "math" else GSM_EVAL
    data = ROOT / ("data/math/test.jsonl" if task == "math" else "data/gsm8k/test.jsonl")
    for attempt, batch in enumerate((64, 32, 16), 1):
        cmd = [
            str(PYTHON), str(script), "--model-path", str(MODEL),
            "--run-dir", str(CHECKPOINT), "--eval-data", str(data),
            "--output-file", str(shard(task, index)), "--batch-size", str(batch),
            "--max-new-tokens", "512", "--num-shards", str(NUM_SHARDS),
            "--shard-index", str(index),
        ]
        log = LOGS / task / f"shard{index}_attempt{attempt}.log"
        with log.open("w") as stream:
            rc = subprocess.run(cmd, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc == 0 and shard(task, index).is_file(): return True
    return False


def evaluate() -> None:
    work: queue.Queue[Any] = queue.Queue()
    for task in ("math", "gsm8k"):
        for index in range(NUM_SHARDS): work.put((task, index))
    failures: list[str] = []
    def worker(gpu: int) -> None:
        while True:
            item = work.get()
            if item is None: work.task_done(); return
            task, index = item
            try:
                if not eval_one(task, index, gpu):
                    with LOCK: failures.append(f"{task}:{index}")
            finally: work.task_done()
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(8)]
    for thread in threads: thread.start()
    work.join()
    for _ in threads: work.put(None)
    for thread in threads: thread.join()
    if failures: raise RuntimeError(f"eval failures: {failures}")
    for task in ("math", "gsm8k"):
        merge = MATH_MERGE if task == "math" else GSM_MERGE
        expected = "5000" if task == "math" else "1319"
        inputs = [str(shard(task, i)) for i in range(NUM_SHARDS)]
        rc = subprocess.run([
            str(PYTHON), str(merge), "--inputs", *inputs, "--output", str(full(task)),
            "--expected-samples", expected,
        ], cwd=ROOT).returncode
        if rc != 0: raise RuntimeError(f"merge failed: {task}")


def exact_rho() -> float:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    weight = model.lm_head.weight.detach(); vocab = weight.shape[0]
    mean = weight.float().mean(dim=0); denom = 0.0
    state = load_file(str(CHECKPOINT / "affine_vocab_adapter.safetensors"))
    up = state["lm_head.affine.up.weight"].float().cuda()
    down = state["lm_head.affine.down.weight"].float().cuda()
    ata = torch.zeros((16, 16), dtype=torch.float64, device="cuda")
    for begin in range(0, vocab, 4096):
        centered = weight[begin : begin + 4096].float() - mean
        denom += float(centered.square().sum().item())
        a = centered @ up; ata += a.T.double() @ a.double()
    ddt = down.double() @ down.double().T
    return math.sqrt(float((ata * ddt).sum().item()) / denom)


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text())[key]["accuracy_pct"])


def report(rho: float) -> None:
    base_m = accuracy(CROSS / "outputs/math/llama31_8b_hidden_hr4_seed42_full.json", "clean")
    base_g = accuracy(CROSS / "outputs/gsm8k/llama31_8b_hidden_hr4_seed42_full.json", "full")
    unc_m = accuracy(CROSS / "outputs/math/llama31_8b_lmhead_ar16_s1_hr4_seed42_full.json", "clean")
    unc_g = accuracy(CROSS / "outputs/gsm8k/llama31_8b_lmhead_ar16_s1_hr4_seed42_full.json", "full")
    con_m = accuracy(full("math"), "clean"); con_g = accuracy(full("gsm8k"), "full")
    text = f"""# Energy-constrained Llama-3.1 A-LoRA result

Updated: {now()}

- Constraint: tau={TAU}, lambda={ENERGY_LAMBDA}.
- Exact final centered-output rho: {rho:.8f}.

| config | MATH | Δ vs hidden | GSM8K | Δ vs hidden |
| --- | ---: | ---: | ---: | ---: |
| hidden r4 | {base_m:.4f}% | - | {base_g:.4f}% | - |
| unconstrained A r16 | {unc_m:.4f}% | {unc_m-base_m:+.4f} pp | {unc_g:.4f}% | {unc_g-base_g:+.4f} pp |
| energy-constrained A r16 | {con_m:.4f}% | {con_m-base_m:+.4f} pp | {con_g:.4f}% | {con_g-base_g:+.4f} pp |
"""
    RESULTS.write_text(text); print(text)


def main() -> None:
    for directory in (HERE / "checkpoints", OUTPUTS / "math", OUTPUTS / "gsm8k", LOGS / "math", LOGS / "gsm8k"):
        directory.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"started_at": now(), "phase": "training"}, indent=2))
    train(); STATE.write_text(json.dumps({"phase": "evaluation", "updated_at": now()}, indent=2))
    evaluate(); STATE.write_text(json.dumps({"phase": "energy_measurement", "updated_at": now()}, indent=2))
    rho = exact_rho(); report(rho)
    STATE.write_text(json.dumps({"phase": "complete", "finished_at": now(), "rho": rho}, indent=2))


if __name__ == "__main__":
    main()
