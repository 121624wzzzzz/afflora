#!/usr/bin/env python3
"""Restartable P0 matrix: snapshot, smoke, train, independently evaluate, summarize."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = "/home/wz/anaconda3/envs/torch24/bin/python"
MODELS = {
    "qwen3_06b": ROOT.parent / "models/Qwen3-0.6B-Base",
    "qwen25_7b": ROOT.parent / "models/Qwen2.5-7B-Base",
}
SOURCES = [
    "scripts/train_affine_vocab_lora.py", "src/affine_vocab_lora/__init__.py",
    "src/affine_vocab_lora/adapter.py", "corrected_sft_experiment/data_pipeline.py",
    "corrected_sft_experiment/train_corrected_sft.py",
    "corrected_sft_experiment/evaluate_corrected_sft.py",
]
LOCK = threading.RLock()
STATE = {}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare():
    manifest_path = HERE / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest["sha256"].items():
            if sha(HERE / relative) != expected:
                raise RuntimeError(f"Frozen protocol/source/data changed: {relative}")
        return manifest
    paths = []
    for relative in SOURCES:
        target = HERE / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        paths.append(target)
    for name in ("train", "dev", "test", "manifest"):
        suffix = ".json" if name == "manifest" else ".jsonl"
        target = HERE / "data" / (name + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "corrected_sft_experiment/data" / target.name, target)
        paths.append(target)
    original = json.loads((HERE / "data/manifest.json").read_text())
    rows = {}
    identities = {}
    for split, count in (("train", 22780), ("dev", 1000), ("test", 1000)):
        path = HERE / "data" / f"{split}.jsonl"
        if sha(path) != original["generated_files"][split]["sha256"]:
            raise RuntimeError(f"Original data manifest hash mismatch: {split}")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        ids = [record["record_id"] for record in records]
        assert len(records) == count and len(set(ids)) == count
        identities[split] = set(ids)
        rows[split] = count
    assert not identities["train"] & (identities["dev"] | identities["test"])
    assert not identities["dev"] & identities["test"]
    for filename in ("train.py", "run.py", "summarize.py", "launch.sh", "DESIGN.md"):
        paths.append(HERE / filename)
    manifest = {
        "created_at": now(), "python": PYTHON, "seeds": [42, 43, 44],
        "hidden_ranks": [0, 1, 8], "boundary": ["none", "input", "output"],
        "data_rows": rows, "models": {},
        "sha256": {str(path.relative_to(HERE)): sha(path) for path in paths},
    }
    for alias, path in MODELS.items():
        config = json.loads((path / "config.json").read_text())
        weights = list(path.glob("*.safetensors"))
        assert weights, f"Missing local weights: {path}"
        manifest["models"][alias] = {
            "path": str(path), "hidden_size": config["hidden_size"],
            "tie_word_embeddings": config["tie_word_embeddings"],
            "config_sha256": sha(path / "config.json"),
            "weight_files": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights},
        }
    write_json(manifest_path, manifest)
    return manifest


def jobs(smoke=False):
    if smoke:
        return [make_job(alias, 1, placement, 42, smoke=True)
                for alias in MODELS for placement in ("input", "output")]
    result = []
    for seed in (42, 43, 44):
        for rank in (0, 8, 1):
            for alias in MODELS:
                for placement in ("input", "output", "none"):
                    if rank == 0 and placement == "none":
                        if seed != 42:
                            continue
                        result.append(make_job(alias, 0, "none", None))
                    else:
                        result.append(make_job(alias, rank, placement, seed))
    return result


def make_job(alias, rank, placement, seed, smoke=False):
    name = f"{alias}_{placement}_hr{rank}" + (f"_sd{seed}" if seed is not None else "_base")
    if smoke:
        name = "smoke_" + name
    return {"name": name, "model": alias, "hidden_rank": rank,
            "placement": placement, "seed": seed, "smoke": smoke}


def variant(job):
    if job["placement"] == "none":
        return "hidden_lora"
    value = "affine_input" if job["placement"] == "input" else "affine_lm_head"
    return value + ("_plus_hidden_lora" if job["hidden_rank"] else "")


def run_dir(job):
    return HERE / "checkpoints" / job["name"]


def train_command(job):
    big = job["model"] == "qwen25_7b"
    cmd = [PYTHON, "-u", str(HERE / "train.py"),
           "--model-path", str(MODELS[job["model"]]),
           "--train-data", str(HERE / "data/train.jsonl"),
           "--output-dir", str(run_dir(job)), "--variant", variant(job),
           "--hidden-lora-rank", str(job["hidden_rank"]),
           "--hidden-lora-alpha", str(2 * job["hidden_rank"]),
           "--hidden-lora-dropout", "0.05", "--affine-rank", "16", "--affine-alpha", "128",
           "--affine-dropout", "0", "--max-seq-len", "1024",
           "--per-device-train-batch-size", "4" if big else "8",
           "--gradient-accumulation-steps", "4" if big else "2",
           "--learning-rate", "2e-4", "--num-train-epochs", "1",
           "--lr-scheduler-type", "cosine", "--warmup-ratio", "0.03",
           "--max-grad-norm", "1", "--logging-steps", "10",
           "--save-strategy", "no", "--master-dtype", "fp32", "--base-dtype", "bf16",
           "--bf16", "--seed", str(job["seed"])]
    if big:
        cmd.append("--gradient-checkpointing")
    if job["smoke"]:
        cmd += ["--max-train-samples", "32", "--max-steps", "2", "--logging-steps", "1"]
    return cmd


def environment(gpu, job):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               DS_IGNORE_CUDA_DETECTION="1", PYTHONUNBUFFERED="1",
               HF_HOME=str(HERE / "cache/home"),
               HF_DATASETS_CACHE=str(HERE / "cache" / job["model"] / f"gpu{gpu}"),
               XDG_CACHE_HOME=str(HERE / "cache/xdg"))
    return env


def update(job, **fields):
    with LOCK:
        STATE["jobs"].setdefault(job["name"], {}).update(fields)
        STATE["updated_at"] = now()
        write_json(HERE / "state.json", STATE)
        with (HERE / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": now(), "job": job["name"], **fields}) + "\n")
        print(now(), job["name"], fields, flush=True)


def execute(job, gpu, stage, command):
    logfile = HERE / "logs" / f"{job['name']}.{stage}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    update(job, status=stage, gpu=gpu, command=command, log=str(logfile))
    with logfile.open("a") as stream:
        stream.write("\n" + now() + " " + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment(gpu, job),
                                   stdout=stream, stderr=subprocess.STDOUT)
        update(job, pid=process.pid)
        code = process.wait()
    if code:
        raise RuntimeError(f"{stage} exited {code}; see {logfile}")


def validate_train(job):
    path = run_dir(job)
    required = ["run_args.json", "train_results.json", "initialization_audit.json", "trainable_summary.json"]
    if job["hidden_rank"]:
        required += ["adapter_config.json", "adapter_model.safetensors"]
    if job["placement"] != "none":
        required += ["affine_vocab_config.json", "affine_vocab_adapter.safetensors"]
    for name in required:
        assert (path / name).is_file() and (path / name).stat().st_size, name
    args = json.loads((path / "run_args.json").read_text())
    assert args["variant"] == variant(job) and args["seed"] == job["seed"]
    assert args["hidden_lora_rank"] == job["hidden_rank"]
    assert args["corrected_data_pipeline"]["implementation_sha256"] == sha(HERE / "source/corrected_sft_experiment/data_pipeline.py")
    metrics = json.loads((path / "train_results.json").read_text())
    assert metrics["global_step"] == (2 if job["smoke"] else 1424), metrics
    assert math.isfinite(metrics["train_loss"])


def validate_report(path, split, smoke=False):
    report = json.loads(path.read_text())
    expected = [json.loads(line)["record_id"] for line in (HERE / "data" / f"{split}.jsonl").read_text().splitlines()]
    if smoke:
        expected = expected[:8]
    per = report["per_example"]
    assert report["num_examples"] == len(expected)
    assert [row["record_id"] for row in per] == expected
    assert all(row["token_count"] > 0 and math.isfinite(row["nll_sum"]) for row in per)
    nll = sum(row["nll_sum"] for row in per)
    tokens = sum(row["token_count"] for row in per)
    assert tokens == report["supervised_tokens"]
    assert math.isclose(nll / tokens, report["avg_ce"], rel_tol=1e-12)


def work(job, gpu):
    if job["seed"] is not None:
        marker = run_dir(job) / "TRAIN_COMPLETE.json"
        if marker.exists():
            validate_train(job)
        else:
            execute(job, gpu, "training", train_command(job))
            validate_train(job)
            write_json(marker, {"validated_at": now()})
    for split in (("dev",) if job["smoke"] else ("dev", "test")):
        output = HERE / "reports" / f"{job['name']}.{split}.json"
        if output.exists():
            validate_report(output, split, job["smoke"])
            continue
        cmd = [PYTHON, "-u", str(HERE / "source/corrected_sft_experiment/evaluate_corrected_sft.py"),
               "--model-path", str(MODELS[job["model"]]),
               "--data", str(HERE / "data" / f"{split}.jsonl"),
               "--output", str(output), "--batch-size", "1", "--max-seq-len", "1024"]
        if job["seed"] is not None:
            cmd += ["--run-dir", str(run_dir(job))]
        if job["smoke"]:
            cmd += ["--end-index", "8"]
        execute(job, gpu, "eval_" + split, cmd)
        validate_report(output, split, job["smoke"])
    update(job, status="complete", pid=None, gpu=None, completed_at=now())


def wait_for_gpu(gpu, job):
    announced = False
    while True:
        result = subprocess.run(["nvidia-smi", "--id=" + str(gpu),
                                 "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True)
        used, utilization = map(int, result.stdout.strip().split(","))
        if used < 2048 and utilization < 15:
            return
        if not announced:
            update(job, status="waiting_for_gpu", gpu=gpu, memory_used_mib=used)
            announced = True
        time.sleep(15)


def batch(joblist, gpus):
    pending = queue.Queue()
    for job in joblist:
        pending.put(job)
    failures = []

    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                # Each worker owns its GPU until the training/evaluation chain ends.
                wait_for_gpu(gpu, job)
                work(job, gpu)
            except Exception as exc:
                with LOCK:
                    failures.append(job["name"])
                update(job, status="failed", pid=None, gpu=None, error=str(exc))
            finally:
                with LOCK:
                    subprocess.run([PYTHON, str(HERE / "summarize.py")], check=False)
                pending.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "smoke"))
    parser.add_argument("--gpus", default="1,2,3,4")
    args = parser.parse_args()
    manifest = prepare()
    all_jobs = jobs()
    if args.mode == "prepare":
        print(json.dumps({"models": manifest["models"], "jobs": len(all_jobs),
                          "train_runs": sum(j["seed"] is not None for j in all_jobs)}, indent=2))
        return
    lockfile = (HERE / "scheduler.lock").open("w")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    global STATE
    STATE = json.loads((HERE / "state.json").read_text()) if (HERE / "state.json").exists() else {"jobs": {}}
    STATE.update(started_at=now(), scheduler_pid=os.getpid(), phase="smoke", matrix=all_jobs)
    write_json(HERE / "state.json", STATE)
    failures = batch(jobs(smoke=True), args.gpus.split(","))
    if failures:
        STATE["phase"] = "smoke_failed"
    elif args.mode == "smoke":
        STATE["phase"] = "smoke_complete"
    else:
        STATE["phase"] = "matrix"
        write_json(HERE / "state.json", STATE)
        failures = batch(all_jobs, args.gpus.split(","))
        STATE["phase"] = "complete_with_failures" if failures else "complete"
    STATE["finished_at"] = now()
    write_json(HERE / "state.json", STATE)
    subprocess.run([PYTHON, str(HERE / "summarize.py")], check=True)
    if failures:
        raise SystemExit("Failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
