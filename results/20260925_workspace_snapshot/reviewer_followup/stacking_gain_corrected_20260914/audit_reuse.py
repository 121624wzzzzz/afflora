"""Gate P0 reuse on metadata/tensor audits AND fresh full dev/test reloads."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
from safetensors import safe_open
import support

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "placement_capacity_corrected_20260911"
OUT = HERE / "reuse_audit"
LOCK = threading.RLock()
STATE = {}
ATOL_CE = 2e-6
ATOL_ROW_CE = 5e-5


def cases():
    return [support.make_job(model, 8, arm, seed) for model in support.MODELS
            for seed in (42, 43, 44) for arm in ("none", "output")]


def old_module():
    spec = importlib.util.spec_from_file_location("verified_p0", OLD / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update(name=None, **fields):
    with LOCK:
        if name is None:
            STATE.update(fields)
        else:
            STATE["jobs"].setdefault(name, {}).update(fields)
        STATE["updated_at"] = support.now()
        support.write_json(OUT / "state.json", STATE)
        print(support.now(), name or "reuse", fields, flush=True)


def preflight():
    p0 = old_module()
    manifest = p0.prepare()
    assert json.loads((OLD / "state.json").read_text())["phase"] == "complete"
    assert json.loads((OLD / "FINAL_AUDIT.json").read_text())["status"] == "passed"
    # The core trainer, adapter and corrected pipeline must be byte-identical.
    for rel in manifest["sha256"]:
        if rel.startswith(("source/", "data/")):
            assert support.sha(HERE / rel) == manifest["sha256"][rel], rel
    for alias, info in manifest["models"].items():
        base = Path(info["path"])
        assert support.sha(base / "config.json") == info["config_sha256"]
        for name, expected in info["weight_files"].items():
            stat = (base / name).stat()
            assert stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], name
    records = {}
    hashes = {}
    expected_args = {"hidden_lora_rank": 8, "hidden_lora_alpha": 16,
        "hidden_lora_dropout": .05, "hidden_lora_target_modules": "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        "hidden_lora_layers_to_transform": None, "affine_rank": 16, "affine_alpha": 128.,
        "affine_dropout": 0., "affine_bias_scale": 1., "affine_learning_rate_scale": 1.,
        "affine_bias_learning_rate_scale": 1., "affine_energy_lambda": 0.,
        "affine_bias_energy_lambda": 0., "reference_kl_lambda": 0.,
        "anchor_fraction": 0., "auxiliary_anchor_lambda": 0.,
        "initial_hidden_lora_adapter": None, "initial_affine_adapter": None,
        "freeze_initial_hidden_lora": False, "include_emb_lmh_lora_rank": 0,
        "max_seq_len": 1024, "learning_rate": 2e-4, "num_train_epochs": 1.,
        "max_steps": -1, "max_train_samples": None, "master_dtype": "fp32",
        "base_dtype": "bf16", "bf16": True, "fp16": False,
        "lr_scheduler_type": "cosine", "warmup_ratio": .03, "max_grad_norm": 1.,
        "save_strategy": "no", "tie_affine_input_lm_head_adapters": False,
        "affine_lm_head_bias": False, "no_affine_input_bias": False}
    for job in cases():
        p0.validate_train(job)
        cp = OLD / "checkpoints" / job["name"]
        assert (cp / "TRAIN_COMPLETE.json").is_file()
        args = json.loads((cp / "run_args.json").read_text())
        for key, expected in expected_args.items():
            assert args[key] == expected, (job["name"], key, args.get(key), expected)
        assert Path(args["model_path"]).resolve() == support.MODELS[job["model"]].resolve()
        assert support.sha(Path(args["train_data"])) == support.sha(HERE / "data/train.jsonl")
        big = job["model"] == "qwen25_7b"
        assert args["per_device_train_batch_size"] == (4 if big else 8)
        assert args["gradient_accumulation_steps"] == (4 if big else 2)
        assert args["gradient_checkpointing"] == big
        metrics = json.loads((cp / "train_results.json").read_text())
        audit = json.loads((cp / "initialization_audit.json").read_text())
        assert metrics["epoch"] == 1 and metrics["global_step"] == 1424
        assert audit["train_rows"] == 22780
        key = (job["model"], job["seed"])
        hashes.setdefault(key, set()).add(audit["hidden_init_sha256"])
        acfg = json.loads((cp / "adapter_config.json").read_text())
        assert acfg["r"] == 8 and acfg["lora_alpha"] == 16
        assert not acfg["rank_pattern"] and not acfg["alpha_pattern"]
        assert set(acfg["target_modules"]) == set(expected_args["hidden_lora_target_modules"].split(","))
        tensors = {}
        for file, group in (("adapter_model.safetensors", "hidden"), ("affine_vocab_adapter.safetensors", "boundary")):
            path = cp / file
            if not path.exists():
                assert group == "boundary" and job["placement"] == "none"
                continue
            count = 0
            with safe_open(str(path), framework="np") as stream:
                for tensor_name in stream.keys():
                    array = stream.get_tensor(tensor_name)
                    assert array.dtype == np.float32 and np.isfinite(array).all(), (path, tensor_name)
                    count += array.size
            assert count == audit["trainable"][group], (path, count, audit["trainable"])
            tensors[group] = count
        for split in ("dev", "test"):
            p0.validate_report(OLD / "reports" / f"{job['name']}.{split}.json", split)
        files = [x for x in cp.iterdir() if x.is_file() and x.suffix in (".json", ".safetensors")]
        files += [OLD / "reports" / f"{job['name']}.{split}.json" for split in ("dev", "test")]
        records[job["name"]] = {"checkpoint": str(cp), "parameters": tensors,
            "hidden_init_sha256": audit["hidden_init_sha256"],
            "files": {str(x): support.sha(x) for x in files}}
    assert all(len(v) == 1 for v in hashes.values())
    report = {"status": "passed", "checked_at": support.now(), "checkpoint_count": 12,
              "source_manifest_sha256": support.sha(OLD / "manifest.json"),
              "aggregate_ce_tolerance": ATOL_CE, "per_example_ce_tolerance": ATOL_ROW_CE,
              "records": records, "model_metadata": manifest["models"]}
    existing = OUT / "preflight.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        assert previous["records"] == report["records"], "Reused files changed after preflight"
    support.write_json(existing, report)
    return report


def compare(job, split):
    fresh_path = OUT / "reports" / f"{job['name']}.{split}.json"
    support.validate_report(fresh_path, split)
    old = json.loads((OLD / "reports" / f"{job['name']}.{split}.json").read_text())
    fresh = json.loads(fresh_path.read_text())
    assert fresh["run_dir"] == str(OLD / "checkpoints" / job["name"])
    assert fresh["affine_ablation"] == "none"
    assert fresh["seed"] == job["seed"] and fresh["variant"] == old["variant"]
    assert fresh["supervised_tokens"] == old["supervised_tokens"]
    diffs = []
    for a, b in zip(old["per_example"], fresh["per_example"]):
        assert (a["record_id"], a["token_count"]) == (b["record_id"], b["token_count"])
        diffs.append(abs(a["nll_sum"] - b["nll_sum"]) / a["token_count"])
    delta = fresh["avg_ce"] - old["avg_ce"]
    assert abs(delta) <= ATOL_CE and max(diffs) <= ATOL_ROW_CE, (job["name"], split, delta, max(diffs))
    return {"status": "passed", "old_ce": old["avg_ce"], "fresh_ce": fresh["avg_ce"],
            "ce_difference": delta, "max_per_example_ce_difference": max(diffs),
            "examples": fresh["num_examples"], "supervised_tokens": fresh["supervised_tokens"],
            "fresh_report_sha256": support.sha(fresh_path)}


def run_case(job, gpu):
    comparisons = {}
    for split in ("dev", "test"):
        path = OUT / "reports" / f"{job['name']}.{split}.json"
        if not path.exists():
            command = [support.PYTHON, "-u", str(HERE / "source/corrected_sft_experiment/evaluate_corrected_sft.py"),
                       "--run-dir", str(OLD / "checkpoints" / job["name"]),
                       "--data", str(HERE / "data" / f"{split}.jsonl"), "--output", str(path),
                       "--batch-size", "1", "--max-seq-len", "1024"]
            log = OUT / "logs" / f"{job['name']}.{split}.log"
            update(job["name"], status="eval_" + split, gpu=gpu, log=str(log))
            with log.open("a") as stream:
                process = subprocess.Popen(command, cwd=support.ROOT, env=support.environment(gpu, job),
                                           stdout=stream, stderr=subprocess.STDOUT)
                update(job["name"], pid=process.pid)
                code = process.wait()
            if code:
                raise RuntimeError(f"Reload evaluation failed: {code}; {log}")
        comparisons[split] = compare(job, split)
    update(job["name"], status="complete", gpu=None, pid=None, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    for folder in (OUT, OUT / "reports", OUT / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    handle = (OUT / "lock").open("w")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight()
    if args.mode == "preflight":
        print("12 reuse candidates passed source/config/tensor/initialization audits.")
        return
    global STATE
    STATE = json.loads((OUT / "state.json").read_text()) if (OUT / "state.json").exists() else {"jobs": {}}
    update(phase="reloading", scheduler_pid=os.getpid(), started_at=support.now())
    pending = deque(cases())
    errors = []

    def worker(gpu):
        while True:
            with LOCK:
                if not pending:
                    return
            raw = subprocess.check_output(["nvidia-smi", "--id="+gpu, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            used, util = map(int, raw.strip().split(","))
            if used >= 3072 or util >= 15:
                time.sleep(10)
                continue
            with LOCK:
                if not pending:
                    return
                job = pending.popleft()
            try:
                run_case(job, gpu)
            except Exception as exc:
                errors.append(str(exc))
                update(job["name"], status="failed", error=str(exc), pid=None, gpu=None)

    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus.split(",")]
    for t in workers: t.start()
    for t in workers: t.join()
    # Recheck file hashes after reload so a changed endpoint cannot be reused.
    preflight()
    complete = sum(j.get("status") == "complete" for j in STATE["jobs"].values())
    update(phase="complete" if complete == 12 and not errors else "failed",
           completed=complete, errors=errors, finished_at=support.now())
    if STATE["phase"] != "complete":
        raise SystemExit("Reuse audit failed; these endpoints must not enter the experiment.")
    support.write_json(OUT / "APPROVED.json", {"status": "passed", "completed": complete,
        "approved_at": support.now(), "preflight_sha256": support.sha(OUT / "preflight.json"),
        "comparisons": {k: v["comparisons"] for k, v in STATE["jobs"].items()}})


if __name__ == "__main__": main()
