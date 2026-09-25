"""Eight-GPU stacking matrix, gated by full checkpoint reuse verification."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import support
import summarize

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "placement_capacity_corrected_20260911"
LOCK = support.LOCK
STATE = {}
STOP = threading.Event()
ARMS = ("none", "output", "both", "hidden_budget")
ORIGINAL_ENV = support.environment


def jobs(smoke=False):
    result = []
    for seed in ((42,) if smoke else (42,43,44)):
        for rank in ((64,) if smoke else (8,32,64)):
            for model in support.MODELS:
                for arm in ARMS:
                    job = support.make_job(model,rank,arm,seed,smoke)
                    reuse = not smoke and rank == 8 and arm in ("none","output")
                    job["reuse"] = reuse
                    root = OLD if reuse else HERE
                    job["checkpoint"] = str(root / "checkpoints" / job["name"])
                    result.append(job)
    return result


def variant(job):
    return {"none":"hidden_lora", "hidden_budget":"hidden_lora",
            "output":"affine_lm_head_plus_hidden_lora",
            "both":"affine_input_lm_head_plus_hidden_lora"}[job["placement"]]


def run_dir(job): return Path(job["checkpoint"])


def environment(gpu,job):
    env = ORIGINAL_ENV(gpu,job)
    env["STACKING_ARM"] = job["placement"]
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


support.variant = variant
support.run_dir = run_dir
support.environment = environment


def checkpoint_hashes(cp):
    return {name: support.sha(cp/name) for name in
            ("run_args.json","adapter_config.json","adapter_model.safetensors",
             "affine_vocab_config.json","affine_vocab_adapter.safetensors") if (cp/name).is_file()}


def prepare():
    path = HERE / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        for rel, expected in manifest["sha256"].items():
            assert support.sha(HERE/rel) == expected, f"Frozen file changed: {rel}"
    else:
        assert json.loads((HERE/"budget_check.json").read_text())["status"] == "passed"
        old_manifest = json.loads((OLD/"manifest.json").read_text())
        for rel, expected in old_manifest["sha256"].items():
            if rel.startswith(("source/","data/")):
                assert support.sha(HERE/rel) == expected, rel
        files = []
        for sub in ("source","data","ifeval_deps","nltk_data"):
            files.extend(x for x in (HERE/sub).rglob("*") if x.is_file()
                         and "__pycache__" not in x.parts and x.suffix != ".pyc")
        files.extend(x for x in HERE.iterdir() if x.is_file() and x.suffix in (".py",".sh",".md"))
        files.append(HERE/"budget_check.json")
        preflight = json.loads((HERE/"reuse_audit/preflight.json").read_text())
        assert preflight["status"] == "passed" and preflight["checkpoint_count"] == 12
        manifest = {"created_at":support.now(), "matrix":jobs(), "new_training_runs":60,
                    "reused_checkpoints":12, "smoke_jobs":8,
                    "sha256":{str(x.relative_to(HERE)):support.sha(x) for x in sorted(set(files))},
                    "models":old_manifest["models"], "reuse_records":preflight["records"],
                    "versions":{name:importlib.metadata.version(name) for name in
                                ("torch","transformers","peft","datasets","safetensors")}}
        manifest["tokenizer_files"] = {
            str(x):support.sha(x) for info in manifest["models"].values()
            for x in Path(info["path"]).iterdir() if x.is_file() and x.suffix in (".json",".txt",".model")}
        support.write_json(path,manifest)
    for name,version in manifest["versions"].items():
        assert importlib.metadata.version(name) == version, name
    for path,expected in manifest["tokenizer_files"].items():
        assert support.sha(path) == expected, path
    for info in manifest["models"].values():
        base = Path(info["path"])
        assert support.sha(base/"config.json") == info["config_sha256"]
        for name,expected in info["weight_files"].items():
            stat = (base/name).stat()
            assert (stat.st_size,stat.st_mtime_ns) == (expected["bytes"],expected["mtime_ns"]), name
    for record in manifest["reuse_records"].values():
        for path,expected in record["files"].items():
            assert support.sha(path) == expected, f"Reused file changed: {path}"
    return manifest


def validate_train(job):
    # P0 infers boundary files from placement; hidden_budget has no boundary.
    support.validate_train({**job, "placement":"none"} if job["placement"] == "hidden_budget" else job)
    cp = run_dir(job)
    metrics = json.loads((cp/"train_results.json").read_text())
    audit = json.loads((cp/"initialization_audit.json").read_text())
    assert audit["train_rows"] == (32 if job["smoke"] else 22780)
    assert audit["train_batch"] * audit["gradient_accumulation"] == 16
    if not job["smoke"]: assert metrics["epoch"] == 1.
    if job["placement"] == "hidden_budget":
        cfg = json.loads((cp/"adapter_config.json").read_text())
        budget = audit["budget_control"]
        assert cfg["rank_pattern"] == budget["rank_pattern"]
        assert budget["target_extra_parameters"] <= budget["actual_extra_parameters"] <= budget["target_extra_parameters"]*1.003
        assert all(v == job["hidden_rank"]+1 for v in cfg["rank_pattern"].values())
        assert all(v == 2*(job["hidden_rank"]+1) for v in cfg["alpha_pattern"].values())


def validate_smoke_pairing():
    for model in support.MODELS:
        audits = [json.loads((run_dir(job)/"initialization_audit.json").read_text())
                  for job in jobs(True) if job["model"] == model]
        assert len({a["shared_hidden_init_sha256"] for a in audits}) == 1, model
        boundary = {value for a in audits for value in a["affine_components"].values()}
        assert len(boundary) == 1, model


def validate_report(job,split):
    path = HERE/"reports"/f"{job['name']}.{split}.json"
    support.validate_report(path,split,job["smoke"])
    report = json.loads(path.read_text())
    assert report["run_dir"] == job["checkpoint"]
    assert report["variant"] == variant(job) and report["seed"] == job["seed"]
    assert report["affine_ablation"] == "none"
    assert report["data"] == str(HERE/"data"/f"{split}.jsonl")


def validate_generation(job):
    folder = HERE/"ifeval"/job["name"]
    result = json.loads((folder/"scores.json").read_text())
    identity = json.loads((folder/"generation_metadata.json").read_text())
    assert result["identity"] == identity
    assert identity["checkpoint"] == job["checkpoint"]
    assert identity["checkpoint_files"] == checkpoint_hashes(run_dir(job))
    assert identity["protocol_sha256"] == support.sha(HERE/"manifest.json")
    assert identity["data_sha256"] == support.sha(HERE/"data/ifeval.jsonl")
    assert identity["batch_size"] == 8 and identity["max_new_tokens"] == (32 if job["smoke"] else 512)
    assert identity["do_sample"] is False and identity["enable_thinking"] is False
    assert identity["python_scoring_seed"] == 0 and identity["sensitivity_excluded_keys"] == [1122,1129]
    assert identity["stop_policy"] == "base_eos_plus_native_im_end"
    assert identity["eos_token_ids"] == [151643,151645]
    count = 8 if job["smoke"] else 541
    assert result["examples"] == identity["count"] == count
    rows = [json.loads(line) for line in (HERE/"data/ifeval.jsonl").read_text().splitlines()][:count]
    responses = [json.loads(line) for line in (folder/"responses.jsonl").read_text().splitlines()]
    assert len(responses) == len(result["per_example"]) == count
    assert support.sha(folder/"responses.jsonl") == result["responses_sha256"]
    for row,response,scored in zip(rows,responses,result["per_example"]):
        assert all(row[k] == response[k] for k in ("key","prompt","instruction_id_list","kwargs"))
        assert row["key"] == scored["key"]
        assert 1 <= response["generated_tokens"] <= identity["max_new_tokens"]
        if response["hit_token_cap"]:
            assert response["generated_tokens"] == identity["max_new_tokens"] and response["terminal_token_id"] is None
        else:
            assert response["terminal_token_id"] in identity["eos_token_ids"]
        for mode in ("strict","loose"):
            flags = scored[mode+"_instructions"]
            assert len(flags) == len(row["instruction_id_list"])
            assert all(type(flag) is bool for flag in flags)
            assert scored[mode] == all(flags)
    for mode in ("strict","loose"):
        expected = sum(row[mode] for row in result["per_example"])/count
        assert math.isclose(result[mode+"_prompt_accuracy"],expected,rel_tol=0,abs_tol=1e-12)
    valid = [row for row in result["per_example"] if row["key"] not in (1122,1129)]
    sensitivity = result["valid_prompt_sensitivity"]
    assert sensitivity["examples"] == len(valid)
    assert sensitivity["excluded_keys"] == [row["key"] for row in result["per_example"] if row["key"] in (1122,1129)]
    assert all(not row["strict"] or row["loose"] for row in valid)
    for mode in ("strict","loose"):
        assert math.isclose(sensitivity[mode+"_prompt_accuracy"],sum(row[mode] for row in valid)/len(valid),rel_tol=0,abs_tol=1e-12)


def work(job,gpu):
    cp = run_dir(job)
    marker = cp/"TRAIN_COMPLETE.json"
    if job["reuse"]:
        approved = json.loads((HERE/"reuse_audit/APPROVED.json").read_text())
        assert approved["status"] == "passed" and job["name"] in approved["comparisons"]
        validate_train(job)
    elif marker.exists():
        validate_train(job)
        assert json.loads(marker.read_text())["checkpoint_hashes"] == checkpoint_hashes(cp)
    else:
        support.execute(job,gpu,"training",support.train_command(job))
        validate_train(job)
        support.write_json(marker,{"validated_at":support.now(),"checkpoint_hashes":checkpoint_hashes(cp)})
    for split in (("dev",) if job["smoke"] else ("dev","test")):
        path = HERE/"reports"/f"{job['name']}.{split}.json"
        if job["reuse"]:
            source = HERE/"reuse_audit/reports"/path.name
            expected = approved["comparisons"][job["name"]][split]["fresh_report_sha256"]
            assert support.sha(source) == expected
            if not path.exists(): shutil.copy2(source,path)
            assert support.sha(path) == expected
        elif not path.exists():
            command = [support.PYTHON,"-u",str(HERE/"source/corrected_sft_experiment/evaluate_corrected_sft.py"),
                       "--run-dir",str(cp),"--data",str(HERE/"data"/f"{split}.jsonl"),
                       "--output",str(path),"--batch-size","1","--max-seq-len","1024"]
            if job["smoke"]: command += ["--end-index","8"]
            support.execute(job,gpu,"eval_"+split,command)
        validate_report(job,split)
    folder = HERE/"ifeval"/job["name"]
    if not (folder/"scores.json").exists():
        command = [support.PYTHON,"-u",str(HERE/"evaluate_generation.py"),
                   "--run-dir",str(cp),"--output-dir",str(folder),"--batch-size","8"]
        if job["smoke"]: command += ["--limit","8","--max-new-tokens","32"]
        support.execute(job,gpu,"ifeval",command)
    validate_generation(job)
    support.update(job,status="complete",pid=None,gpu=None,completed_at=support.now())


def batch(joblist,gpus):
    pending = deque(joblist)
    failures = []

    def worker(gpu):
        while not STOP.is_set():
            with LOCK:
                if not pending: return
            raw = subprocess.check_output(["nvidia-smi","--id="+gpu,
                      "--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
            used,util = map(int,raw.strip().split(","))
            if used >= 3072 or util >= 15:
                time.sleep(10)
                continue
            with LOCK:
                if not pending: return
                job = pending.popleft()
            try:
                work(job,gpu)
            except Exception as exc:
                failures.append(job["name"])
                support.update(job,status="failed",error=str(exc),pid=None,gpu=None)
            with LOCK:
                try:
                    summarize.summarize()
                except Exception as exc:
                    STATE.setdefault("audit_errors",[]).append(str(exc))
                    support.write_json(HERE/"state.json",STATE)
                    STOP.set()
    workers = [threading.Thread(target=worker,args=(gpu,)) for gpu in gpus]
    for t in workers: t.start()
    for t in workers: t.join()
    return failures


def verify_all():
    manifest = prepare()
    for job in jobs():
        assert STATE["jobs"][job["name"]]["status"] == "complete",job["name"]
        validate_train(job)
        for split in ("dev","test"): validate_report(job,split)
        validate_generation(job)
    summary = summarize.summarize()
    assert summary["completed"] == 72 and not summary["audit_errors"]
    support.write_json(HERE/"FINAL_AUDIT.json",{"status":"passed","checked_at":support.now(),
        "cells":72,"new_training_runs":60,"reloaded_reuse_checkpoints":12,
        "ce_reports":144,"ifeval_reports":72,"ifeval_prompts_per_report":541,
        "frozen_files":len(manifest["sha256"]),"audit_errors":[]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run","check"))
    parser.add_argument("--gpus",default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    for name in ("checkpoints","reports","logs","ifeval"):
        (HERE/name).mkdir(exist_ok=True)
    manifest = prepare()
    if args.mode == "prepare":
        print(json.dumps({"cells":72,"new_training_runs":60,"reused_checkpoints":12,
                          "frozen_files":len(manifest["sha256"])}));return
    handle = (HERE/"scheduler.lock").open("w")
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    global STATE
    STATE = json.loads((HERE/"state.json").read_text()) if (HERE/"state.json").exists() else {"jobs":{}}
    support.STATE = STATE
    STATE.update(matrix=jobs(),scheduler_pid=os.getpid())
    if args.mode == "check": verify_all();return
    STATE.update(phase="awaiting_reuse_verification",started_at=support.now())
    support.write_json(HERE/"state.json",STATE)
    while not (HERE/"reuse_audit/APPROVED.json").exists():
        auditstate = HERE/"reuse_audit/state.json"
        if auditstate.exists() and json.loads(auditstate.read_text()).get("phase") == "failed":
            raise RuntimeError("Reuse verification failed; cannot start matrix")
        time.sleep(10)
    approved = json.loads((HERE/"reuse_audit/APPROVED.json").read_text())
    assert approved["status"] == "passed" and approved["completed"] == 12
    prepare()
    STATE.update(phase="smoke",reuse_approval_sha256=support.sha(HERE/"reuse_audit/APPROVED.json"))
    support.write_json(HERE/"state.json",STATE)
    failures = batch(jobs(True),args.gpus.split(","))
    if failures or STOP.is_set():
        STATE["phase"] = "smoke_failed"
    else:
        validate_smoke_pairing()
        STATE["phase"] = "matrix"
        support.write_json(HERE/"state.json",STATE)
        failures = batch(jobs(),args.gpus.split(","))
        STATE["phase"] = "complete_with_failures" if failures or STOP.is_set() else "complete"
    STATE.update(finished_at=support.now(),failures=failures)
    support.write_json(HERE/"state.json",STATE)
    if STATE["phase"] == "complete": verify_all()
    else: raise RuntimeError(f"Experiment ended with {STATE['phase']}: {failures}")


if __name__ == "__main__": main()
