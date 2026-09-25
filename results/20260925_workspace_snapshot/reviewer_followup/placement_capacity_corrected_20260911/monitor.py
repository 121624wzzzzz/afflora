#!/usr/bin/env python3
"""Read-only experiment inspection with an append-only polling history."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent


def tail(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        return stream.read().decode(errors="replace")


def main():
    state = json.loads((HERE / "state.json").read_text())
    active = []
    warnings = []
    jobs = {name: value for name, value in state["jobs"].items() if not name.startswith("smoke_")}
    summary = json.loads((HERE / "summary.json").read_text())
    warnings.extend(summary.get("audit_errors", []))
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    gpus = {}
    for line in gpu_result.stdout.splitlines():
        index, memory, utilization = map(int, line.split(","))
        if index in (1, 2, 3, 4):
            gpus[str(index)] = {"memory_mib": memory, "utilization": utilization}
    for name, job in jobs.items():
        status = job["status"]
        if status == "complete":
            continue
        row = {"name": name, "status": status, "gpu": job.get("gpu"), "pid": job.get("pid")}
        if status == "failed":
            warnings.append(f"{name}: {job.get('error')}")
        path = Path(job["log"]) if job.get("log") else None
        if path and path.exists():
            text = tail(path)
            row["log_age_s"] = round(time.time() - path.stat().st_mtime)
            if status == "training":
                steps = re.findall(r"(\d+)/1424", text)
                if steps:
                    row["step"] = int(steps[-1])
                    row["percent"] = round(100 * row["step"] / 1424, 1)
                losses = re.findall(r"\{[^\n{}]*'loss':[^\n{}]*\}", text)
                if losses:
                    values = ast.literal_eval(losses[-1])
                    row["loss"] = float(values["loss"])
                    if not math.isfinite(row["loss"]):
                        warnings.append(f"{name}: nonfinite training loss")
                    if "grad_norm" in values:
                        row["grad_norm"] = float(values["grad_norm"])
                        if not math.isfinite(row["grad_norm"]):
                            warnings.append(f"{name}: nonfinite gradient norm")
                if row["log_age_s"] > 240:
                    warnings.append(f"{name}: training log unchanged for {row['log_age_s']}s")
            elif status.startswith("eval_") and row["log_age_s"] > 1800:
                warnings.append(f"{name}: evaluation has no log update for 30 minutes; inspect process/GPU")
            if "Traceback (most recent call last)" in text:
                warnings.append(f"{name}: traceback in current log (may include an earlier attempt)")
        pid = job.get("pid")
        if pid and status in ("training", "eval_dev", "eval_test"):
            row["process_alive"] = Path(f"/proc/{pid}").exists()
            if not row["process_alive"]:
                warnings.append(f"{name}: PID {pid} missing; recheck transition")
        active.append(row)
    counts = Counter(job["status"] for job in jobs.values())
    report = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "phase": state["phase"], "completed": counts.get("complete", 0), "total": 50,
        "pending": 50 - len(jobs), "counts": dict(counts), "active": active,
        "gpus": gpus, "warnings": warnings,
    }
    with (HERE / "monitor_history.jsonl").open("a") as stream:
        stream.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
