#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"
LOG = HERE / "monitor.jsonl"
SCOPE = str(HERE)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def scoped_processes() -> list[dict[str, str]]:
    result = subprocess.run(["ps", "-eo", "pid=,etime=,%cpu=,%mem=,args="], text=True, capture_output=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        if SCOPE in line and "monitor.py" not in line:
            parts = line.strip().split(None, 4)
            if len(parts) == 5:
                rows.append(dict(zip(("pid", "etime", "cpu", "mem", "args"), parts)))
    return rows


def gpu_snapshot() -> list[dict[str, str]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, memory, utilization, temperature = [item.strip() for item in line.split(",")]
        rows.append({"index": index, "memory_mib": memory, "utilization_pct": utilization, "temperature_c": temperature})
    return rows


def log(row: dict) -> None:
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def terminate_orphans(processes: list[dict[str, str]]) -> None:
    pids = [int(row["pid"]) for row in processes]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(10)
    for pid in pids:
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrator-pid", type=int, required=True)
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()
    next_snapshot = 0.0
    while True:
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"phase": "not_started", "jobs": {}}
        processes = scoped_processes()
        current = time.monotonic()
        if current >= next_snapshot:
            snapshot = {
                "time": now(), "kind": "hourly_health", "orchestrator_pid": args.orchestrator_pid,
                "orchestrator_alive": alive(args.orchestrator_pid), "phase": state.get("phase"),
                "job_statuses": {name: row.get("status") for name, row in state.get("jobs", {}).items()},
                "scoped_processes": processes, "gpus": gpu_snapshot(),
            }
            log(snapshot)
            if not processes:
                log({"time": now(), "kind": "alert_no_scoped_worker", "phase": state.get("phase")})
            next_snapshot = current + args.interval
        if state.get("phase") == "complete":
            log({"time": now(), "kind": "monitor_complete"})
            return
        if not alive(args.orchestrator_pid):
            log({"time": now(), "kind": "alert_orchestrator_dead", "orphan_count": len(processes)})
            terminate_orphans(processes)
            return
        # Check orchestrator liveness at least once a minute while writing full
        # GPU/process snapshots at the requested hourly cadence.
        time.sleep(min(60.0, max(1.0, next_snapshot - time.monotonic())))


if __name__ == "__main__":
    main()
