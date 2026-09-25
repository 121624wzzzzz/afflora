"""Operational poller; logs progress every minute until the scheduler finishes."""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import time

import monitor
import support

HERE = Path(__file__).resolve().parent


def main():
    handle = (HERE / "watch.lock").open("w")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    terminal = {"complete", "complete_with_failures", "smoke_failed"}
    while True:
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                monitor.main()
            report = json.loads(output.getvalue())
            state = monitor.read(HERE / "state.json")
            pid = state.get("scheduler_pid")
            report["watch_pid"] = os.getpid()
            report["scheduler_alive"] = bool(pid and Path(f"/proc/{pid}").exists())
            if not report["scheduler_alive"] and report["phase"] not in terminal:
                report["warnings"].append("Main scheduler process is absent; intervention required")
            support.write_json(HERE / "monitor_latest.json", report)
            print(json.dumps({k: report[k] for k in
                  ("time", "phase", "reuse_completed", "smoke", "completed", "counts", "warnings")}), flush=True)
            if report["warnings"]:
                with (HERE / "monitor_alerts.jsonl").open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
            if report["phase"] in terminal:
                # The scheduler writes its terminal phase just before its final audit.
                if report["phase"] == "complete" and report["scheduler_alive"] and not (HERE / "FINAL_AUDIT.json").exists():
                    time.sleep(10)
                    continue
                if report["phase"] == "complete" and not (HERE / "FINAL_AUDIT.json").exists():
                    report["warnings"].append("Matrix ended without FINAL_AUDIT.json")
                    support.write_json(HERE / "monitor_latest.json", report)
                return
        except Exception as exc:
            print(json.dumps({"time": support.now(), "monitor_error": repr(exc)}), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
