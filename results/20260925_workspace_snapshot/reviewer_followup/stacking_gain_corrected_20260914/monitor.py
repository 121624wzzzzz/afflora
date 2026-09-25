"""Inspect reuse verification, smoke tests, and the stacking matrix."""
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


def read(path): return json.loads(path.read_text()) if path.exists() else {}


def main():
    state = read(HERE/"state.json")
    reuse = read(HERE/"reuse_audit/state.json")
    source = reuse if state.get("phase", "awaiting_reuse_verification") == "awaiting_reuse_verification" else state
    active, warnings = [], list(state.get("audit_errors",[]))
    warnings += read(HERE/"summary.json").get("audit_errors",[])
    for name,job in source.get("jobs",{}).items():
        if job["status"] == "complete": continue
        item = {"name":name,"status":job["status"],"gpu":job.get("gpu"),"pid":job.get("pid")}
        if job["status"] == "failed": warnings.append(f"{name}: {job.get('error')}")
        log = Path(job["log"]) if job.get("log") else None
        if log and log.exists():
            age = time.time()-log.stat().st_mtime
            item["log_age_s"] = round(age)
            with log.open("rb") as stream:
                stream.seek(max(0,log.stat().st_size-65536))
                tail = stream.read().decode(errors="replace")
            if job["status"] == "training":
                steps = re.findall(r"(\d+)/1424",tail)
                if steps:
                    item["step"] = int(steps[-1]);item["percent"] = round(100*item["step"]/1424,1)
                losses = re.findall(r"\{[^\n{}]*'loss':[^\n{}]*\}",tail)
                if losses:
                    try:
                        values = ast.literal_eval(losses[-1])
                        for field in ("loss","grad_norm"):
                            if field in values:
                                item[field] = float(values[field])
                                if not math.isfinite(item[field]):warnings.append(f"{name}: nonfinite {field}")
                    except (ValueError,SyntaxError): warnings.append(f"{name}: inspect training metrics {losses[-1]}")
                if age>240:warnings.append(f"{name}: training log stale for {int(age)} seconds")
            if job["status"] == "ifeval":
                generations = re.findall(r"generated (\d+)/(\d+)",tail)
                if generations:item["generated"] = "/".join(generations[-1])
                if age>1200:warnings.append(f"{name}: IFEval log stale for 20 minutes")
            if job["status"].startswith("eval_") and age>1800:
                warnings.append(f"{name}: evaluation log stale for 30 minutes")
            if "Traceback (most recent call last)" in tail:warnings.append(f"{name}: traceback in log")
        if job.get("pid"):
            item["process_alive"] = Path(f"/proc/{job['pid']}").exists()
            if not item["process_alive"] and item.get("log_age_s",0)>30:
                warnings.append(f"{name}: process missing; verify stage transition")
        active.append(item)
    raw = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    gpus = {}
    for line in raw.splitlines():
        idx,mem,util = map(int,line.split(","));gpus[str(idx)] = {"memory_mib":mem,"utilization":util}
    counts = Counter(j["status"] for name,j in state.get("jobs",{}).items() if not name.startswith("smoke_"))
    smoke = Counter(j["status"] for name,j in state.get("jobs",{}).items() if name.startswith("smoke_"))
    report = {"time":datetime.now().astimezone().isoformat(timespec="seconds"),
              "phase":state.get("phase","awaiting_reuse_verification"),
              "reuse_completed":sum(j.get("status")=="complete" for j in reuse.get("jobs",{}).values()),
              "smoke":dict(smoke),"completed":counts.get("complete",0),"total":72,
              "counts":dict(counts),"active":active,"warnings":warnings,"gpus":gpus}
    with (HERE/"monitor_history.jsonl").open("a") as stream:stream.write(json.dumps(report)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
