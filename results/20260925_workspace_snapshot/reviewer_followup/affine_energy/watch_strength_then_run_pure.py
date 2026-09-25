#!/usr/bin/env python
"""Hourly watcher: summarize the strength sweep, select a setting, run pure A-LoRA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reviewer_followup/affine_energy"
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
SWEEP_STATE = HERE / "energy_strength_sweep_state.json"
WATCH_STATE = HERE / "pure_followup_watcher_state.json"
WATCH_EVENTS = HERE / "pure_followup_watcher_events.jsonl"
SELECTION = HERE / "PURE_ALORA_SELECTION.md"
PURE_RUNNER = HERE / "run_pure_alora_energy_followup.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_state(phase: str, **payload: Any) -> None:
    WATCH_STATE.write_text(
        json.dumps({"phase": phase, "updated_at": now(), **payload}, indent=2),
        encoding="utf-8",
    )


def event(kind: str, **payload: Any) -> None:
    with WATCH_EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), "kind": kind, **payload}) + "\n")


def accuracy(path: Path, key: str) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))[key]["accuracy_pct"])


def constrained_candidates() -> list[dict[str, Any]]:
    math_files = list((HERE / "strength_sweep_outputs/math").glob("*_full.json"))
    existing = HERE / "outputs/math/llama31_8b_lmhead_ar16_s1_hr4_energy_tau0p00729_l100_seed42_full.json"
    if existing.is_file():
        math_files.append(existing)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for math_path in math_files:
        math_report = json.loads(math_path.read_text(encoding="utf-8"))
        run_dir = Path(math_report["run_dir"])
        args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
        tau = float(args.get("affine_energy_tau", 0.0))
        lam = float(args.get("affine_energy_lambda", 0.0))
        if lam <= 0 or (tau, lam) in seen:
            continue
        if "strength_sweep_outputs" in str(math_path):
            gsm_path = HERE / "strength_sweep_outputs/gsm8k" / math_path.name
        else:
            gsm_path = HERE / "outputs/gsm8k" / math_path.name
        if not gsm_path.is_file():
            raise FileNotFoundError(gsm_path)
        math_acc = float(math_report["clean"]["accuracy_pct"])
        gsm_acc = accuracy(gsm_path, "full")
        rows.append({
            "tau": tau,
            "lambda": lam,
            "math": math_acc,
            "gsm8k": gsm_acc,
            "score": (math_acc + gsm_acc) / 2.0,
            "name": run_dir.name,
        })
        seen.add((tau, lam))
    return rows


def select_and_document() -> dict[str, Any]:
    rows = constrained_candidates()
    if not rows:
        raise RuntimeError("No completed constrained sweep candidates")
    # Predeclared selection rule: equal weight to one percentage point on MATH
    # and GSM8K.  Exact ties prefer the less stiff (smaller-lambda) constraint.
    selected = max(rows, key=lambda row: (row["score"], -row["lambda"], row["tau"]))
    ordered = sorted(rows, key=lambda row: row["score"], reverse=True)
    lines = [
        "# Constraint selected for the pure A-LoRA follow-up", "",
        f"Updated: {now()}", "",
        "Selection rule: maximize the equally weighted mean of full MATH-clean and GSM8K accuracy. ",
        "Exact ties prefer the smaller lambda. All candidates are seed 42.", "",
        "| selected | tau | lambda | MATH | GSM8K | mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        mark = "yes" if row is selected else ""
        lines.append(
            f"| {mark} | {row['tau']:.8f} | {row['lambda']:g} | "
            f"{row['math']:.4f}% | {row['gsm8k']:.4f}% | {row['score']:.4f}% |"
        )
    lines.extend([
        "", f"Chosen tau: {selected['tau']:.8f}",
        f"Chosen lambda: {selected['lambda']:g}",
    ])
    SELECTION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=3600)
    args = parser.parse_args()
    if WATCH_EVENTS.exists():
        WATCH_EVENTS.unlink()
    while True:
        sweep: dict[str, Any] = {}
        if SWEEP_STATE.is_file():
            sweep = json.loads(SWEEP_STATE.read_text(encoding="utf-8"))
        phase = sweep.get("phase", "not_started")
        event("sweep_check", sweep_phase=phase)
        if phase == "complete":
            break
        write_state(
            "waiting_for_strength_sweep", sweep_phase=phase,
            next_check_after_seconds=args.interval_seconds,
        )
        time.sleep(args.interval_seconds)

    write_state("selecting_constraint")
    selected = select_and_document()
    event("constraint_selected", **selected)
    write_state(
        "running_pure_alora", selected_tau=selected["tau"],
        selected_lambda=selected["lambda"], selection=str(SELECTION),
    )
    cmd = [
        str(PYTHON), str(PURE_RUNNER), "--tau", str(selected["tau"]),
        "--energy-lambda", str(selected["lambda"]),
    ]
    rc = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy()).returncode
    if rc != 0:
        write_state("failed", returncode=rc, selected=selected)
        raise RuntimeError(f"Pure A-LoRA runner failed with return code {rc}")
    write_state(
        "complete", selected=selected,
        selection=str(SELECTION), result=str(HERE / "PURE_ALORA_ENERGY_RESULTS.md"),
    )
    event("followup_complete", selected=selected)


if __name__ == "__main__":
    main()
