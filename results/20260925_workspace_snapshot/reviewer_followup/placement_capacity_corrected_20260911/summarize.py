#!/usr/bin/env python3
"""Paired seed summaries and placement-capacity interactions, with audit gates."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent


def describe(values):
    if not values:
        return None
    result = {"n": len(values), "values": values, "mean": mean(values)}
    if len(values) == 3:
        sd = stdev(values)
        half = 4.302652729911275 * sd / math.sqrt(3)
        result.update(sample_sd=sd, ci95=[result["mean"] - half, result["mean"] + half])
    return result


def show(value):
    if value is None:
        return "pending"
    text = f"{value['mean']:+.6f} (n={value['n']})"
    if "ci95" in value:
        text += f" [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}]"
    return text


def summarize():
    statefile = HERE / "state.json"
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    matrix = state.get("matrix", [])
    audits = {}
    observations = {}
    errors = []
    for job in matrix:
        if state.get("jobs", {}).get(job["name"], {}).get("status") != "complete":
            continue
        path = HERE / "reports" / f"{job['name']}.test.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        key = (job["model"], job["hidden_rank"], job["placement"], job["seed"])
        observations[key] = report
        auditpath = HERE / "checkpoints" / job["name"] / "initialization_audit.json"
        if auditpath.exists():
            audits[key] = json.loads(auditpath.read_text())
    for alias in ("qwen3_06b", "qwen25_7b"):
        reports = [report for key, report in observations.items() if key[0] == alias]
        identities = {tuple((row["record_id"], row["token_count"]) for row in report["per_example"]) for report in reports}
        if len(identities) > 1:
            errors.append(f"Unpaired evaluation examples/tokens: {alias}")
        for seed in (42, 43, 44):
            for rank in (1, 8):
                hashes = {audit["hidden_init_sha256"] for key, audit in audits.items() if key[0] == alias and key[1] == rank and key[3] == seed}
                if len(hashes) > 1:
                    errors.append(f"Hidden initialization mismatch: {alias} r{rank} seed{seed}")
            hashes = {audit["affine_init_sha256"] for key, audit in audits.items() if key[0] == alias and key[2] != "none" and key[3] == seed}
            if len(hashes) > 1:
                errors.append(f"Affine initialization mismatch: {alias} seed{seed}")
    result = {"phase": state.get("phase", "prepared"), "audit_errors": errors, "models": {}}
    complete = len(observations)
    lines = ["# Corrected-SFT placement × hidden capacity", "",
             f"Phase: `{result['phase']}`. Validated full matrix cells: **{complete}/50**.", "",
             "Final-epoch, independently reloaded test CE; lower is better. Partial seed results are exploratory.",
             "Intervals require all three paired seeds and describe training randomness on this fixed test set.", ""]
    if errors:
        lines += ["**AUDIT FAILURE: do not interpret these results.**", "", *errors, ""]
    for alias in ("qwen3_06b", "qwen25_7b"):
        model = {"rows": [], "contrasts": {}}
        lines += [f"## {alias}", "", "| hidden rank | placement | seeds | mean test CE | total trainable |", "| ---: | --- | --- | ---: | ---: |"]
        for rank in (0, 1, 8):
            for placement in ("none", "input", "output"):
                matching = [(key, report) for key, report in observations.items() if key[:3] == (alias, rank, placement)]
                if not matching:
                    continue
                values = [report["avg_ce"] for key, report in matching]
                seeds = [key[3] for key, report in matching]
                params = [audits[key]["total_trainable"] if key in audits else 0 for key, report in matching]
                row = {"rank": rank, "placement": placement, "seeds": seeds, "ce": describe(values), "parameters": params}
                model["rows"].append(row)
                lines.append(f"| {rank} | {placement} | {','.join(str(s) for s in seeds)} | {mean(values):.6f} | {params[0]:,} |")
        lines += ["", "| paired contrast | mean [95% t CI when n=3] |", "| --- | --- |"]
        deltas = {}
        for rank in (0, 1, 8):
            deltas[rank] = {}
            for seed in (42, 43, 44):
                inp = observations.get((alias, rank, "input", seed))
                out = observations.get((alias, rank, "output", seed))
                if inp and out:
                    deltas[rank][seed] = out["avg_ce"] - inp["avg_ce"]
            name = f"D({rank}) = output - input"
            model["contrasts"][name] = describe(list(deltas[rank].values()))
            lines.append(f"| {name} | {show(model['contrasts'][name])} |")
            for placement in ("input", "output"):
                values = []
                for seed in (42, 43, 44):
                    base = observations.get((alias, rank, "none", None if rank == 0 else seed))
                    treatment = observations.get((alias, rank, placement, seed))
                    if base and treatment:
                        values.append(treatment["avg_ce"] - base["avg_ce"])
                name = f"r{rank}: {placement} - no boundary"
                model["contrasts"][name] = describe(values)
                lines.append(f"| {name} | {show(model['contrasts'][name])} |")
        for rank in (8, 1):
            values = [deltas[rank][seed] - deltas[0][seed] for seed in (42, 43, 44) if seed in deltas[rank] and seed in deltas[0]]
            name = f"I({rank}) = D({rank}) - D(0)"
            model["contrasts"][name] = describe(values)
            lines.append(f"| {name} | {show(model['contrasts'][name])} |")
        lines += ["", "A reversal needs D(0)>0 AND D(8)<0; a negative interaction alone is insufficient.", ""]
        result["models"][alias] = model
    for filename, content in (("RESULTS.md", "\n".join(lines) + "\n"), ("summary.json", json.dumps(result, indent=2) + "\n")):
        path = HERE / filename
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


if __name__ == "__main__":
    summarize()
