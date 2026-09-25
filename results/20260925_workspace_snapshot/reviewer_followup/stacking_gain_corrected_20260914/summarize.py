"""All fixed stacking comparisons, with pairing and reuse approval gates."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

import support

HERE = Path(__file__).resolve().parent
ARMS = ("none", "output", "both", "hidden_budget")


def describe(values):
    if not values: return None
    result = {"n": len(values), "values": values, "mean": mean(values)}
    if len(values) == 3:
        sd = stdev(values)
        half = 4.302652729911275 * sd / math.sqrt(3)
        result.update(sd=sd, ci95=[result["mean"]-half, result["mean"]+half])
    return result


def show(value, scale=1):
    if not value: return "pending"
    text = f"{value['mean']*scale:+.6f} (n={value['n']})"
    if "ci95" in value:
        text += f" [{value['ci95'][0]*scale:+.6f}, {value['ci95'][1]*scale:+.6f}]"
    return text


def summarize():
    state = json.loads((HERE / "state.json").read_text())
    observations, audits, errors = {}, {}, []
    for job in state["matrix"]:
        if state.get("jobs", {}).get(job["name"], {}).get("status") != "complete":
            continue
        key = (job["model"], job["hidden_rank"], job["placement"], job["seed"])
        report = json.loads((HERE / "reports" / f"{job['name']}.test.json").read_text())
        scores = json.loads((HERE / "ifeval" / job["name"] / "scores.json").read_text())
        cp = Path(job["checkpoint"])
        audit = json.loads((cp / "initialization_audit.json").read_text())
        observations[key] = (report, scores)
        audits[key] = audit
        if job.get("reuse"):
            approval = HERE / "reuse_audit/APPROVED.json"
            if not approval.exists() or job["name"] not in json.loads(approval.read_text())["comparisons"]:
                errors.append(f"Unapproved reuse: {job['name']}")
    for alias in support.MODELS:
        ids = {tuple((r["record_id"], r["token_count"]) for r in ce["per_example"])
               for key, (ce, _) in observations.items() if key[0] == alias}
        if len(ids) > 1: errors.append(f"Unpaired CE examples/tokens: {alias}")
        ids = {tuple(r["key"] for r in score["per_example"])
               for key, (_, score) in observations.items() if key[0] == alias}
        if len(ids) > 1: errors.append(f"Unpaired IFEval prompts: {alias}")
        for rank in (8, 32, 64):
            for seed in (42, 43, 44):
                selected = [audit for key, audit in audits.items() if key[0] == alias and key[1] == rank and key[3] == seed]
                hashes = {a.get("shared_hidden_init_sha256", a["hidden_init_sha256"]) for a in selected}
                if len(hashes) > 1: errors.append(f"Shared hidden initialization mismatch: {alias} r{rank} seed{seed}")
                affine = set()
                for a in selected:
                    if "affine_components" in a: affine.update(a["affine_components"].values())
                    elif a.get("affine_init_sha256"): affine.add(a["affine_init_sha256"])
                if len(affine) > 1: errors.append(f"Boundary initialization mismatch: {alias} r{rank} seed{seed}")
    result = {"phase": state["phase"], "completed": len(observations), "total":72,
              "audit_errors": errors, "models": {}}
    lines = ["# Stacking A-LoRA on hidden LoRA", "",
        f"Phase: `{state['phase']}`; **{len(observations)}/72** cells have complete CE and IFEval.", "",
        "Primary question: incremental benefit versus hidden-only. Every fixed arm/seed is reported.",
        "CE: lower is better. IFEval: higher is better. All intervals below are nominal paired t intervals",
        "over three training seeds on fixed, previously inspected evaluation sets. Partial results are exploratory.", ""]
    if errors: lines += ["**AUDIT FAILURE: do not interpret results.**", *errors, ""]
    for alias in support.MODELS:
        model = {"rows":[], "contrasts":{}}
        lines += [f"## {alias}", "", "| hidden rank | arm | seeds | test CE | IFEval strict | IFEval loose | valid-539 strict | trainable params |",
                  "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for rank in (8, 32, 64):
            for arm in ARMS:
                keys = sorted([k for k in observations if k[:3] == (alias, rank, arm)])
                if not keys: continue
                row = {"rank":rank, "arm":arm, "seeds":[k[3] for k in keys],
                       "ce":describe([observations[k][0]["avg_ce"] for k in keys]),
                       "strict":describe([observations[k][1]["strict_prompt_accuracy"] for k in keys]),
                       "loose":describe([observations[k][1]["loose_prompt_accuracy"] for k in keys]),
                       "valid_strict":describe([observations[k][1]["valid_prompt_sensitivity"]["strict_prompt_accuracy"] for k in keys]),
                       "parameters":[audits[k]["total_trainable"] for k in keys]}
                model["rows"].append(row)
                lines.append(f"| {rank} | {arm} | {','.join(map(str,row['seeds']))} | {row['ce']['mean']:.6f} | {row['strict']['mean']:.6f} | {row['loose']['mean']:.6f} | {row['valid_strict']['mean']:.6f} | {row['parameters'][0]:,} |")
        lines += ["", "| paired contrast | delta CE [95% CI] | delta IFEval strict, pp [95% CI] | delta valid-539 strict, pp [95% CI] |", "| --- | --- | --- | --- |"]
        for rank in (8, 32, 64):
            for treatment, control in (("output","none"),("both","none"),("hidden_budget","none"),("both","hidden_budget"),("both","output")):
                ce_values, strict_values, loose_values, valid_values, seeds = [], [], [], [], []
                for seed in (42,43,44):
                    a = observations.get((alias,rank,treatment,seed))
                    b = observations.get((alias,rank,control,seed))
                    if a and b:
                        seeds.append(seed)
                        ce_values.append(a[0]["avg_ce"]-b[0]["avg_ce"])
                        strict_values.append(a[1]["strict_prompt_accuracy"]-b[1]["strict_prompt_accuracy"])
                        loose_values.append(a[1]["loose_prompt_accuracy"]-b[1]["loose_prompt_accuracy"])
                        valid_values.append(a[1]["valid_prompt_sensitivity"]["strict_prompt_accuracy"]-b[1]["valid_prompt_sensitivity"]["strict_prompt_accuracy"])
                name = f"r{rank}: {treatment} - {control}"
                model["contrasts"][name] = {"seeds":seeds,"ce":describe(ce_values),
                    "strict":describe(strict_values),"loose":describe(loose_values),
                    "valid_strict":describe(valid_values),
                    "ce_improvements":sum(x<0 for x in ce_values),"strict_improvements":sum(x>0 for x in strict_values)}
                lines.append(f"| {name} | {show(describe(ce_values))} | {show(describe(strict_values),100)} | {show(describe(valid_values),100)} |")
        lines += ["", "The hidden-budget arm matches the bilateral added budget (7B +512 parameters).",
                  "It is one fixed allocation, not an optimized hidden-rank frontier. A small CE gain alone",
                  "does not establish generation gains or superiority to other PEFT methods.",
                  "Official all-541 scoring uses Python seed 0. Valid-539 excludes keys 1122/1129",
                  "because the unmodified official checker replaces their punctuation targets with",
                  "random letters. A generation-gain claim must survive this fixed sensitivity.", ""]
        result["models"][alias] = model
    support.write_json(HERE / "summary.json", result)
    temp = HERE / "RESULTS.md.tmp"
    temp.write_text("\n".join(lines)+"\n")
    temp.replace(HERE / "RESULTS.md")
    if errors: raise RuntimeError("; ".join(errors))
    return result


if __name__ == "__main__": summarize()
