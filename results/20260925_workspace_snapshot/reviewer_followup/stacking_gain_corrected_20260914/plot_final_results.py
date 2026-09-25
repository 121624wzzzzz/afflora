#!/usr/bin/env python3
"""Export the audited 72-cell matrix; never plot an incomplete final result."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent
MODELS = {"qwen3_06b": "Qwen3-0.6B-Base", "qwen25_7b": "Qwen2.5-7B-Base"}
CONTROLS = {"none": "Hidden LoRA", "hidden_budget": "Hidden parameter-budget control"}
RANKS = [8, 32, 64]


def main():
    summary_path = ROOT / "summary.json"
    audit_path = ROOT / "FINAL_AUDIT.json"
    summary = json.loads(summary_path.read_text())
    if summary["completed"] != 72 or summary["audit_errors"] or not audit_path.exists():
        raise SystemExit("Final export requires all 72 completed cells and FINAL_AUDIT.json.")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "passed" or audit.get("cells") != 72 or audit.get("audit_errors"):
        raise SystemExit("Final export requires a passed 72-cell audit.")
    records = []
    for model in MODELS:
        contrasts = summary["models"][model]["contrasts"]
        for rank in RANKS:
            for control in CONTROLS:
                contrast = contrasts[f"r{rank}: both - {control}"]
                if contrast["seeds"] != [42, 43, 44]:
                    raise ValueError(f"Incomplete seed coverage: {model}, {rank}, {control}")
                row = {"model": model, "hidden_rank": rank, "treatment": "both", "control": control}
                for metric in ["ce", "strict", "valid_strict"]:
                    value = contrast[metric]
                    row.update({f"{metric}_delta_mean": value["mean"],
                                f"{metric}_delta_sd": value["sd"],
                                f"{metric}_delta_ci95_low": value["ci95"][0],
                                f"{metric}_delta_ci95_high": value["ci95"][1]})
                records.append(row)
    out = ROOT / "final_figures"
    out.mkdir(exist_ok=True)
    with (out / "paired_contrasts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    colors = {"qwen3_06b": "#2166AC", "qwen25_7b": "#B35806"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), sharex=True, sharey="row")
    for col, (control, label) in enumerate(CONTROLS.items()):
        for model, model_label in MODELS.items():
            values = [r for r in records if r["model"] == model and r["control"] == control]
            for panel, metric, scale in [(0, "ce", -1.0), (1, "strict", 100.0)]:
                means = [scale * r[f"{metric}_delta_mean"] for r in values]
                intervals = [sorted([scale * r[f"{metric}_delta_ci95_low"],
                                     scale * r[f"{metric}_delta_ci95_high"]]) for r in values]
                lower = [m - bounds[0] for m, bounds in zip(means, intervals)]
                upper = [bounds[1] - m for m, bounds in zip(means, intervals)]
                axes[panel, col].errorbar(RANKS, means, yerr=[lower, upper],
                                         color=colors[model], marker="o", capsize=3,
                                         linewidth=1.7, markersize=5, label=model_label)
        axes[0, col].set_title(f"Bilateral addition vs. {label}", fontsize=10)
        axes[1, col].set_xlabel("Hidden LoRA rank")
        for panel in range(2):
            axes[panel, col].axhline(0, color="#666666", linewidth=0.9, linestyle="--")
            axes[panel, col].set_xticks(RANKS)
            axes[panel, col].grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Test CE reduction\n(control minus bilateral)")
    axes[1, 0].set_ylabel("IFEval strict prompt gain\n(percentage points)")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    axes[0, 0].yaxis.set_major_formatter(formatter)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Increment from adding independent input/output boundary adapters", y=1.02, fontsize=13)
    fig.text(0.5, 0.025, "Positive favors bilateral addition. Mean and nominal paired t 95% CI across 3 seeds; no multiplicity correction.\n"
             "IFEval: all 541 prompts, greedy, max 512 tokens. Base-model special-token limitations apply.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"stacking_increment.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    caption = (
        "# Stacking increment: final audited matrix\n\n"
        "The panels show adding independent input/output boundary adapters to Hidden LoRA, "
        "relative to Hidden LoRA and to the predeclared hidden parameter-budget control. "
        "Hidden ranks are 8, 32, and 64; boundary rank is 16. Positive values favor the bilateral addition.\n\n"
        "Error bars are nominal paired t 95% intervals over seeds 42, 43, and 44, "
        "without correction for multiple comparisons. They describe training-seed variation on the "
        "fixed evaluation sets, not uncertainty over all possible tasks or data distributions. "
        "CE uses the frozen 1,000-example test split. IFEval uses all 541 prompts and the frozen "
        "greedy, batch-8, 512-token generation protocol. The CSV also includes the fixed valid539 sensitivity.\n\n"
        "The budget control uses one predeclared allocation of additional hidden ranks, "
        "not an optimized allocation frontier. Qwen3 has identical frozen output rows for native "
        "chat special tokens; see ../GENERATION_LIMITATION.md. CE reductions do not by themselves "
        "establish generation quality or distinguish calibration from semantic improvement.\n\n"
        "CSV deltas are treatment minus control. Plot CE signs are reversed so positive is favorable; "
        "IFEval accuracy deltas are multiplied by 100 to display percentage points.\n"
    )
    (out / "CAPTION.md").write_text(caption)
    source_paths = [summary_path, audit_path, ROOT / "manifest.json", Path(__file__).resolve()]
    (out / "SOURCE_HASHES.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                        for p in source_paths}, indent=2) + "\n")
    print(out / "stacking_increment.png")


if __name__ == "__main__":
    main()
