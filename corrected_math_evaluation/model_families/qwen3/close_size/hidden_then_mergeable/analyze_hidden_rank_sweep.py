#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DESTINATION = HERE / "HIDDEN_RANK_ANALYSIS.md"
MODELS = ("qwen3_06b", "qwen3_17b")
HIDDEN_RANKS = (1, 2, 4, 8, 16)
SEED = 42


def read_score(model: str, rank: int, task: str) -> float | None:
    path = OUTPUTS / task / f"{model}_hidden_hr{rank}_seed{SEED}_full.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = 5000 if task == "math" else 1319
    rows = report["results"]
    if report["full"]["num_samples"] != expected or [row["dataset_index"] for row in rows] != list(range(expected)):
        raise ValueError(f"Incomplete or duplicate coverage: {path}")
    return float(report["clean" if task == "math" else "full"]["accuracy_pct"])


def main() -> None:
    lines = [
        "# Qwen3 close-size hidden LoRA rank analysis",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "This is the Qwen3 main-structure stage before running mergeable AffLoRA.",
        "MATH reports the clean-4,995 score.",
        "",
    ]
    for model in MODELS:
        rows = []
        for rank in HIDDEN_RANKS:
            m = read_score(model, rank, "math")
            g = read_score(model, rank, "gsm8k")
            rows.append((rank, m, g))
        complete = [row for row in rows if row[1] is not None and row[2] is not None]
        lines += [
            f"## {model}",
            "",
            "| hidden rank | MATH | GSM8K | mean(MATH,GSM8K) |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for rank, m, g in rows:
            if m is None or g is None:
                lines.append(f"| {rank} | - | - | - |")
            else:
                lines.append(f"| {rank} | {m:.4f}% | {g:.4f}% | {(m+g)/2:.4f}% |")
        lines += ["", "### Ranking by MATH", ""]
        if complete:
            lines += ["| rank by MATH | hidden rank | MATH | GSM8K |", "| ---: | ---: | ---: | ---: |"]
            for idx, (rank, m, g) in enumerate(sorted(complete, key=lambda row: row[1], reverse=True), 1):
                lines.append(f"| {idx} | {rank} | {m:.4f}% | {g:.4f}% |")
            best_math = max(complete, key=lambda row: row[1])
            best_gsm = max(complete, key=lambda row: row[2])
            best_avg = max(complete, key=lambda row: (row[1] + row[2]) / 2)
            chosen = []
            for row in (best_math, best_gsm, best_avg):
                if row[0] not in chosen:
                    chosen.append(row[0])
            lines += [
                "",
                "### Suggested mergeable follow-up hidden ranks",
                "",
            ]
            for rank in chosen:
                lines.append(f"- hidden hr{rank}")
            lines += [
                "",
                f"Rationale: best MATH hr{best_math[0]}, best GSM8K hr{best_gsm[0]}, best average hr{best_avg[0]}.",
                "",
            ]
        else:
            lines += ["Pending.", ""]
    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
