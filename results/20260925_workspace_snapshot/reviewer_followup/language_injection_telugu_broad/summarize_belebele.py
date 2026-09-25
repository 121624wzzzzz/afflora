#!/usr/bin/env python
"""Report paired broad-Telugu held-out accuracy effects with a bootstrap CI."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    score_dir = Path(args.scores_dir)
    control = []
    candidate = []
    for seed in range(59, 67):
        control.append(json.loads((score_dir / f"qwen25_15b_broadtelugu_tied_l0_anchor0p1_sd{seed}.json").read_text())["accuracy"])
        candidate.append(json.loads((score_dir / f"qwen25_15b_broadtelugu_tied_l100early400_anchor0p1_dualkl0p05_sd{seed}.json").read_text())["accuracy"])
    delta = [treatment - baseline for baseline, treatment in zip(control, candidate)]
    rng = random.Random(args.seed)
    means = sorted(
        sum(delta[rng.randrange(len(delta))] for _ in delta) / len(delta)
        for _ in range(args.bootstrap_samples)
    )
    result = {
        "metric": "Telugu Belebele option-likelihood accuracy on fixed 176-row held-out split",
        "control": control,
        "candidate": candidate,
        "paired_delta": delta,
        "control_mean": sum(control) / len(control),
        "candidate_mean": sum(candidate) / len(candidate),
        "mean_paired_delta": sum(delta) / len(delta),
        "bootstrap_95_ci": [means[int(.025 * len(means))], means[int(.975 * len(means))]],
        "positive_seeds": sum(value > 0 for value in delta),
        "negative_seeds": sum(value < 0 for value in delta),
        "ties": sum(value == 0 for value in delta),
        "bootstrap_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
