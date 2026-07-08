#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def load_results(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]
    by_idx = {int(x["dataset_index"]): x for x in results}
    if len(by_idx) != len(results):
        raise ValueError(f"duplicate dataset_index in {path}")
    return data, by_idx


def summary_of(data):
    if "summary" in data:
        return data["summary"]
    return {"full": data["full"], "clean": data["clean"]}


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def metric(items):
    n = len(items)
    correct = sum(1 for x in items if x["is_correct"])
    return {"num_samples": n, "correct": correct, "accuracy_pct": pct(correct, n)}


def paired_delta(a_items, b_items):
    if len(a_items) != len(b_items):
        raise ValueError("paired inputs must have same length")
    n = len(a_items)
    diffs = [(1 if b["is_correct"] else 0) - (1 if a["is_correct"] else 0) for a, b in zip(a_items, b_items)]
    plus = sum(1 for d in diffs if d == 1)
    minus = sum(1 for d in diffs if d == -1)
    return {
        "delta_accuracy_pct_points": pct(sum(diffs), n),
        "afflora_correct_hidden_wrong": plus,
        "hidden_correct_afflora_wrong": minus,
        "same_correct": sum(1 for a, b in zip(a_items, b_items) if a["is_correct"] and b["is_correct"]),
        "same_wrong": sum(1 for a, b in zip(a_items, b_items) if not a["is_correct"] and not b["is_correct"]),
    }


def bootstrap_delta_ci(a_items, b_items, *, rounds=10000, seed=42):
    rng = random.Random(seed)
    n = len(a_items)
    diffs = [(1 if b["is_correct"] else 0) - (1 if a["is_correct"] else 0) for a, b in zip(a_items, b_items)]
    vals = []
    greater = 0
    for _ in range(rounds):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        v = pct(s, n)
        vals.append(v)
        if v > 0:
            greater += 1
    vals.sort()
    lo = vals[int(0.025 * rounds)]
    hi = vals[int(0.975 * rounds) - 1]
    return {
        "rounds": rounds,
        "seed": seed,
        "ci95_delta_accuracy_pct_points": [lo, hi],
        "bootstrap_prob_delta_gt_0": greater / rounds,
    }


def subset(by_idx, indices):
    return [by_idx[i] for i in indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--hidden", type=Path, required=True)
    ap.add_argument("--afflora", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    base_data, base = load_results(args.base)
    hidden_data, hidden = load_results(args.hidden)
    aff_data, aff = load_results(args.afflora)

    indices = sorted(hidden)
    if indices != sorted(aff) or indices != sorted(base):
        raise ValueError("result files do not cover the same dataset_index set")

    contaminated = sorted(i for i in indices if hidden[i]["is_known_contaminated"])
    clean = [i for i in indices if i not in set(contaminated)]

    def model_section(name, data, by_idx):
        return {
            "summary_from_file": summary_of(data),
            "contaminated": metric(subset(by_idx, contaminated)),
            "contaminated_indices": [
                {
                    "dataset_index": i,
                    "is_correct": by_idx[i]["is_correct"],
                    "prediction": by_idx[i]["prediction"],
                    "ground_truth_extracted": by_idx[i]["ground_truth_extracted"],
                }
                for i in contaminated
            ],
        }

    full_hidden = subset(hidden, indices)
    full_aff = subset(aff, indices)
    clean_hidden = subset(hidden, clean)
    clean_aff = subset(aff, clean)

    out = {
        "schema_version": 1,
        "inputs": {
            "base": str(args.base),
            "hidden": str(args.hidden),
            "afflora": str(args.afflora),
        },
        "num_samples": len(indices),
        "known_contaminated": {
            "num_samples": len(contaminated),
            "indices": contaminated,
        },
        "models": {
            "base": model_section("base", base_data, base),
            "hidden": model_section("hidden", hidden_data, hidden),
            "afflora": model_section("afflora", aff_data, aff),
        },
        "hidden_vs_afflora": {
            "full": {
                **paired_delta(full_hidden, full_aff),
                **bootstrap_delta_ci(full_hidden, full_aff),
            },
            "clean": {
                **paired_delta(clean_hidden, clean_aff),
                **bootstrap_delta_ci(clean_hidden, clean_aff),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["hidden_vs_afflora"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
