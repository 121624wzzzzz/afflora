#!/usr/bin/env python
"""Build an auditable, non-evaluation UltraChat constraint-following anchor."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "corrected_sft_experiment/data_ultrachat_100k_len1024/train.jsonl"
OUTPUT = ROOT / "reviewer_followup/tied_energy_lambda_sweep_qwen3/ultrachat_explicit_constraints_len1024.jsonl"
MANIFEST = OUTPUT.with_suffix(".manifest.json")
PATTERN = re.compile(r"\b(exactly|at least|at most|must|do not|without|only)\b", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    kept = 0
    with INPUT.open(encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8") as out:
        for line in source:
            row = json.loads(line)
            first = row["conversations"][0]
            if first.get("role") != "user":
                raise RuntimeError("Filtered UltraChat row does not begin with user")
            if PATTERN.search(first["content"]):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    manifest = {
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "kept": kept,
        "prompt_filter": PATTERN.pattern,
        "purpose": "non-evaluation explicit-constraint behavior anchor",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
