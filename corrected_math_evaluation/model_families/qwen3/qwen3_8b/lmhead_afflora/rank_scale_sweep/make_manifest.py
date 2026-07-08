#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import peft
import torch
import transformers


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data").is_dir() and (p / "scripts").is_dir())
FILES = [
    ROOT / "data/metamathqa_40k/train.jsonl",
    ROOT / "data/math/test.jsonl",
    ROOT / "data/gsm8k/test.jsonl",
    ROOT / "scripts/train_affine_vocab_lora.py",
    ROOT / "src/affine_vocab_lora/adapter.py",
    ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_math_full.py",
    ROOT / "corrected_math_evaluation/shared/evaluators/evaluate_gsm8k_full.py",
    ROOT / "corrected_math_evaluation/shared/data_quality/known_contaminated_math_indices.json",
    HERE / "run_experiment.py",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "versions": {"python_torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "peft": peft.__version__},
        "files": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in FILES},
    }
    destination = HERE / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
