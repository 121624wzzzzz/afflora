#!/usr/bin/env python
"""Prepare a deterministic, leakage-free Telugu Belebele adaptation split.

This is a *task/language-injection pilot*, not a claim of general multilingual
instruction tuning: the target is Telugu reading comprehension under a fixed
multiple-choice response contract.  The held-out rows are never repeated in
training and are scored by option likelihood rather than by held-out CE.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


INSTRUCTION = (
    "క్రింది పాఠ్యాన్ని చదివి ప్రశ్నకు సరైన ఎంపికను ఎంచుకోండి. "
    "సమాధానంలో ఎంపిక సంఖ్య మాత్రమే ఇవ్వండి."
)


def render(row: dict[str, str]) -> str:
    options = "\n".join(f"{i}. {row[f'mc_answer{i}']}" for i in range(1, 5))
    return f"{INSTRUCTION}\n\nపాఠ్యం:\n{row['flores_passage']}\n\nప్రశ్న:\n{row['question']}\n\nఎంపికలు:\n{options}"


def conversation(row: dict[str, str], record_id: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source": "facebook/belebele",
        "source_dialect": row["dialect"],
        "source_question_number": row["question_number"],
        "conversations": [
            {"role": "user", "content": render(row)},
            {"role": "assistant", "content": str(row["correct_answer_num"])},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--test-size", type=int, default=180)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    if len(rows) <= args.test_size:
        raise ValueError(f"Need more than {args.test_size} rows, found {len(rows)}")
    ids = list(range(len(rows)))
    random.Random(args.seed).shuffle(ids)
    test_ids = set(ids[: args.test_size])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, selected in (("train", [i for i in ids if i not in test_ids]), ("test", [i for i in ids if i in test_ids])):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for i in selected:
                stream.write(json.dumps(conversation(rows[i], f"belebele_tel_{i}"), ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(Path(args.input).resolve()),
                "dialect": "tel_Telu",
                "split_seed": args.seed,
                "train_rows": len(rows) - args.test_size,
                "test_rows": args.test_size,
                "response_contract": "the single correct option number 1-4",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
