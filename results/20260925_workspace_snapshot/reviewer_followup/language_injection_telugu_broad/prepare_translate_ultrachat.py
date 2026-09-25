#!/usr/bin/env python
"""Build a deterministic, auditable English-to-Telugu instruction corpus.

The source is the local UltraChat training split.  Only its first user/assistant
turn is used, so every output record is a single instruction example.  We select
short source pairs before translation, shard them by stable source index, and
retain the English source in each record.  This makes translation restartable and
keeps the later SFT corpus fully traceable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def select_rows(source: Path, limit: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream):
            row = json.loads(line)
            turns = row.get("conversations", [])
            if len(turns) < 2 or turns[0].get("role") != "user" or turns[1].get("role") != "assistant":
                continue
            user, assistant = turns[0]["content"].strip(), turns[1]["content"].strip()
            if not user or not assistant or len(user) > 1200 or len(assistant) > 1800 or len(user) + len(assistant) > 2800:
                continue
            rows.append({
                "source_index": row.get("source_index", line_no),
                "source_record_id": row.get("record_id"),
                "english_user": user,
                "english_assistant": assistant,
            })
    random.Random(seed).shuffle(rows)
    if len(rows) < limit:
        raise ValueError(f"Only {len(rows)} eligible rows, need {limit}")
    return rows[:limit]


def batches(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def translate(tokenizer, model, texts: list[str], device: torch.device, batch_size: int) -> list[str]:
    translated: list[str] = []
    tokenizer.src_lang = "eng_Latn"
    target_id = tokenizer.convert_tokens_to_ids("tel_Telu")
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise ValueError("NLLB tokenizer does not expose tel_Telu")
    for batch in batches(texts, batch_size):
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
            )
        translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return translated


def write_selection(source: Path, selection: list[dict[str, object]], selection_path: Path, seed: int, limit: int) -> None:
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source.resolve()),
        "source_sha256": digest(source),
        "selection_seed": seed,
        "selection_limit": limit,
        "criteria": "first user/assistant turn; user<=1200 chars; assistant<=1800 chars; total<=2800 chars",
        "selected_source_indices": [row["source_index"] for row in selection],
    }
    selection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_nllb_model(model_path: str, device: torch.device):
    """Load the trusted local NLLB checkpoint without transformers' torch>=2.6 gate.

    The repository snapshot is the official ModelScope mirror, and weights_only
    prevents arbitrary pickle execution.  Transformers' gate applies even in that
    safe mode on this cluster's Torch 2.4 environment.
    """
    model_root = Path(model_path)
    checkpoint = model_root / "pytorch_model.bin"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing NLLB checkpoint: {checkpoint}")
    config = AutoConfig.from_pretrained(model_root)
    model = AutoModelForSeq2SeqLM.from_config(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatibility = model.load_state_dict(state_dict, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"Unexpected NLLB state dict mismatch: {incompatibility}")
    return model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, device=device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--selection-limit", type=int, default=30000)
    parser.add_argument("--selection-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--write-selection-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if not 0 <= args.selection_start < args.selection_limit:
        raise ValueError("selection-start must be in [0, selection-limit)")

    source = Path(args.source)
    selection = select_rows(source, args.selection_limit, args.seed)
    selection_path = Path(args.selection_file)
    if args.write_selection_only:
        write_selection(source, selection, selection_path, args.seed, args.selection_limit)
        return
    if not args.output:
        raise ValueError("--output is required unless --write-selection-only is used")
    if not selection_path.exists():
        raise FileNotFoundError(f"Selection manifest must be written first: {selection_path}")
    shard = [
        row for i, row in enumerate(selection)
        if i >= args.selection_start and i % args.num_shards == args.shard_index
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_nllb_model(args.model, device)
    users = translate(tokenizer, model, [str(row["english_user"]) for row in shard], device, args.batch_size)
    assistants = translate(tokenizer, model, [str(row["english_assistant"]) for row in shard], device, args.batch_size)
    if len(users) != len(shard) or len(assistants) != len(shard):
        raise RuntimeError("translation cardinality mismatch")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row, user, assistant in zip(shard, users, assistants, strict=True):
            stream.write(json.dumps({
                **row,
                "telugu_user": user.strip(),
                "telugu_assistant": assistant.strip(),
                "translation_model": "facebook/nllb-200-distilled-600M",
                "translation_direction": "eng_Latn->tel_Telu",
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(shard), "shard": args.shard_index}), flush=True)


if __name__ == "__main__":
    main()
