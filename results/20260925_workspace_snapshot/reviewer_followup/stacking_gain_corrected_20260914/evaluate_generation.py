"""Frozen, restartable IFEval generation and local official scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "source/corrected_sft_experiment"),
               str(HERE / "source"), str(HERE / "ifeval_deps")]
os.environ["NLTK_DATA"] = str(HERE / "nltk_data")

import torch
from transformers import set_seed
import evaluate_corrected_sft as ce
import support


def checkpoint_hashes(path):
    names = ("run_args.json", "adapter_config.json", "adapter_model.safetensors",
             "affine_vocab_config.json", "affine_vocab_adapter.safetensors")
    return {name: support.sha(path / name) for name in names if (path / name).is_file()}


def prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                   add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def stop_token_ids(checkpoint):
    args = json.loads((checkpoint / "run_args.json").read_text())
    base = Path(args["model_path"])
    tokenizer = ce.AutoTokenizer.from_pretrained(base, use_fast=True)
    config = json.loads((base / "generation_config.json").read_text())
    configured = config.get("eos_token_id") or tokenizer.eos_token_id
    ids = set(configured if isinstance(configured, list) else [configured])
    turn_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assert len(turn_end) == 1 and turn_end[0] != tokenizer.unk_token_id
    ids.update(turn_end)
    return sorted(ids)


def score(rows, responses):
    random.seed(0)
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    from instruction_following_eval import evaluation_lib as official
    by_prompt = {row["prompt"]: answer["response"] for row, answer in zip(rows, responses)}
    assert len(by_prompt) == len(rows)
    details = []
    for row in rows:
        inp = official.InputExample(**{k: row[k] for k in ("key", "prompt", "instruction_id_list", "kwargs")})
        strict = official.test_instruction_following_strict(inp, by_prompt)
        loose = official.test_instruction_following_loose(inp, by_prompt)
        details.append({"key": row["key"], "strict": strict.follow_all_instructions,
                        "loose": loose.follow_all_instructions,
                        "strict_instructions": strict.follow_instruction_list,
                        "loose_instructions": loose.follow_instruction_list})
    total = sum(len(row["instruction_id_list"]) for row in rows)
    # Unmodified upstream checker randomizes '#' and '!' as ASCII letters.
    # Keep its seeded all-prompt scores and expose an unaffected sensitivity.
    valid = [row for row in details if row["key"] not in (1122, 1129)]
    return {"examples": len(rows), "instructions": total,
        "valid_prompt_sensitivity": {
            "excluded_keys": [row["key"] for row in details if row["key"] in (1122,1129)],
            "examples": len(valid),
            "strict_prompt_accuracy": sum(r["strict"] for r in valid)/len(valid),
            "loose_prompt_accuracy": sum(r["loose"] for r in valid)/len(valid)},
        "strict_prompt_accuracy": sum(r["strict"] for r in details) / len(rows),
        "loose_prompt_accuracy": sum(r["loose"] for r in details) / len(rows),
        "strict_instruction_accuracy": sum(sum(r["strict_instructions"]) for r in details) / total,
        "loose_instruction_accuracy": sum(sum(r["loose_instructions"]) for r in details) / total,
        "mean_generated_tokens": sum(r["generated_tokens"] for r in responses) / len(rows),
        "cap_hit_rate": sum(r["hit_token_cap"] for r in responses) / len(rows),
        "per_example": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=541)
    args = parser.parse_args()
    torch.set_num_threads(4)
    set_seed(0)
    cp, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = ce.load_rows(HERE / "data/ifeval.jsonl")[:args.limit]
    identity = {"checkpoint": str(cp), "checkpoint_files": checkpoint_hashes(cp),
        "data_sha256": support.sha(HERE / "data/ifeval.jsonl"),
        "protocol_sha256": support.sha(HERE / "manifest.json"),
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "count": len(rows), "do_sample": False, "enable_thinking": False,
        "langdetect_seed": 0, "generation_seed": 0, "python_scoring_seed": 0,
        "sensitivity_excluded_keys": [1122,1129],
        "eos_token_ids": stop_token_ids(cp),
        "stop_policy": "base_eos_plus_native_im_end"}
    meta = out / "generation_metadata.json"
    if meta.exists():
        assert json.loads(meta.read_text()) == identity, "Generation identity changed"
    else:
        support.write_json(meta, identity)
    path = out / "responses.jsonl"
    responses = ce.load_rows(path) if path.exists() else []
    assert len(responses) <= len(rows)
    for row, response in zip(rows, responses):
        assert all(row[k] == response[k] for k in ("key", "prompt", "instruction_id_list", "kwargs"))
        assert isinstance(response["response"], str)
    if len(responses) < len(rows):
        model, tokenizer, _, _ = ce.load_model(argparse.Namespace(run_dir=str(cp), model_path=None,
                                                           affine_ablation="none", device="cuda"))
        tokenizer.padding_side = "left"
        eos = set(identity["eos_token_ids"])
        assert tokenizer.encode("<|im_end|>", add_special_tokens=False)[0] in eos
        with path.open("a", encoding="utf-8") as stream, torch.inference_mode():
            for begin in range(len(responses), len(rows), args.batch_size):
                batch = rows[begin:begin+args.batch_size]
                encoded = tokenizer([prompt(tokenizer, r["prompt"]) for r in batch],
                                    return_tensors="pt", padding=True).to("cuda")
                generated = model.generate(**encoded, do_sample=False,
                    max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=list(eos))[:, encoded["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for row, text, tokens in zip(batch, texts, generated.tolist()):
                    stop = next((i for i, token in enumerate(tokens) if token in eos), None)
                    length = stop+1 if stop is not None else len(tokens)
                    result = {**row, "response": text, "generated_tokens": length,
                              "terminal_token_id": tokens[stop] if stop is not None else None,
                              "hit_token_cap": stop is None and length == args.max_new_tokens}
                    responses.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False)+"\n")
                stream.flush()
                print(f"generated {len(responses)}/{len(rows)}", flush=True)
        del model
        torch.cuda.empty_cache()
    assert len(responses) == len(rows)
    assert checkpoint_hashes(cp) == identity["checkpoint_files"], "Checkpoint changed during generation"
    result = score(rows, responses)
    result.update(identity=identity, responses_sha256=support.sha(path), scored_at=support.now())
    support.write_json(out / "scores.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_example", "identity")}), flush=True)


if __name__ == "__main__": main()
