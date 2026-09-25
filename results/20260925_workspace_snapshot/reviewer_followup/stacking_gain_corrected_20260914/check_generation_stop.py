"""Real-model check of the native assistant end token used in SFT."""
import argparse
import json
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList
import evaluate_generation as generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cp = Path(args.checkpoint).resolve()
    stops = generation.stop_token_ids(cp)
    model, tokenizer, _, _ = generation.ce.load_model(argparse.Namespace(
        run_dir=str(cp), model_path=None, affine_ablation="none", device="cuda"))
    tokenizer.padding_side = "left"
    prompts = ["Reply with exactly the word yes.", "Write one short sentence about the sky."]
    batch = tokenizer([generation.prompt(tokenizer, p) for p in prompts],
                      return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        result = model.generate(**batch, do_sample=False, max_new_tokens=128,
                                eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
    records = []
    for prompt, row in zip(prompts, result[:,batch["input_ids"].shape[1]:].tolist()):
        end = next((i for i, token in enumerate(row) if token in stops), None)
        length = end+1 if end is not None else len(row)
        records.append({"prompt": prompt, "generated_tokens": length,
                        "terminal_token_id": row[end] if end is not None else None,
                        "hit_token_cap": end is None,
                        "raw_response": tokenizer.decode(row[:length], skip_special_tokens=False)})
    # Model compliance is an outcome, not a test prerequisite. Independently
    # verify both EOS IDs and unequal batch stopping lengths in the real loader.
    width = batch["input_ids"].shape[1]
    ordinary = tokenizer.encode("yes", add_special_tokens=False)[0]

    class ForceEnd(LogitsProcessor):
        def __call__(self, input_ids, scores):
            step = input_ids.shape[1]-width
            scores.fill_(-float("inf"))
            scores[0,151645 if step >= 3 else ordinary] = 0
            scores[1,151643 if step >= 5 else ordinary] = 0
            return scores

    with torch.inference_mode():
        controlled = model.generate(**batch, do_sample=False, max_new_tokens=16,
             eos_token_id=stops, pad_token_id=tokenizer.pad_token_id,
             logits_processor=LogitsProcessorList([ForceEnd()]))[:,width:].tolist()
    assert controlled[0][3] == 151645 and controlled[1][5] == 151643
    assert len(controlled[0]) == len(controlled[1]) == 6
    assert all(t == tokenizer.pad_token_id for t in controlled[0][4:])
    audit = {"status": "passed", "checkpoint": str(cp), "stop_ids": stops,
             "checkpoint_files": generation.checkpoint_hashes(cp),
             "evaluator_sha256": generation.support.sha(generation.HERE/"evaluate_generation.py"),
             "controlled_batched_eos_check": "passed: native turn-end and base EOS stop at expected lengths",
             "natural_outputs_are_diagnostics": True, "records": records}
    generation.support.write_json(Path(args.output), audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__": main()
