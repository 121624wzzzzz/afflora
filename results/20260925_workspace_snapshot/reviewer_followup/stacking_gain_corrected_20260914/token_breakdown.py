"""Post-hoc diagnostic: separate supervised im_end from all other token NLL."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
import evaluate_generation as generation

HERE = Path(__file__).resolve().parent
OUT = HERE / "token_breakdown"
OLD = HERE.parent / "placement_capacity_corrected_20260911"
STATE = {}


def update(**fields):
    STATE.update(fields, updated_at=generation.support.now())
    generation.support.write_json(OUT/"state.json", STATE)


@torch.inference_mode()
def evaluate(name, cp):
    expected = json.loads((HERE/"reports"/f"{name}.test.json").read_text())
    hashes = generation.checkpoint_hashes(cp)
    model, tokenizer, _, _ = generation.ce.load_model(argparse.Namespace(
        run_dir=str(cp),model_path=None,affine_ablation="none",device="cuda"))
    rows = generation.ce.load_rows(HERE/"data/test.jsonl")
    assert len(rows) == expected["num_examples"] == 1000
    end_ids = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    assert end_ids == [151645]
    details = []
    for i, (row, old) in enumerate(zip(rows,expected["per_example"])):
        item = generation.ce.tokenize_conversation(row,tokenizer,1024)
        ids, attention, labels = generation.ce.collate([item],tokenizer.pad_token_id,torch.device("cuda"))
        logits = model(input_ids=ids,attention_mask=attention).logits
        shifted = labels[:,1:]
        losses = F.cross_entropy(logits[:,:-1,:].float().transpose(1,2),shifted,
                                 ignore_index=-100,reduction="none")
        masks = {"total":shifted.ne(-100),"end":shifted.eq(151645),
                 "non_end":shifted.ne(-100)&shifted.ne(151645)}
        result = {"record_id":row["record_id"]}
        for key,mask in masks.items():
            result[key+"_nll"] = float((losses*mask).sum().cpu())
            result[key+"_tokens"] = int(mask.sum().cpu())
        assert result["record_id"] == old["record_id"]
        assert result["total_tokens"] == old["token_count"]
        assert abs(result["total_nll"]-old["nll_sum"])/old["token_count"] <= 5e-5
        assert result["end_tokens"]+result["non_end_tokens"] == result["total_tokens"]
        details.append(result)
        del logits, losses
        if (i+1)%100 == 0:
            update(current=name,examples=i+1)
            print(name,i+1,"/1000",flush=True)
    totals = {key:sum(r[key] for r in details) for key in details[0] if key != "record_id"}
    metrics = {key+"_ce":totals[key+"_nll"]/totals[key+"_tokens"]
               for key in ("total","end","non_end")}
    metrics["end_token_fraction"] = totals["end_tokens"]/totals["total_tokens"]
    metrics["replay_avg_ce_delta"] = metrics["total_ce"]-expected["avg_ce"]
    assert abs(metrics["replay_avg_ce_delta"]) <= 2e-6
    assert generation.checkpoint_hashes(cp) == hashes
    result = {"name":name,"checkpoint":str(cp),"checkpoint_files":hashes,
              "diagnostic_manifest_sha256":generation.support.sha(OUT/"manifest.json"),
              "totals":totals,"metrics":metrics,"per_example":details}
    generation.support.write_json(OUT/f"{name}.json",result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    used = int(subprocess.check_output(["nvidia-smi","--id="+gpu,
        "--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
    assert used < 12000, "Only share a lightly loaded GPU with our own generation job"
    live = json.loads((HERE/"state.json").read_text())
    assert any(j.get("gpu")==gpu and j.get("status")=="ifeval" and n.startswith("qwen3_")
               for n,j in live["jobs"].items())
    torch.cuda.set_per_process_memory_fraction(.2)
    cases = [(f"qwen3_06b_{arm}_hr8_sd42", (OLD if arm in ("none","output") else HERE)/
              "checkpoints"/f"qwen3_06b_{arm}_hr8_sd42") for arm in ("none","output","both","hidden_budget")]
    assert not (OUT/"manifest.json").exists(), "Diagnostic already initialized"
    generation.support.write_json(OUT/"manifest.json",{
        "created_at":generation.support.now(),"type":"post-hoc diagnostic; not a new confirmatory endpoint",
        "question":"Does CE improvement come from supervised im_end or the remaining supervised tokens?",
        "models":[n for n,_ in cases],"test_examples":1000,"batch_size":1,"max_seq_len":1024,
        "main_protocol_sha256":generation.support.sha(HERE/"manifest.json"),
        "files":{str(p):generation.support.sha(p) for p in
                 (Path(__file__),HERE/"data/test.jsonl",HERE/"source/corrected_sft_experiment/evaluate_corrected_sft.py",
                  HERE/"source/corrected_sft_experiment/data_pipeline.py")},
        "checkpoints":{n:generation.checkpoint_hashes(cp) for n,cp in cases}})
    update(phase="running",pid=os.getpid(),gpu=gpu,completed=[])
    results = []
    for name,cp in cases:
        update(current=name,examples=0)
        results.append(evaluate(name,cp))
        update(completed=[r["name"] for r in results])
    baseline = results[0]
    contrasts = {}
    for value in results[1:]:
        assert all(value["totals"][key+"_tokens"]==baseline["totals"][key+"_tokens"]
                   for key in ("total","end","non_end"))
        total = baseline["totals"]["total_tokens"]
        contrasts[value["name"]] = {
            "delta_total_ce":value["metrics"]["total_ce"]-baseline["metrics"]["total_ce"],
            "end_contribution_to_delta_ce":(value["totals"]["end_nll"]-baseline["totals"]["end_nll"])/total,
            "non_end_contribution_to_delta_ce":(value["totals"]["non_end_nll"]-baseline["totals"]["non_end_nll"])/total,
            "delta_non_end_ce":value["metrics"]["non_end_ce"]-baseline["metrics"]["non_end_ce"]}
    generation.support.write_json(OUT/"summary.json",{"status":"passed","baseline":baseline["metrics"],
        "models":{r["name"]:r["metrics"] for r in results},"contrasts":contrasts})
    update(phase="complete",current=None,examples=1000)


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        update(phase="failed",error=repr(exc))
        raise
