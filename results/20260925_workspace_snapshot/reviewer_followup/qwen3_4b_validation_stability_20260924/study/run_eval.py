"""Reload verified fixed dev-only adapter; only write new evaluation products."""
import argparse
from common import *
from modeling import build,load_adapter,adapter_state,digest
from run import evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);s=read(p.parse_args().spec)
 assert s['evaluation_only'] and s['eval_splits']==['dev']
 for rel,h in read(HERE/'CODE_FROZEN.json')['files'].items():assert sha(HERE/rel)==h
 for path,h in s['source_hashes'].items():assert sha(path)==h
 for rel in [f"data/wikisql_dev.jsonl",f"tokens/wikisql_qwen3_4b_base_dev.json"]:
  path=HERE/rel;assert sha(path)==read(HERE/'INPUT_AUDIT.json')['files'][str(path)]
 cp=Path(s['source_checkpoint']);original=read(cp/'spec.json')
 assert all(s[k]==v for k,v in original.items())
 model,audit=build(s);load_adapter(model,cp/'adapter.safetensors')
 assert digest(adapter_state(model))==read(cp/'TRAINING.json')['adapter_tensor_sha256']
 (HERE/'checkpoints'/s['name']).symlink_to(cp,target_is_directory=True)
 evaluate(model,dict(s,eval_split='dev'),'dev')
 # Source checkpoint remains immutable. Independent audit_one checks new outputs.
if __name__=='__main__':main()
