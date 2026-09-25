from common import *
from modeling import build,frozen_digest
from run import evaluate
from transformers import AutoTokenizer
from metric_audit import audit_metrics

def main():
 for rel,h in read(HERE/'CODE_FROZEN.json')['files'].items():assert sha(HERE/rel)==h
 result=[]
 for s in read(HERE/'BASE_JOBS.json'):
  model,a=build(s);assert a['trainable_parameters']==0
  before=frozen_digest(model);summaries={};task=s['task']
  tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[s['model']]['path'],local_files_only=True)
  for split in ['dev','test']:
   for rel in [f'data/{task}_{split}.jsonl',f"tokens/{task}_{s['model']}_{split}.json"]:
    path=HERE/rel;assert sha(path)==read(HERE/'INPUT_AUDIT.json')['files'][str(path)]
   summary=evaluate(model,dict(s,eval_split=split),split)
   ep=HERE/'evaluations'/s['name']/split;rs=rows(ep/'responses.jsonl');ds=rows(HERE/f'data/{task}_{split}.jsonl');labels=read(HERE/f'data/{task}_labels.json')
   assert [r['id'] for r in rs]==[r['id'] for r in ds]
   for r in rs:assert tok.decode(r['token_ids'],skip_special_tokens=False)==r['text']
   audit_metrics(rs,ds,summary,labels);summaries[split]=summary
  assert frozen_digest(model)==before
  result.append(dict(model=s['model'],task=task,summaries=summaries,status='passed'))
  write(HERE/'BASE_RESULTS.json',dict(at=now(),models=result))
  del model
  import gc,torch
  gc.collect();torch.cuda.empty_cache()
 write(HERE/'BASE_AUDIT.json',dict(at=now(),status='passed',model_tasks=len(result),metrics='independent parser/sklearn and token decoding',base_frozen=True))
if __name__=='__main__':main()
