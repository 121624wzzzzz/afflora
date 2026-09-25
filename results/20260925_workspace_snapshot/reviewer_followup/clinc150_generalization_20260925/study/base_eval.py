"""Read-only base evaluation with independent label-metric audit."""
from common import *
from modeling import build,frozen_digest
from run import evaluate
from scoring import aggregate
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score

def main():
 for rel,h in read(HERE/'CODE_FROZEN.json')['files'].items():assert sha(HERE/rel)==h
 result=[]
 for s in read(HERE/'BASE_JOBS.json'):
  model,a=build(s);assert a['trainable_parameters']==0
  before=frozen_digest(model);summaries={}
  tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[s['model']]['path'],local_files_only=True)
  for split in ['dev','test']:
   for rel in [f'data/clinc150_{split}.jsonl',f"tokens/clinc150_{s['model']}_{split}.json"]:
    path=HERE/rel;assert sha(path)==read(HERE/'INPUT_AUDIT.json')['files'][str(path)]
   summary=evaluate(model,dict(s,eval_split=split),split)
   ep=HERE/'evaluations'/s['name']/split;rs=rows(ep/'responses.jsonl');ds=rows(HERE/f'data/clinc150_{split}.jsonl');labels=read(HERE/'data/labels.json')
   assert [r['id'] for r in rs]==[r['id'] for r in ds]
   predictions=[]
   for r,g in zip(rs,ds):
    assert tok.decode(r['token_ids'],skip_special_tokens=False)==r['text']
    try:pred=json.loads(r['text'].strip())
    except ValueError:pred=None
    valid=isinstance(pred,str) and pred in labels;pred=pred if valid else '__invalid__';predictions.append(pred)
    assert r['content_correct']==(pred==g['target'])
   y=[r['target'] for r in ds]
   assert abs(summary['primary']-100*accuracy_score(y,predictions))<1e-9
   assert abs(summary['macro_f1']-100*f1_score(y,predictions,labels=labels,average='macro',zero_division=0))<1e-9
   summaries[split]=summary
  assert frozen_digest(model)==before
  result.append(dict(model=s['model'],summaries=summaries,status='passed'))
  write(HERE/'BASE_RESULTS.json',dict(at=now(),models=result))
  del model
  import gc,torch
  gc.collect();torch.cuda.empty_cache()
 write(HERE/'BASE_AUDIT.json',dict(at=now(),status='passed',models=len(result),metrics='independent sklearn and JSON parsing',base_frozen=True))
if __name__=='__main__':main()
