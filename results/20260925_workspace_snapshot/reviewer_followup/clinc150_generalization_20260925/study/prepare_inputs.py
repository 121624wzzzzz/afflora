from common import *
from transformers import AutoTokenizer
from plan import spec,variant
from scoring import score,aggregate
from sklearn.metrics import f1_score,accuracy_score
import collections

def main():
 raw=read(HERE/'raw/data_full.json');labels=sorted({label for text,label in raw['train']});assert len(labels)==150
 write(HERE/'data/labels.json',labels);seen=set();parts={};excluded=[]
 for split,key in [('test','test'),('dev','val'),('train','train')]:
  by=collections.defaultdict(list)
  for i,(text,label) in enumerate(raw[key]):
   assert label in labels
   norm=' '.join(text.casefold().split());rid=f'clinc150-{split}-{i}'
   if norm in seen:excluded.append(dict(id=rid,reason='normalized_text_duplicate'));continue
   seen.add(norm);by[label].append(dict(id=rid,task='clinc150',text=text,target=label,source_split=key))
  selected=[];extra=set(sorted(labels,key=lambda s:htext('extra:'+s))[:98])
  for label in labels:
   values=sorted(by[label],key=lambda r:htext('clinc150-generalization-20260925:'+r['id']))
   n=(13+int(label in extra)) if split=='train' else 10 if split=='dev' else len(values)
   assert len(values)>=n;selected+=values[:n]
  parts[split]=sorted(selected,key=lambda r:htext('order:'+r['id']));write_rows(HERE/f'data/clinc150_{split}.jsonl',parts[split])
 assert len(parts['train'])==2048 and len(parts['dev'])==1500
 files={};models=read(HERE/'models.json')
 for m,c in models.items():
  for f,h in c['files'].items():assert sha(f)==h;files[f]=h
  cfg=read(Path(c['path'])/'config.json');assert cfg['tie_word_embeddings'];tok=AutoTokenizer.from_pretrained(c['path'],local_files_only=True);eos=128001 if m.startswith('llama') else 151643;assert cfg['eos_token_id']==eos
  for split,ds in parts.items():
   ts=[]
   for row in ds:
    x=tok.encode(prompt(row),add_special_tokens=False)
    if m.startswith('llama'):x=[128000]+x
    y=tok.encode(canonical(row['target']),add_special_tokens=False)+[eos]
    assert len(x)+len(y)<=4096 and len(y)<32
    ts.append(dict(id=row['id'],prompt_ids=x,target_ids=y))
   tp=HERE/f'tokens/clinc150_{m}_{split}.json';write(tp,ts);files[str(tp)]=sha(tp)
  print('verified',m,flush=True)
 for split in parts:
  dp=HERE/f'data/clinc150_{split}.jsonl';files[str(dp)]=sha(dp)
 # Gold, wrong-label and malformed-output checks against independent sklearn metrics.
 probes=[score(canonical(r['target']),r) for r in parts['dev']]
 assert aggregate(probes,'clinc150')['primary']==100
 for i in range(0,len(probes),7):probes[i]=score('not JSON',parts['dev'][i])
 for i in range(1,len(probes),11):probes[i]=score(canonical(labels[(labels.index(parts['dev'][i]['target'])+1)%150]),parts['dev'][i])
 y=[r['gold_label'] for r in probes];pred=[r['predicted_label'] or '__invalid__' for r in probes];got=aggregate(probes,'clinc150')
 assert abs(got['primary']-100*accuracy_score(y,pred))<1e-9
 assert abs(got['macro_f1']-100*f1_score(y,pred,labels=labels,average='macro',zero_division=0))<1e-9
 write(HERE/'DATA_AUDIT.json',dict(at=now(),status='passed',source=read(HERE/'SOURCE.json'),raw_sha256=sha(HERE/'raw/data_full.json'),counts={s:len(v) for s,v in parts.items()},excluded=excluded,scorer_sklearn_checks='passed',source_class_counts={s:dict(collections.Counter(r['target'] for r in ds)) for s,ds in parts.items()}))
 write(HERE/'INPUT_AUDIT.json',dict(at=now(),status='passed',files=files))
 tuning=[];jobs=[];templates=[];smokes=[];base=[]
 for m in models:
  for v in [variant('affine','shared',16),variant('vocab','shared',1)]:
   for ratio in [.015625,.0625,.25,1.]:
    for seed in range(807100,807103):tuning.append(spec(m,'clinc150',v,seed,'tuning',ratio))
   for seed in range(907100,907105):templates.append(spec(m,'clinc150',v,seed,'confirmation'))
  for seed in range(907100,907105):jobs.append(spec(m,'clinc150',variant('none','none',0),seed,'confirmation'))
  for v in [variant('affine','shared',16),variant('vocab','shared',1),variant('none','none',0)]:
   s=spec(m,'clinc150',v,1007100,'smoke');s['smoke']=True;smokes.append(s)
  s=spec(m,'clinc150',variant('none','none',0),0,'base');s.update(arm='base',expected_total_parameters=0);base.append(s)
 for f,x in [('TUNING_JOBS.json',tuning),('FORMAL_JOBS.json',jobs),('PENDING_FORMAL.json',jobs),('CONFIRMATION_TEMPLATES.json',templates),('SMOKE_JOBS.json',smokes),('BASE_JOBS.json',base)]:write(HERE/f,x)
 write(HERE/'PLAN_COUNTS.json',dict(training_total=84,dev_only_tuning=48,confirmation=30,smokes=6,base_evaluations=2,checkpoint_GiB=sum(s['expected_total_parameters']*8 for s in tuning+jobs+templates+smokes)/2**30))
 print(read(HERE/'PLAN_COUNTS.json'),flush=True)
if __name__=='__main__':main()
