from common import *
from transformers import AutoTokenizer
from plan import spec,variant
from scoring import score,aggregate
from label_parser import parse_label
from sklearn.metrics import f1_score,accuracy_score
import collections,pyarrow.parquet as pq

def main():
 raw=read(HERE/'raw/data_full.json');labels=sorted({label for text,label in raw['train']});assert len(labels)==150
 write(HERE/'data/clinc150_labels.json',labels)
 source=Path(read(HERE/'SOURCE.json')['implementation_source'])
 parts={}
 # Preserve original CLINC rows exactly; changes are scoring and search, not datasets.
 for split in ['train','dev','test']:
  ds=rows(source/f'data/clinc150_{split}.jsonl');parts['clinc150',split]=ds;write_rows(HERE/f'data/clinc150_{split}.jsonl',ds)
 emotion_labels=['sadness','joy','love','anger','fear','surprise'];write(HERE/'data/emotion_labels.json',emotion_labels)
 seen=set();excluded=[]
 for split,rawsplit in [('test','test'),('dev','validation'),('train','train')]:
  ds=[]
  for i,r in enumerate(pq.read_table(HERE/f'raw/emotion_{rawsplit}.parquet').to_pylist()):
   key=' '.join(r['text'].casefold().split());rid=f'emotion-{split}-{i}'
   if key in seen:excluded.append(dict(id=rid,reason='normalized_text_duplicate'));continue
   seen.add(key);ds.append(dict(id=rid,task='emotion',source_split=rawsplit,text=r['text'],target=emotion_labels[r['label']]))
  ds.sort(key=lambda r:htext('emotion-generalization-20260925:'+r['id']))
  if split=='train':ds=ds[:2048]
  assert set(r['target'] for r in ds)==set(emotion_labels)
  parts['emotion',split]=ds;write_rows(HERE/f'data/emotion_{split}.jsonl',ds)
 for task in TASKS:
  norms=[{' '.join(r['text'].casefold().split()) for r in parts[task,s]} for s in ['train','dev','test']]
  assert all(not norms[a]&norms[b] for a,b in [(0,1),(0,2),(1,2)])
 files={};models=read(HERE/'models.json')
 for m,c in models.items():
  for f,h in c['files'].items():assert sha(f)==h;files[f]=h
  cfg=read(Path(c['path'])/'config.json');assert cfg['tie_word_embeddings'];tok=AutoTokenizer.from_pretrained(c['path'],local_files_only=True);eos=128001 if m.startswith('llama') else 151643;assert cfg['eos_token_id']==eos
  for (task,split),ds in parts.items():
   ts=[]
   for row in ds:
    x=tok.encode(prompt(row),add_special_tokens=False)
    if m.startswith('llama'):x=[128000]+x
    y=tok.encode(canonical(row['target']),add_special_tokens=False)+[eos]
    assert len(x)+len(y)<=4096 and len(y)<32
    ts.append(dict(id=row['id'],prompt_ids=x,target_ids=y))
   tp=HERE/f'tokens/{task}_{m}_{split}.json';write(tp,ts);files[str(tp)]=sha(tp)
  print('verified',m,flush=True)
 for task,split in parts:
  dp=HERE/f'data/{task}_{split}.jsonl';files[str(dp)]=sha(dp)
 # Meaningful acceptance/rejection cases, without any target supplied to parser.
 for text,expected,strict in [('"joy"','joy',True),('joy','joy',False),(' Emotion: joy\nEmotion: joy','joy',False),('joy\nsadness',None,False),('joy because happy',None,False),('enjoy',None,False),('JOY',None,False),('"other"',None,False),('',None,False),('null',None,False)]:assert parse_label(text,emotion_labels)==(expected,strict)
 for task in TASKS:
  ds=parts[task,'dev'];labels=read(HERE/f'data/{task}_labels.json');probes=[]
  for i,r in enumerate(ds):
   text=canonical(r['target']) if i%4==0 else r['target'] if i%4==1 else 'invalid text' if i%4==2 else canonical(labels[(labels.index(r['target'])+1)%len(labels)])
   probes.append(score(text,r))
  y=[r['gold_label'] for r in probes];pred=[r['predicted_label'] or '__invalid__' for r in probes];got=aggregate(probes,task)
  assert abs(got['primary']-100*accuracy_score(y,pred))<1e-9
  assert abs(got['macro_f1']-100*f1_score(y,pred,labels=labels,average='macro',zero_division=0))<1e-9
 write(HERE/'DATA_AUDIT.json',dict(at=now(),status='passed',source=read(HERE/'SOURCE.json'),raw_hashes={str(f.relative_to(HERE)):sha(f) for f in (HERE/'raw').iterdir()},counts={f'{t}/{s}':len(v) for (t,s),v in parts.items()},emotion_excluded=excluded,scorer_checks='parser adversarial cases and sklearn accuracy/macroF1 passed',class_counts={f'{t}/{s}':dict(collections.Counter(r['target'] for r in v)) for (t,s),v in parts.items()}))
 write(HERE/'INPUT_AUDIT.json',dict(at=now(),status='passed',files=files))
 tuning=[];jobs=[];templates=[];smokes=[];base=[]
 for m in models:
  for task in TASKS:
   start=TASK_SETTINGS[task]['seed_start']
   for v in [variant('affine','shared',16),variant('vocab','shared',1)]:
    for ratio in [.00390625,.015625,.0625,.25,1.,4.,16.]:
     for seed in range(start,start+3):tuning.append(spec(m,task,v,seed,'tuning',ratio))
    for seed in range(start+100000,start+100005):templates.append(spec(m,task,v,seed,'confirmation'))
   for seed in range(start+100000,start+100005):jobs.append(spec(m,task,variant('none','none',0),seed,'confirmation'))
   for v in [variant('affine','shared',16),variant('vocab','shared',1),variant('none','none',0)]:
    s=spec(m,task,v,start+200000,'smoke');s['smoke']=True;smokes.append(s)
   s=spec(m,task,variant('none','none',0),0,'base');s.update(arm='base',expected_total_parameters=0);base.append(s)
 for f,x in [('TUNING_JOBS.json',tuning),('FORMAL_JOBS.json',jobs),('PENDING_FORMAL.json',jobs),('CONFIRMATION_TEMPLATES.json',templates),('SMOKE_JOBS.json',smokes),('BASE_JOBS.json',base)]:write(HERE/f,x)
 alljobs=tuning+jobs+templates+smokes
 assert len(alljobs)==480
 write(HERE/'PLAN_COUNTS.json',dict(training_total=len(alljobs),dev_only_tuning=len(tuning),confirmation=len(jobs+templates),smokes=len(smokes),base_evaluations=len(base),checkpoint_GiB=sum(s['expected_total_parameters']*8 for s in alljobs)/2**30))
 print(read(HERE/'PLAN_COUNTS.json'),flush=True)
if __name__=='__main__':main()
