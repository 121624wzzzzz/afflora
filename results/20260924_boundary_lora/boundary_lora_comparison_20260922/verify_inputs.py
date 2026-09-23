import sys,collections
from transformers import AutoTokenizer
from common import *
from scoring import score,aggregate
from prepare import official_engine

def main():
 files={};counts={};models=read(HERE/'models.json')
 for m,cfg in models.items():
  for path,h in cfg['files'].items():
   assert sha(path)==h,path;files[path]=h
  tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True);eos=128001 if m.startswith('llama') else 151643
  for task in TASKS:
   for split in ['train','dev','test']:
    dp=HERE/f'data/{task}_{split}.jsonl';tp=HERE/f'tokens/{task}_{m}_{split}.json';data=rows(dp);tokens=read(tp)
    assert len(data)==len(tokens)
    for row,t in zip(data,tokens):
     assert row['id']==t['id'];expected=tok.encode(prompt(row),add_special_tokens=False)
     if m.startswith('llama'):expected=[128000]+expected
     assert expected==t['prompt_ids'],(m,task,split,row['id'],'prompt')
     assert t['target_ids'][-1]==eos
    files[str(dp)]=sha(dp);files[str(tp)]=sha(tp);counts[f'{m}/{task}/{split}']=len(data)
  print('verified model and token inputs',m,flush=True)
 write(HERE/'INPUT_AUDIT.json',dict(status='passed',at=now(),files=files,counts=counts))
 # Reuse is conservative: only a matching full-scope configuration with intact checkpoint + test responses.
 base=HERE.parent.parent; candidates=[]
 for name in ['qwen_shared_rank8_20260921','qwen_shared_20260921']:
  old=base/name/'study'
  for r in read(old/'REUSE_AUDIT.json')['runs']:
   candidates.append((Path(r['checkpoint']),Path(r['summary']),16))
  for s in read(old/'FORMAL_JOBS.json'):
   if (old/'audits'/f"{s['name']}.json").exists():candidates.append((old/'checkpoints'/s['name'],old/'evaluations'/s['name']/'test/SUMMARY.json',8 if 'rank8' in name else 16))
 core=base/'model_architecture_14h_20260917/core'
 for a in read(core/'ANCHOR_INDEX.json'):candidates.append((Path(a['source_checkpoint']),Path(a['source'])/a['source_summary'],16))
 ll=base/'llama_sizes_20260921/study'
 # The local formal paths include only newly trained fits; historical paths come from source reuse metadata if available.
 for cp in (ll/'checkpoints').glob('*'):
  candidates.append((cp,ll/'evaluations'/cp.name/'test/SUMMARY.json',16))
 index={}
 for cp,sp,default_rank in candidates:
  try:
   summary=read(sp);s=summary['spec'];arm=s['arm']
   if s.get('smoke') or s.get('lr')!=2e-4:continue
   if arm=='hidden':v=('none','none',0,False)
   elif arm.startswith('hidden_shared'):v=('affine','shared',int(arm.replace('hidden_shared','')),True)
   elif arm in ['hidden_input','hidden_output','hidden_both']:v=('affine',{'hidden_input':'e','hidden_output':'u','hidden_both':'eu'}[arm],default_rank,arm!='hidden_output')
   else:continue
   key=(s['model'],s['task'],*v,s['seed']);index.setdefault(key,(cp,sp))
  except (FileNotFoundError,KeyError):continue
 reused=[];pending=[];hashed={}
 for s in read(HERE/'FORMAL_JOBS.json'):
  key=(s['model'],s['task'],s['method'],s['placement'],s['rank'],s['bias'],s['seed'])
  if s['stage']!='common' or key not in index:pending.append(s);continue
  cp,sp=index[key]
  try:
   init=read(cp/'INITIALIZATION.json');tr=read(cp/'TRAINING.json');rs=rows(sp.parent/'responses.jsonl');summary=read(sp)
   assert tr['frozen_before']==tr['frozen_after'] and tr['reload_loss_error']==0
   assert init['trainable_parameters']==s['expected_total_parameters']
   assert sha(cp/'adapter.safetensors')==tr['adapter_sha256']
   assert sha(sp.parent/'responses.jsonl')==summary['responses_sha256']
   data=rows(HERE/f"data/{s['task']}_test.jsonl");tokens=read(HERE/f"tokens/{s['task']}_{s['model']}_test.json")
   assert summary['input_ids_sha256']==htext(canonical([r['prompt_ids'] for r in tokens]))
   assert [r['id'] for r in data]==[r['id'] for r in rs]
   if s['task']=='wikisql':
    engine=official_engine('test');from lib.query import Query
    from scoring_sql import parse
   for r,g in zip(rs,data):
    for k,value in score(r['text'],g).items():assert r[k]==value
    if s['task']=='wikisql' and r['query_valid']:
     q,_=parse(r['text'])
     try:correct=engine.execute_query(g['table_id'],Query.from_dict(q),lower=True)==g['gold_execution']
     except Exception:correct=False
     assert correct==r['content_correct']
   agg=aggregate(rs,s['task']);assert all(summary[k]==v for k,v in agg.items())
   for f in [cp/'INITIALIZATION.json',cp/'TRAINING.json',cp/'adapter.safetensors',sp,sp.parent/'responses.jsonl']:
    hashed[str(f)]=sha(f)
   reused.append(dict(spec=s,checkpoint=str(cp),summary=str(sp),primary=summary['primary'],hidden_initialization_sha256=init['shared_hidden_initialization_sha256']))
  except (FileNotFoundError,KeyError,AssertionError) as e:
   print('not reusable; schedule new',s['name'],str(e),flush=True);pending.append(s)
 write(HERE/'REUSE_AUDIT.json',dict(status='passed',at=now(),runs=reused,files=hashed));write(HERE/'PENDING_FORMAL.json',pending)
 print('reuse',len(reused),'new formal',len(pending),flush=True)
 write(HERE/'READY.json',dict(status='passed',at=now(),reused=len(reused),pending_formal=len(pending)))
if __name__=='__main__':main()
