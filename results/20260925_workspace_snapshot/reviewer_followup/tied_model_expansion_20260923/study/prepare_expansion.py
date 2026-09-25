from common import *
from plan import architecture
from scoring import score,aggregate
from prepare import official_engine
from transformers import AutoTokenizer
import torch,collections

def key(s):return tuple(s[k] for k in ['stage','model','task','method','placement','rank','bias','seed'])
def main():
 sources=read(HERE/'SOURCE.json');models=read(HERE/'models.json');files={};token_count=0
 for m,c in models.items():
  cfg=read(Path(c['path'])/'config.json');assert cfg['tie_word_embeddings'] is True
  for path,h in c['files'].items():assert sha(path)==h,path;files[path]=h
  tok=AutoTokenizer.from_pretrained(c['path'],local_files_only=True);eos=128001 if m.startswith('llama') else 151643
  assert cfg['eos_token_id']==eos
  for task in TASKS:
   split_ids=[]
   for split in ['train','dev','test']:
    dp=HERE/f'data/{task}_{split}.jsonl';tp=HERE/f'tokens/{task}_{m}_{split}.json';data=rows(dp);tokens=read(tp);assert len(data)==len(tokens)
    split_ids.append({r['id'] for r in data})
    for r,t in zip(data,tokens):
     assert r['id']==t['id'];expected=tok.encode(prompt(r),add_special_tokens=False)
     if m.startswith('llama'):expected=[128000]+expected
     assert t['prompt_ids']==expected and t['target_ids']==tok.encode(canonical(r['target']),add_special_tokens=False)+[eos];token_count+=1
    files[str(dp)]=sha(dp);files[str(tp)]=sha(tp)
   assert all(not split_ids[a]&split_ids[b] for a,b in [(0,1),(0,2),(1,2)])
  print('verified weights/native tokenizer',m,flush=True)
 write(HERE/'INPUT_AUDIT.json',dict(at=now(),status='passed',files=files,token_records_reencoded=token_count))
 candidates={};inherited=[];source_hashes={}
 for label,path in sources.items():
  old=Path(path);state=read(old/'STATE.json');assert state['stage']=='complete' and not state['failed']
  for rel,h in read(old/'CODE_FROZEN.json')['files'].items():assert sha(old/rel)==h
  for r in read(old/'RESULTS.json')['records']:
   s=r['spec']
   if s['model'] not in models or s['stage'] not in ['common','confirmation']:continue
   if r['reused']:
    oldr=next(z for z in read(old/'REUSE_AUDIT.json')['runs'] if z['spec']['name']==s['name']);cp=Path(oldr['checkpoint']);summary=Path(oldr['summary'])
   else:cp=old/'checkpoints'/s['name'];summary=old/'evaluations'/s['name']/'test/SUMMARY.json'
   candidates[key(s)]=(cp,summary)
  for s in read(old/'SMOKE_JOBS.json'):
   if s['model'] not in models:continue
   ap=old/'audits'/f"{s['name']}.json"
   if not ap.exists():continue
   a=read(ap);assert a['status']=='passed';cp=old/'checkpoints'/s['name'];tr=read(cp/'TRAINING.json');assert sha(cp/'adapter.safetensors')==tr['adapter_sha256']
   for tag,h in a['summary_sha256'].items():
    ep=old/'evaluations'/s['name']/tag;assert sha(ep/'SUMMARY.json')==h and sha(ep/'responses.jsonl')==read(ep/'SUMMARY.json')['responses_sha256']
   inherited.append(list(architecture(s)));source_hashes[str(ap)]=sha(ap)
 # Historical H-only reference candidates for other sizes; accept only exact dataset/order/scope.
 core=HERE.parent.parent/'model_architecture_14h_20260917/core'
 for a in read(core/'ANCHOR_INDEX.json'):
  if a['model'] not in models or a['arm']!='hidden':continue
  s=dict(stage='common',model=a['model'],task=a['task'],method='none',placement='none',rank=0,bias=False,seed=a['seed'])
  candidates.setdefault(key(s),(Path(a['source_checkpoint']),Path(a['source'])/a['source_summary']))
 checked=[];reused=[];pending=[];train_orders={};engines={};tokcache={}
 def verify(s,cp,summary):
  tr=read(cp/'TRAINING.json');init=read(cp/'INITIALIZATION.json');summ=read(summary);saved=summ['spec']
  assert saved['lr']==s['lr'] and saved.get('boundary_lr_ratio',1)==s.get('boundary_lr_ratio',1)
  assert saved['seed']==s['seed'] and saved['microbatch']==s['microbatch']
  assert init['trainable_parameters']==s['expected_total_parameters'] and init['tied_weights']
  assert tr['steps']==64 and tr['examples']==2048 and tr['reload_loss_error']==0 and tr['frozen_before']==tr['frozen_after']
  assert tr['optimizer_whitelist_verified'] and tr['optimizer_fp32'];assert sha(cp/'adapter.safetensors')==tr['adapter_sha256']
  orderkey=(s['task'],s['seed'])
  if orderkey not in train_orders:
   ds=rows(HERE/f"data/{s['task']}_train.jsonl");order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(s['seed'])).tolist();train_orders[orderkey]=dict(indices=order,ids=[ds[i]['id'] for i in order])
  assert read(cp/'TRAIN_ORDER.json')==train_orders[orderkey]
  rs=rows(summary.parent/'responses.jsonl');data=rows(HERE/f"data/{s['task']}_test.jsonl");tokens=read(HERE/f"tokens/{s['task']}_{s['model']}_test.json")
  assert summ['responses_sha256']==sha(summary.parent/'responses.jsonl') and summ['input_ids_sha256']==htext(canonical([t['prompt_ids'] for t in tokens]))
  assert [r['id'] for r in rs]==[r['id'] for r in data]
  if s['model'] not in tokcache:tokcache[s['model']]=AutoTokenizer.from_pretrained(models[s['model']]['path'],local_files_only=True)
  tok=tokcache[s['model']]
  if s['task']=='wikisql':
   if 'test' not in engines:engines['test']=official_engine('test')
   from lib.query import Query
   from scoring_sql import parse
  for r,g in zip(rs,data):
   assert tok.decode(r['token_ids'],skip_special_tokens=False)==r['text']
   for k,v in score(r['text'],g).items():assert r[k]==v
   if s['task']=='wikisql' and r['query_valid']:
    q,_=parse(r['text'])
    try:correct=engines['test'].execute_query(g['table_id'],Query.from_dict(q),lower=True)==g['gold_execution']
    except Exception:correct=False
    assert correct==r['content_correct']
  assert all(summ[k]==v for k,v in aggregate(rs,s['task']).items())
  for f in [cp/'adapter.safetensors',cp/'INITIALIZATION.json',cp/'TRAINING.json',cp/'TRAIN_ORDER.json',summary,summary.parent/'responses.jsonl']:source_hashes[str(f)]=sha(f)
  return dict(spec=s,checkpoint=str(cp),summary=str(summary),primary=summ['primary'],hidden_initialization_sha256=init['shared_hidden_initialization_sha256'])
 for s in read(HERE/'FORMAL_JOBS.json'):
  if key(s) not in candidates:pending.append(s);continue
  cp,summary=candidates[key(s)]
  try:reused.append(verify(s,cp,summary))
  except (AssertionError,FileNotFoundError,KeyError) as e:print('rerun unmatched reference',s['name'],str(e),flush=True);pending.append(s)
 # The two small Qwen models already have the exact four-candidate/two-seed search and fresh confirmation.
 old=Path(sources['closure']);selection=read(old/'SELECTION_FROZEN.json');prior_selections=[];confirmation=[];reuse_tuning=[]
 for m in ['qwen3_06b_base','qwen25_15b_base']:
  for row in selection['selection']:
   if row['architecture'][0]!=m or row['architecture'][3]!='shared':continue
   scores=collections.defaultdict(list)
   for s in read(old/'TUNING_JOBS.json'):
    if list(architecture(s))!=row['architecture']:continue
    cp=old/'checkpoints'/s['name'];ap=old/'audits'/f"{s['name']}.json";a=read(ap);assert a['status']=='passed' and set(a['summary_sha256'])=={'dev'}
    ep=old/'evaluations'/s['name']/'dev';assert sha(ep/'SUMMARY.json')==a['summary_sha256']['dev'];summ=read(ep/'SUMMARY.json');assert sha(ep/'responses.jsonl')==summ['responses_sha256'];assert sha(cp/'adapter.safetensors')==read(cp/'TRAINING.json')['adapter_sha256'];assert not (ep.parent/'test').exists()
    scores[s['boundary_lr_ratio']].append(summ['primary']);reuse_tuning.append(s['name']);source_hashes[str(ap)]=sha(ap)
   best=max(scores,key=lambda q:(sum(scores[q])/len(scores[q]),-abs(q-1),-q));assert best==row['ratio'];prior_selections.append(row)
 for s in read(HERE/'ALL_CONFIRMATION_TEMPLATES.json'):
  if s['model'] not in ['qwen3_06b_base','qwen25_15b_base']:confirmation.append(s);continue
  row=next(x for x in prior_selections if x['architecture']==list(architecture(s)));s=dict(s,boundary_lr_ratio=row['ratio']);s['name']+=f"_selected{row['ratio']:g}"
  cp,summary=candidates[key(s)];reused.append(verify(s,cp,summary))
 tuning=[s for s in read(HERE/'ALL_TUNING_JOBS.json') if s['model'] not in ['qwen3_06b_base','qwen25_15b_base']]
 seen={tuple(k) for k in inherited};smokes=[]
 for s in pending+tuning+confirmation:
  k=architecture(s)
  if k in seen:continue
  seen.add(k);z=dict(s,name='smoke_'+s['name'],smoke=True,stage='smoke');z.pop('eval_splits',None);smokes.append(z)
 write(HERE/'REUSE_AUDIT.json',dict(at=now(),status='passed',runs=reused,files=source_hashes,prior_selections=prior_selections,reused_dev_only_runs=reuse_tuning,note='Final weights, initialization reports, order, output tokens and scoring rechecked; old missing initial tensor files are not claimed as rehashed.'))
 write(HERE/'INHERITED_SMOKES.json',dict(status='passed',architectures=inherited));write(HERE/'PENDING_FORMAL.json',pending);write(HERE/'TUNING_JOBS.json',tuning);write(HERE/'CONFIRMATION_TEMPLATES.json',confirmation);write(HERE/'SMOKE_JOBS.json',smokes)
 jobs=pending+tuning+confirmation+smokes
 write(HERE/'PLAN_COUNTS.json',dict(models=len(models),new_formal=len(pending),dev_only_tuning=len(tuning),fresh_confirmation=len(confirmation),smokes=len(smokes),training_total=len(jobs),reused_results=len(reused),reused_tuning=len(reuse_tuning),checkpoint_GiB=sum(s['expected_total_parameters']*8 for s in jobs)/2**30))
 write(HERE/'READY.json',dict(at=now(),status='passed',reused=len(reused)));print(read(HERE/'PLAN_COUNTS.json'),flush=True)
if __name__=='__main__':main()
