import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;S=R/'study';O=R.parent/'model_architecture_14h_20260917/core'
sys.path.insert(0,str(S))
from common import *
from transformers import AutoTokenizer
from scoring import score,aggregate
from prepare import official_engine
models=sorted({j['model'] for j in read(S/'FORMAL_JOBS.json')});seal=read(O.parent/'SEAL_MANIFEST.json')['files'];checked={}
def verify(p):
 rel=str(p.relative_to(O.parent));h=sha(p);assert h==seal[rel]['sha256'],rel;checked[rel]=h
for p in [O/'models.json',O/'BUDGET_PLAN.json',O/'ANCHOR_INDEX.json',O/'DATA_FROZEN.json']:verify(p)
for rel,h in read(O/'DATA_FROZEN.json')['files'].items():
 if rel.startswith(('data/','raw/')) or (rel.startswith('tokens/') and any(m in rel for m in models)):
  assert sha(O/rel)==h,rel
print('data hashes checked',flush=True)
configs=read(S/'models.json');modelchecked={};tokenchecked=0
for m in models:
 for path,h in configs[m]['files'].items():assert sha(Path(path))==h,path
 modelchecked[m]=configs[m]['files'];print('model identity checked',m,flush=True)
 tok=AutoTokenizer.from_pretrained(configs[m]['path'],local_files_only=True)
 for task in TASKS:
  for split in ['train','dev','test']:
   data=rows(S/f'data/{task}_{split}.jsonl');tokens=read(S/f'tokens/{task}_{m}_{split}.json')
   rebuilt=[dict(id=r['id'],prompt_ids=tok.encode(prompt(r),add_special_tokens=False),target_ids=tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]) for r in data]
   assert rebuilt==tokens;tokenchecked+=len(data)
write(S/'MODEL_IDENTITY_AUDIT.json',dict(status='passed',models=modelchecked))
anchors=read(O/'ANCHOR_INDEX.json');reused=[]
for m in models:
 tok=AutoTokenizer.from_pretrained(configs[m]['path'],local_files_only=True)
 for task in TASKS:
  data=rows(S/f'data/{task}_test.jsonl');tokens=read(S/f'tokens/{task}_{m}_test.json');engine=official_engine('test') if task=='wikisql' else None
  if engine:
   from lib.query import Query
   from scoring_sql import parse
  arms=['base','hidden','hidden_budget','hidden_both']+(['hidden_input','hidden_output','input','output','both'] if not configs[m]['tie_word_embeddings'] else [])
  for arm in arms:
   for seed in ([None] if arm=='base' else range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+3)):
    n=f'{task}_{m}_{arm}'+('' if seed is None else f'_s{seed}');ck=O/'checkpoints'/n;out=O/'evaluations'/n/'test'
    if not (out/'SUMMARY.json').exists():
     a=next(a for a in anchors if (a['task'],a['model'],a['arm'],a['seed'])==(task,m,arm,seed));out=O/a['directory'];ck=Path(a['source_checkpoint'])
     src=Path(a['source']);manifest=src/'ARTIFACT_MANIFEST.json';assert sha(manifest)==a['source_manifest_sha256'];original=read(manifest)['files']
     for p in ck.rglob('*'):
      if p.is_file() and '__pycache__' not in p.parts:assert sha(p)==original[str(p.relative_to(src))]['sha256'],str(p)
    else:
     for p in ck.rglob('*'):
      if p.is_file() and '__pycache__' not in p.parts:verify(p)
    for p in out.rglob('*'):
     if p.is_file() and '__pycache__' not in p.parts:verify(p)
    summary=read(out/'SUMMARY.json');rs=rows(out/'responses.jsonl');assert sha(out/'responses.jsonl')==summary['responses_sha256'];assert [r['id'] for r in rs]==[r['id'] for r in data]
    assert summary['input_ids_sha256']==htext(canonical([r['prompt_ids'] for r in tokens]))
    init=read(ck/'INITIALIZATION.json');assert init['trainable_parameters']==read(S/'BUDGET_PLAN.json')[m][arm]
    if arm!='base':
     tr=read(ck/'TRAINING.json');assert tr['frozen_before']==tr['frozen_after'] and tr['reload_loss_error']==0 and tr['steps']==64;assert sha(ck/'adapter.safetensors')==tr['adapter_sha256']
    for r,g in zip(rs,data):
     assert tok.decode(r['token_ids'],skip_special_tokens=False)==r['text']
     for k,v in score(r['text'],g).items():assert r[k]==v
     if engine and r['query_valid']:
      q,_=parse(r['text'])
      try:ref=engine.execute_query(g['table_id'],Query.from_dict(q),lower=True)==g['gold_execution']
      except Exception:ref=False
      assert ref==r['content_correct']
    got=aggregate(rs,task);assert got=={k:summary[k] for k in got}
    reused.append(dict(model=m,task=task,arm=arm,seed=seed,primary=summary['primary'],summary=str(out/'SUMMARY.json'),checkpoint=str(ck)))
  print('reuse rescored',m,task,flush=True)
write(S/'REUSE_AUDIT.json',dict(status='passed',source_manifest_sha256=sha(O.parent/'SEAL_MANIFEST.json'),verified_files=checked,token_records_reencoded=tokenchecked,runs=reused))
write(S/'DATA_FROZEN.json',dict(files={str(p.relative_to(S)):sha(p) for folder in ['data','tokens','raw'] for p in (S/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}))
print('VERIFIED',len(reused),flush=True)
