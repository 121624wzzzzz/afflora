from common import *
from prepare import official_engine
from transformers import AutoTokenizer
from scoring import score
from plan import spec,variant

def main():
 old=Path(read(HERE/'SOURCE.json')['implementation_source']);root=HERE.parent.parent
 ids=set();prompts=set();scanned={}
 for p in root.rglob('wikisql_*.jsonl'):
  if 'data' not in p.parts or p.is_symlink() or p.is_relative_to(HERE):continue
  rs=rows(p)
  if not rs or not all(r.get('task')=='wikisql' and 'question' in r for r in rs):continue
  scanned[str(p)]=sha(p)
  for r in rs:ids.add(r['id']);prompts.add(' '.join(prompt(r).casefold().split()))
 assert len(scanned)>0 and len(ids)>=2048+256+1024
 raw=root/'downstream_transfer_20260917/raw/wikisql/data'
 tables={r['id']:r for r in rows(raw/'dev.tables.jsonl')};candidates=[]
 tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')['qwen3_4b_base']['path'],local_files_only=True)
 seen=set(prompts)
 for i,r in enumerate(rows(raw/'dev.jsonl')):
  t=tables[r['table_id']];g=dict(id=f'wikisql-dev-{i}',task='wikisql',source_split='dev',table_id=r['table_id'],question=r['question'],header=t['header'],types=t['types'],target=r['sql'])
  key=' '.join(prompt(g).casefold().split())
  if g['id'] in ids or key in seen:continue
  tokens=dict(id=g['id'],prompt_ids=tok.encode(prompt(g),add_special_tokens=False),target_ids=tok.encode(canonical(g['target']),add_special_tokens=False)+[151643])
  if len(tokens['prompt_ids'])+len(tokens['target_ids'])>4096 or len(tokens['target_ids'])>256:continue
  seen.add(key);candidates.append((g,tokens))
 candidates.sort(key=lambda x:htext('validation-stability-20260924:'+x[0]['id']));assert len(candidates)>=1024
 selected=candidates[:1024];engine=official_engine('dev');from lib.query import Query
 for g,t in selected:
  g['gold_execution']=engine.execute_query(g['table_id'],Query.from_dict(g['target']),lower=True)
  assert score(canonical(g['target']),g)['content_correct']
 write_rows(HERE/'data/wikisql_dev.jsonl',[g for g,t in selected]);write(HERE/'tokens/wikisql_qwen3_4b_base_dev.json',[t for g,t in selected])
 write(HERE/'VALIDATION_DATA_AUDIT.json',dict(at=now(),status='passed',n=1024,eligible=len(candidates),excluded_known_ids=len(ids),scanned_prepared_files=scanned,source_sha256={str(raw/f):sha(raw/f) for f in ['dev.jsonl','dev.tables.jsonl']},note='No ID or normalized-prompt overlap with scanned locally prepared data; not a universal historical exposure guarantee.'))
 manifest=read(old/'OUTPUT_MANIFEST.json')['files'];assert sha(old/'OUTPUT_MANIFEST.json')==read(old/'FINAL_AUDIT.json')['manifest_sha256']
 for rel,h in read(old/'CODE_FROZEN.json')['files'].items():assert sha(old/rel)==h
 tuning=[];reuse=[]
 for s in read(old/'TUNING_JOBS.json'):
  if s['task']!='wikisql':continue
  cp=old/'checkpoints'/s['name'];a=read(old/'audits'/f"{s['name']}.json");assert a['status']=='passed' and set(a['summary_sha256'])=={'dev'}
  files={}
  for rel in ['spec.json','INITIALIZATION.json','TRAINING.json','TRAIN_ORDER.json','initial_adapter.safetensors','adapter.safetensors']:
   f=cp/rel;h=sha(f);assert h==manifest[str(f.relative_to(old))];files[str(f)]=h
  ep=old/'evaluations'/s['name']/'dev';assert sha(ep/'SUMMARY.json')==a['summary_sha256']['dev'];assert sha(ep/'responses.jsonl')==read(ep/'SUMMARY.json')['responses_sha256'];assert not (ep.parent/'test').exists()
  tuning.append(dict(s,evaluation_only=True,source_checkpoint=str(cp),source_hashes=files));reuse.append(dict(name=s['name'],files=files,source_audit_sha256=sha(old/'audits'/f"{s['name']}.json")))
 templates=[];jobs=[];smokes=[]
 for v in [variant('affine','shared',16),variant('affine','shared',30),variant('vocab','shared',1),variant('vocab','shared',2),variant('none','none',0)]:
  for seed in range(607100,607105):
   s=spec('qwen3_4b_base','wikisql',v,seed,'confirmation');(jobs if v['method']=='none' else templates).append(s)
  s=spec('qwen3_4b_base','wikisql',v,707100,'smoke');s['smoke']=True;smokes.append(s)
 for name,x in [('TUNING_JOBS.json',tuning),('CONFIRMATION_TEMPLATES.json',templates),('FORMAL_JOBS.json',jobs),('PENDING_FORMAL.json',jobs),('SMOKE_JOBS.json',smokes),('CHECKPOINT_REUSE_AUDIT.json',reuse)]:write(HERE/name,x)
 assert len(tuning)==72
 write(HERE/'PLAN_COUNTS.json',dict(training_total=102,evaluation_only=72,fresh_formal=25,smokes=5,reused_results=0,new_checkpoint_GiB=sum(s['expected_total_parameters']*8 for s in jobs+templates+smokes)/2**30))
 print(read(HERE/'PLAN_COUNTS.json'),flush=True)
if __name__=='__main__':main()
