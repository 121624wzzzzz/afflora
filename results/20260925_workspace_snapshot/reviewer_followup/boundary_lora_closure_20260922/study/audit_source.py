"""Reconcile all completed source runs to immutable plans and per-run audits."""
from common import *
from scoring import aggregate
from plan import architecture
import collections

def audit(require_complete=False):
 old=Path(read(HERE/'SOURCE.json')['study']);state=read(old/'STATE.json');frozen=read(old/'CODE_FROZEN.json')
 for rel,h in frozen['files'].items():assert sha(old/rel)==h,rel
 planned=read(old/'SMOKE_JOBS.json')+read(old/'PENDING_FORMAL.json')+read(old/'TUNING_JOBS.json')+read(old/'SELECTION_FROZEN.json')['jobs']
 names={s['name']:s for s in planned};assert len(names)==684
 assert not state['failed'];assert len(state['done'])==len(set(state['done']))
 if require_complete:assert state['stage']=='complete' and len(state['done'])==684 and not state['active'] and not state['pending']
 checked=[];files={};responses=official=0;smokes=[];missing_initial=[]
 # Cache only previously cryptographically checked unchanged files within this review.
 cache=read(HERE/'SOURCE_HASH_CACHE.json') if (HERE/'SOURCE_HASH_CACHE.json').exists() else {}
 def verified(path,expected=None):
  st=path.stat();key=str(path);signature=[st.st_size,st.st_mtime_ns,st.st_ctime_ns]
  entry=cache.get(key);h=entry['sha256'] if entry and entry['stat']==signature else sha(path)
  if expected is not None:assert h==expected,key
  cache[key]=dict(stat=signature,sha256=h);files[key]=h;return h
 for name in state['done']:
  assert name in names;cp=old/'checkpoints'/name;s=read(cp/'spec.json');assert s==names[name]
  a=read(old/'audits'/f'{name}.json');assert a['status']=='passed' and read(cp/'COMPLETE.json')['status']=='passed'
  init=read(cp/'INITIALIZATION.json');tr=read(cp/'TRAINING.json')
  assert tr['frozen_before']==tr['frozen_after'] and tr['reload_loss_error']==0 and tr['changed_groups']==tr['present_groups']
  assert init['trainable_parameters']==s['expected_total_parameters']==a['parameter_scope']
  verified(cp/'adapter.safetensors',tr['adapter_sha256'])
  if (cp/'initial_adapter.safetensors').exists():verified(cp/'initial_adapter.safetensors')
  else:missing_initial.append(name)
  for f in ['spec.json','TRAIN_ORDER.json','INITIALIZATION.json','TRAINING.json','COMPLETE.json']:verified(cp/f)
  verified(old/'audits'/f'{name}.json')
  for tag,h in a['summary_sha256'].items():
   ep=old/'evaluations'/name/tag;verified(ep/'SUMMARY.json',h);summ=read(ep/'SUMMARY.json');verified(ep/'responses.jsonl',summ['responses_sha256'])
   agg=aggregate(rows(ep/'responses.jsonl'),s['task']);assert all(summ[k]==v for k,v in agg.items())
  assert set(a['summary_sha256'])==set(['smoke_dev'] if s['smoke'] else s.get('eval_splits',['dev','test']))
  if s['stage']=='tuning':assert not (old/'evaluations'/name/'test').exists()
  responses+=a['responses'];official+=a['official_sql_executions'];checked.append(name)
  if s['smoke']:smokes.append(list(architecture(s)))
 # Recheck the 160 previously reused artifacts rather than trusting their old flag.
 for path,h in read(old/'REUSE_AUDIT.json')['files'].items():verified(Path(path),h)
 # Recompute all frozen LR choices from dev only.
 selection=read(old/'SELECTION_FROZEN.json')
 for row in selection['selection']:
  scores=collections.defaultdict(list)
  for s in read(old/'TUNING_JOBS.json'):
   if list(architecture(s))==row['architecture']:scores[s['boundary_lr_ratio']].append(read(old/'evaluations'/s['name']/'dev/SUMMARY.json')['primary'])
  choice=max(scores,key=lambda x:(sum(scores[x])/len(scores[x]),-abs(x-1),-x));assert choice==row['ratio']
 write(HERE/'SOURCE_HASH_CACHE.json',cache)
 out=dict(at=now(),status='passed_with_retention_limitations' if missing_initial else 'passed',missing_initial_snapshots=missing_initial,initial_snapshot_limitation='Initial tensor files absent: prior per-run audit and initialization hashes remain, but these original bytes cannot be rehashed now.',complete=state['stage']=='complete',audited=len(checked),expected=684,responses=responses,official_sql_executions=official,files=files,checked=checked)
 write(HERE/'SOURCE_AUDIT.json',out);write(HERE/'INHERITED_SMOKES.json',dict(status='passed',architectures=smokes,source=str(old)))
 print('source audit passed',len(checked),'complete',out['complete'],flush=True)
 return out
if __name__=='__main__':audit()
