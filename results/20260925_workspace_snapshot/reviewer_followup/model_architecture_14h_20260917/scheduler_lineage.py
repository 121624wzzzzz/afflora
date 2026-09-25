"""Read-only verification of supervisor handoffs and declared admission policies."""
from reporting_common import ROOT,read,sha

def verify_scheduler_lineage(state):
 manifests={};handoffs={}
 for name in ['SCHEDULER_SOURCE.json','SCHEDULER_V2_SOURCE.json','SCHEDULER_V3_SOURCE.json']:
  p=ROOT/name
  if not p.exists():continue
  for rel,h in read(p)['files'].items():assert sha(ROOT/rel)==h,rel
  manifests[name]=sha(p)
 for version,snapshot,proof in [(2,'STATE_HANDOFF_PRE_V2.json','HANDOFF_PROCESS_PROOF.json'),(3,'STATE_HANDOFF_PRE_V3.json','HANDOFF_PROCESS_PROOF_V3.json')]:
  if not (ROOT/snapshot).exists():continue
  old=read(ROOT/snapshot);assert set(old['jobs'])==set(state['jobs'])
  assert old['phases']==state['phases']
  for key,j in old['jobs'].items():
   for field in ['spec','priority','stage','phase','allowance_seconds']:
    assert state['jobs'][key][field]==j[field],(version,key,field)
  workers=read(ROOT/proof)['workers']
  for key,p in workers.items():assert state['jobs'][key]['pid']==p['pid'],(version,key)
  handoffs[str(version)]={'adopted_workers':len(workers),'snapshot_sha256':sha(ROOT/snapshot),'proof_sha256':sha(ROOT/proof)}
 if (ROOT/'COVERAGE_AMENDMENT_V3.json').exists():
  policy=read(ROOT/'COVERAGE_AMENDMENT_V3.json');review=read(ROOT/'STATE_COVERAGE_REVIEW_V3.json');old=read(ROOT/'STATE_HANDOFF_PRE_V3.json')
  assert sha(ROOT/'STATE_COVERAGE_REVIEW_V3.json')==policy['review_state_sha256']
  assert set(policy['effective_priorities'])==set(state['jobs'])
  changed={x['name']:x for x in policy['changes']};assert len(changed)==92
  assert sum(x['amended_priority']==18 for x in changed.values())==40
  assert sum(x['amended_priority']==35 for x in changed.values())==52
  for key,j in state['jobs'].items():
   expected=j['priority']
   if j['phase']=='core' and expected==25 and j['spec']['arm'] in ['base','hidden','hidden_budget','hidden_both']:expected=18
   if j['priority']==22:expected=35
   assert policy['effective_priorities'][key]==expected
   if expected!=j['priority']:
    assert key in changed and changed[key]['original_priority']==j['priority'] and changed[key]['amended_priority']==expected
    assert review['jobs'][key]['status']==old['jobs'][key]['status']=='pending'
   else:assert key not in changed
   if old['jobs'][key]['status']=='pending' and j.get('started_at'):
    assert j['admission_effective_priority']==expected,(key,'admission priority')
  handoffs['3']['changed_pending_priorities']=len(changed)
 return {'source_manifests':manifests,'handoffs':handoffs,'all_original_job_specs_and_priority_fields_preserved':True}
