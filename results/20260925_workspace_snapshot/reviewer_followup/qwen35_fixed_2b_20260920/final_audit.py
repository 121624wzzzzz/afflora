"""Verify every result and shared initialization after the finite queue completes."""
import collections,torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from common import *
from design import *
def main():
 assert read(HERE/'SCHEDULER_COMPLETE.json')['status']=='passed'
 for f in ['CODE_FROZEN.json','DATA_FROZEN.json']:
  for rel,h in read(HERE/f)['files'].items():assert sha(HERE/rel)==h,(f,rel)
 cfg=read(HERE/'models.json')[MODEL]
 for p,h in cfg['files'].items():assert sha(p)==h,p
 tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True);token_checks=0
 for task in TASKS:
  for split in ['train','dev','confirm']:
   data=rows(HERE/f'data/{task}_{split}.jsonl');tokens=read(HERE/f'tokens/{task}_{MODEL}_{split}.json');assert len(data)==len(tokens)
   for r,q in zip(data,tokens):
    assert r['id']==q['id'] and q['prompt_ids']==tok.encode(prompt(r),add_special_tokens=True)
    assert q['target_ids']==tok.encode(target_text(r),add_special_tokens=False)+[TOKEN_EOS_ID];token_checks+=1
 sel=read(HERE/'SELECTION.json');assert sel['code_frozen_sha256']==sha(HERE/'CODE_FROZEN.json') and sel['data_frozen_sha256']==sha(HERE/'DATA_FROZEN.json')
 assert sel['confirmation_jobs_sha256']==sha(HERE/'CONFIRMATION_JOBS.json')
 for rel,h in sel['inputs'].items():assert sha(HERE/rel)==h,rel
 for task in TASKS:
  for arm,rs in sel['scores'][task].items():
   assert sel['selected'][task][arm]==CANDIDATES[max(range(6),key=lambda i:(rs[i]['mean'],-i))]
   for i,r in enumerate(rs):
    actual=[read(HERE/'evaluations'/spec('search',task,arm,i,s)['name']/'dev/SUMMARY.json')['primary'] for s in SEARCH_SEEDS]
    assert r['seed_scores']==actual and r['mean']==sum(actual)/2
 state=read(HERE/'STATE.json');assert len(state['jobs'])==114 and all(j['state']=='passed' for j in state['jobs'])
 paired=collections.defaultdict(list);responses=official=candidate_checks=tensors=0;frozen=set()
 for j in state['jobs']:
  s=j['spec'];validate_spec(s);root=HERE/'checkpoints'/s['name'];a=read(HERE/'audits'/f'{s["name"]}.json')
  assert a['status']=='passed'
  assert a['numerical_policy_final_sha256']==sha(root/'NUMERICAL_POLICY_FINAL.json')
  if s['arm']!='base':
   from reference_check import check_reference
   initial=read(root/'INITIALIZATION.json');repeat=read(root/'INITIAL_REPEAT.json')
   assert check_reference(repeat,s['task'],s['seed'])==initial['cross_process_initial_reference']
   assert read(root/'TRAINING.json')['history'][0]['loss']==repeat['losses'][0]
  responses+=a['responses'];official+=a['official_sql_executions'];frozen.add(read(root/'INITIALIZATION.json')['frozen_before'])
  if s['arm']!='base':assert sha(root/'adapter.safetensors')==read(root/'TRAINING.json')['adapter_sha256']
  for tag,h in a['summary_sha256'].items():
   p=HERE/'evaluations'/s['name']/tag;assert sha(p/'SUMMARY.json')==h and read(p/'SUMMARY.json')['responses_sha256']==sha(p/'responses.jsonl')
   if s['task']=='trec50':
    c=read(p/'CANDIDATE_AUDIT.json');assert c['argmax_equal'] and c['max_abs_joint_logprob_error']<2e-4;candidate_checks+=c['examples']*c['candidate_count']
    assert c['scoring_path']=='five native complete-prefix forwards; no cache'
  if s['stage'] in ['search','confirmation']:paired[s['seed']].append(s)
 assert len(frozen)==1
 for seed,ss in paired.items():
  ref=bd_ref=None;orders={}
  for s in ss:
   root=HERE/'checkpoints'/s['name'];d=load_file(str(root/'initial_adapter.safetensors'))
   shared={k:(v[:8] if '.lora_A.' in k else v[:,:8]) for k,v in d.items() if '.lora_' in k}
   if ref is None:ref=shared
   else:assert shared.keys()==ref.keys() and all(torch.equal(v,ref[k]) for k,v in shared.items());tensors+=len(shared)
   bd={k:v for k,v in d.items() if k.startswith('boundary_')}
   if bd:
    if bd_ref is None:bd_ref=bd
    else:assert bd.keys()==bd_ref.keys() and all(torch.equal(v,bd_ref[k]) for k,v in bd.items())
   order=read(root/'TRAIN_ORDER.json')
   if s['task'] not in orders:orders[s['task']]=order
   else:assert order==orders[s['task']]
 write(HERE/'FINAL_AUDIT.json',{'at':now(),'status':'passed','jobs':len(state['jobs']),'responses':responses,'official_valid_query_checks':official,
       'native_candidate_logprob_checks':candidate_checks,'tokens_reencoded':token_checks,'paired_seeds':len(paired),'actual_shared_tensors_compared':tensors,
       'selection_sha256':sha(HERE/'SELECTION.json'),'analysis_sha256':sha(HERE/'ANALYSIS.json'),'bootstrap_sha256':sha(HERE/'CLUSTER_BOOTSTRAP.json')})
 print('final audit passed',flush=True)
if __name__=='__main__':main()
