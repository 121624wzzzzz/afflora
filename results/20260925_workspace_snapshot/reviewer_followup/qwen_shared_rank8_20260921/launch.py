from pathlib import Path
import sys,os,subprocess,shutil
R=Path(__file__).resolve().parent;S=R/'study';sys.path.insert(0,str(S))
from common import *
assert read(S/'PRIOR_SHARED_AUDIT.json')['status']=='passed'
shutil.copy2(R/'report_rank8.py',S/'report.py')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
subprocess.run([PYTHON,'-B',str(S/'test_shared.py')],env=env,check=True)
files=list(S.glob('*.py'))+list((S/'source').glob('*.py'))+[S/n for n in ['PROTOCOL.md','models.json','BUDGET_PLAN.json','FORMAL_JOBS.json','SMOKE_JOBS.json','REUSE_AUDIT.json','MODEL_IDENTITY_AUDIT.json','PRIOR_SHARED_AUDIT.json','SHARED_ALGEBRA_AUDIT.json','SOURCE_REUSE.json']]
write(S/'CODE_FROZEN.json',dict(at=now(),files={str(p.relative_to(S)):sha(p) for p in files}));write(S/'READY.json',dict(at=now(),status='passed'))
with (R/'scheduler.log').open('w') as f:subprocess.run([PYTHON,'-B',str(S/'scheduler.py')],env=env,cwd=S,stdout=f,stderr=subprocess.STDOUT,check=True)
state=read(S/'STATE.json');assert state['stage']=='complete' and not state['failed'];jobs=read(S/'SMOKE_JOBS.json')+read(S/'FORMAL_JOBS.json');assert len(state['done'])==len(jobs)==64
for rel,h in read(S/'CODE_FROZEN.json')['files'].items():assert sha(S/rel)==h,rel
for rel,h in read(S/'DATA_FROZEN.json')['files'].items():assert sha(S/rel)==h,rel
outputs=official=0
for s in jobs:
 n=s['name'];a=read(S/'audits'/f'{n}.json');assert a['status']=='passed';tr=read(S/'checkpoints'/n/'TRAINING.json');assert sha(S/'checkpoints'/n/'adapter.safetensors')==tr['adapter_sha256']
 for tag,h in a['summary_sha256'].items():
  p=S/'evaluations'/n/tag/'SUMMARY.json';assert sha(p)==h;assert sha(p.parent/'responses.jsonl')==read(p)['responses_sha256']
 outputs+=a['responses'];official+=a['official_sql_executions']
assert len(read(S/'RESULTS.json')['groups'])==4 and all(len(g['paired_seeds'])==3 for g in read(S/'RESULTS.json')['groups'])
assert len(read(S/'UNTIED_RESULTS.json')['groups'])==4 and all(len(g['paired_seeds'])==3 for g in read(S/'UNTIED_RESULTS.json')['groups'])
write(S/'FINAL_AUDIT.json',dict(at=now(),status='passed',formal_runs=48,smokes=16,new_responses=outputs,official_sql_executions=official,results_sha256=sha(S/'RESULTS.json')))
print('FINAL AUDIT PASSED',flush=True)
