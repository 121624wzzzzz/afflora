"""Read-only final verification of completed runs; never changes study inputs."""
import sys,time,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;S=R/'study';sys.path.insert(0,str(S))
from common import *
while True:
 if (S/'STATE.json').exists():
  state=read(S/'STATE.json')
  if state['stage']!='running':break
 time.sleep(60)
assert state['stage']=='complete' and not state['failed'],state
jobs=read(S/'SMOKE_JOBS.json')+read(S/'FORMAL_JOBS.json');assert len(jobs)==64 and len(state['done'])==64
for rel,h in read(S/'CODE_FROZEN.json')['files'].items():assert sha(S/rel)==h,rel
for rel,h in read(S/'DATA_FROZEN.json')['files'].items():assert sha(S/rel)==h,rel
outputs=0;official=0
for s in jobs:
 n=s['name'];a=read(S/'audits'/f'{n}.json');assert a['status']=='passed';tr=read(S/'checkpoints'/n/'TRAINING.json');assert sha(S/'checkpoints'/n/'adapter.safetensors')==tr['adapter_sha256']
 for tag,h in a['summary_sha256'].items():
  p=S/'evaluations'/n/tag/'SUMMARY.json';assert sha(p)==h;summary=read(p);assert sha(p.parent/'responses.jsonl')==summary['responses_sha256']
 outputs+=a['responses'];official+=a['official_sql_executions']
subprocess.run([PYTHON,'-B',str(S/'report.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
assert len(read(S/'RESULTS.json')['groups'])==8 and all(len(x['paired_seeds'])==3 for x in read(S/'RESULTS.json')['groups'])
write(S/'FINAL_AUDIT.json',dict(at=now(),status='passed',formal_runs=48,smokes=16,reused_runs=len(read(S/'REUSE_AUDIT.json')['runs']),new_responses=outputs,official_sql_executions=official,code_manifest_sha256=sha(S/'CODE_FROZEN.json'),data_manifest_sha256=sha(S/'DATA_FROZEN.json'),results_sha256=sha(S/'RESULTS.json')))
print('FINAL AUDIT PASSED',flush=True)
