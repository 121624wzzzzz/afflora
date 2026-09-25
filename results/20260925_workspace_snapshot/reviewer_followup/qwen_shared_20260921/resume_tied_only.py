"""Scope correction requested by user: adopt active tied jobs, exclude untied shared jobs."""
import sys,os,subprocess,time,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;HERE=R/'study';sys.path.insert(0,str(HERE))
from common import read,write,now,PYTHON,sha
lock=(HERE/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
models={'qwen25_15b_base','qwen3_06b_base'}
alljobs=read(HERE/'SMOKE_JOBS.json')+read(HERE/'FORMAL_JOBS.json');jobs=[s for s in alljobs if s['model'] in models]
write(R/'SCOPE_AMENDMENT.json',dict(at=now(),reason='User: untied models must not run tied/shared aLoRA; preserve independent single/dual-side ablations only.',models=sorted(models),formal_runs=24,smokes=8,jobs=jobs,excluded_jobs=[s['name'] for s in alljobs if s['model'] not in models],original_protocol_preserved=True))
previous=read(R/'STATE_BEFORE_TIED_ONLY.json');oldactive={v['name']:(int(g),v) for g,v in previous['active'].items()}
active={};pending=[];done=[];failed=[];passed=set()
def key(s):return s['model'],s['task'],s['arm']
class Adopted:
 def __init__(self,pid,marker):self.pid=pid;self.marker=marker
 def poll(self):
  p=Path('/proc')/str(self.pid)
  try:
   if p.exists() and (p/'stat').read_text().split(') ',1)[1][0]!='Z':return None
  except FileNotFoundError:pass
  return 0 if self.marker.exists() else 1
for s in jobs:
 n=s['name'];audit=HERE/'audits'/f'{n}.json';complete=HERE/'checkpoints'/n/'COMPLETE.json'
 if audit.exists():
  assert read(audit)['status']=='passed';done.append(n)
  if s['smoke']:passed.add(key(s))
 elif n in oldactive:
  gpu,v=oldactive[n];active[gpu]=dict(spec=s,p=Adopted(v['pid'],complete if v['phase']=='train' else audit),phase=v['phase'],log=None)
 elif complete.exists():
  # A just-finished process may have exited between the snapshot and takeover.
  f=(HERE/'logs'/f'{n}.takeover_audit.log').open('w');p=subprocess.Popen([PYTHON,'-B',str(HERE/'audit_one.py'),'--name',n],env=env,cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
  active[-len(active)-1]=dict(spec=s,p=p,phase='audit',log=f)
 elif (HERE/'checkpoints'/n).exists():failed.append(dict(name=n,phase='takeover',reason='Incomplete checkpoint without a live adopted worker'))
 else:pending.append(s)
def state(stage):
 d=dict(at=now(),stage=stage,scope='tied_only_user_correction',done=done,failed=failed,pending=[s['name'] for s in pending],active={str(g):dict(name=v['spec']['name'],pid=v['p'].pid,phase=v['phase']) for g,v in active.items()})
 write(HERE/'STATE.json',d);write(R/'STATE_TIED_ONLY.json',d)
try:
 while pending or active:
  for gpu,v in list(active.items()):
   code=v['p'].poll()
   if code is None:continue
   if v['log']:v['log'].close()
   s=v['spec'];n=s['name']
   if code:
    failed.append(dict(name=n,phase=v['phase'],exit_code=code));del active[gpu]
    if s['smoke']:pending=[j for j in pending if key(j)!=key(s)]
    continue
   if v['phase']=='train':
    f=(HERE/'logs'/f'{n}.audit.log').open('w');p=subprocess.Popen([PYTHON,'-B',str(HERE/'audit_one.py'),'--name',n],env=env,cwd=HERE,stdout=f,stderr=subprocess.STDOUT);v.update(p=p,log=f,phase='audit');continue
   done.append(n)
   if s['smoke']:passed.add(key(s))
   del active[gpu];subprocess.run([PYTHON,'-B',str(HERE/'report.py')],env=env,cwd=HERE,check=True)
  memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  for gpu,mb in sorted((tuple(map(int,x.split(','))) for x in memory.strip().splitlines()),key=lambda x:-x[1]):
   if gpu in active:continue
   s=next((s for s in pending if mb>=(20000 if '15b' in s['model'] else 14000) and (s['smoke'] or key(s) in passed)),None)
   if s is None:continue
   pending.remove(s);n=s['name'];write(HERE/'specs'/f'{n}.json',s);assert not (HERE/'checkpoints'/n).exists()
   f=(HERE/'logs'/f'{n}.log').open('w');p=subprocess.Popen([PYTHON,'-B',str(HERE/'run.py'),'--spec',str(HERE/'specs'/f'{n}.json')],env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
   active[gpu]=dict(spec=s,p=p,phase='train',log=f);print(now(),'start',gpu,n,flush=True)
  state('running');time.sleep(30)
 state('complete' if not failed else 'completed_with_failures')
 if failed:raise RuntimeError(failed)
 for rel,h in read(HERE/'CODE_FROZEN.json')['files'].items():assert sha(HERE/rel)==h,rel
 for rel,h in read(HERE/'DATA_FROZEN.json')['files'].items():assert sha(HERE/rel)==h,rel
 outputs=official=0
 for s in jobs:
  n=s['name'];a=read(HERE/'audits'/f'{n}.json');assert a['status']=='passed';tr=read(HERE/'checkpoints'/n/'TRAINING.json');assert sha(HERE/'checkpoints'/n/'adapter.safetensors')==tr['adapter_sha256']
  for tag,h in a['summary_sha256'].items():
   p=HERE/'evaluations'/n/tag/'SUMMARY.json';assert sha(p)==h;assert sha(p.parent/'responses.jsonl')==read(p)['responses_sha256']
  outputs+=a['responses'];official+=a['official_sql_executions']
 subprocess.run([PYTHON,'-B',str(HERE/'report.py')],env=env,cwd=HERE,check=True)
 groups=read(HERE/'RESULTS.json')['groups'];assert len(groups)==4 and all(len(x['paired_seeds'])==3 for x in groups)
 write(HERE/'FINAL_AUDIT.json',dict(at=now(),status='passed',scope='tied_only',formal_runs=24,smokes=8,new_responses=outputs,official_sql_executions=official,scope_amendment_sha256=sha(R/'SCOPE_AMENDMENT.json'),results_sha256=sha(HERE/'RESULTS.json')))
 print('FINAL AUDIT PASSED',flush=True)
except Exception:
 state('scheduler_error');traceback.print_exc();raise
