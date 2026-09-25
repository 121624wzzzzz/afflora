import os,subprocess,time,traceback,fcntl
from common import *
lock=(HERE/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
pending=read(HERE/'SMOKE_JOBS.json')+read(HERE/'FORMAL_JOBS.json');active={};failed=[];done=[]
# Per-configuration smoke gate; no data-dependent stopping.
def key(s):return (s['model'],s['task'],s['arm'])
passed=set()
def free():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a):int(b) for a,b in (x.split(',') for x in s.strip().splitlines())}
def required(s):return 60000 if s['model'] in ['qwen25_7b_base','qwen3_8b_base'] else (20000 if '15b' in s['model'] else 14000)
def status(stage):write(HERE/'STATE.json',dict(at=now(),stage=stage,done=done,failed=failed,pending=[s['name'] for s in pending],active={str(g):dict(name=v['spec']['name'],pid=v['p'].pid,phase=v['phase']) for g,v in active.items()}))
try:
 assert read(HERE/'READY.json')['status']=='passed'
 while pending or active:
  for gpu,v in list(active.items()):
   code=v['p'].poll()
   if code is None:continue
   v['log'].close();s=v['spec'];n=s['name']
   if code:
    failed.append(dict(name=n,phase=v['phase'],exit_code=code));del active[gpu]
    if s['smoke']:pending=[j for j in pending if key(j)!=key(s)]
    continue
   if v['phase']=='train':
    f=(HERE/'logs'/f'{n}.audit.log').open('w');p=subprocess.Popen([PYTHON,'-B',str(HERE/'audit_one.py'),'--name',n],env=env,cwd=HERE,stdout=f,stderr=subprocess.STDOUT);v.update(p=p,log=f,phase='audit');continue
   done.append(n)
   if s['smoke']:passed.add(key(s))
   del active[gpu]
   subprocess.run([PYTHON,'-B',str(HERE/'report.py')],env=env,cwd=HERE,check=True)
  memory=free()
  for gpu,mb in sorted(memory.items(),key=lambda kv:-kv[1]):
   if gpu in active:continue
   eligible=[s for s in pending if mb>=required(s) and (s['smoke'] or key(s) in passed)]
   candidate=max(eligible,key=required) if eligible else None
   if candidate is None:continue
   pending.remove(candidate);s=candidate;n=s['name'];write(HERE/'specs'/f'{n}.json',s)
   assert not (HERE/'checkpoints'/n).exists(),n
   f=(HERE/'logs'/f'{n}.log').open('w');p=subprocess.Popen([PYTHON,'-B',str(HERE/'run.py'),'--spec',str(HERE/'specs'/f'{n}.json')],env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
   active[gpu]=dict(p=p,log=f,spec=s,phase='train');print(now(),'start',gpu,n,flush=True)
  status('running');time.sleep(30)
 status('complete' if not failed else 'completed_with_failures')
except Exception:
 status('scheduler_error');traceback.print_exc();raise
