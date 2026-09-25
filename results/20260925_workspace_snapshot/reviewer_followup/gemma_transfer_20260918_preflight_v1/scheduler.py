"""Finite queue: native technical checks -> equal LR search -> gated confirmation/Base."""
import fcntl,os,subprocess,time
from common import *
def gpu_memory():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return {int(x.split(',')[0]):int(x.split(',')[1]) for x in r.stdout.splitlines()}
def event(kind,**kw):
 r={'at':now(),'event':kind,**kw}
 with (HERE/'EVENTS.jsonl').open('a') as f:f.write(canonical(r)+'\n')
 print(canonical(r),flush=True)
def invoke(name):
 subprocess.run([PYTHON,'-u',str(HERE/name)],cwd=HERE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),check=True)
def main():
 lock=(HERE/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(HERE/'READY.json')['status']=='passed' and not (HERE/'STATE.json').exists()
 records=[dict(spec=s,state='pending') for s in read(HERE/'SMOKE_JOBS.json')+read(HERE/'SEARCH_JOBS.json')]
 active={};audits={};phase='preflight';start=now();failure=False
 (HERE/'logs').mkdir(exist_ok=True);(HERE/'specs').mkdir(exist_ok=True)
 while True:
  for name,(p,j,f) in list(active.items()):
   rc=p.poll()
   if rc is None:continue
   f.close();del active[name];j.update(returncode=rc,worker_finished_at=now())
   if rc!=0 or not (HERE/'checkpoints'/name/'COMPLETE.json').exists():j['state']='failed';event('worker_failed',name=name,returncode=rc);continue
   f=(HERE/'logs'/f'{name}.audit.log').open('w')
   p=subprocess.Popen([PYTHON,'-u',str(HERE/'audit_one.py'),'--name',name],cwd=HERE,
      env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),stdout=f,stderr=subprocess.STDOUT)
   j.update(state='auditing',audit_pid=p.pid);audits[name]=(p,j,f);event('audit_started',name=name,pid=p.pid)
  for name,(p,j,f) in list(audits.items()):
   rc=p.poll()
   if rc is None:continue
   f.close();del audits[name]
   ok=rc==0 and (HERE/'audits'/f'{name}.json').exists() and read(HERE/'audits'/f'{name}.json')['status']=='passed'
   j.update(state='passed' if ok else 'failed',audit_returncode=rc,finished_at=now());event(j['state'],name=name)
  failure=failure or any(j['state']=='failed' for j in records)
  if not failure:
   try:
    if phase=='preflight' and all(j['state']=='passed' for j in records if j['spec']['stage']=='smoke'):
     invoke('preflight_gate.py');phase='search';event('search_admission_opened')
    if phase=='search' and all(j['state']=='passed' for j in records if j['spec']['stage']=='search'):
     invoke('select_config.py');records.extend(dict(spec=s,state='pending') for s in read(HERE/'CONFIRMATION_JOBS.json')+read(HERE/'BASE_JOBS.json'))
     phase='confirmation';event('confirmation_admission_opened',selection_sha256=sha(HERE/'SELECTION.json'))
   except Exception as e:failure=True;event('gate_failed',error=str(e))
  if not failure:
   memory=gpu_memory();busy={j['gpu'] for p,j,f in active.values()}
   for gpu in range(8):
    if gpu in busy or memory.get(gpu,0)<69632:continue
    stages=['smoke'] if phase=='preflight' else ['search'] if phase=='search' else ['confirmation','base']
    candidates=[j for j in records if j['state']=='pending' and j['spec']['stage'] in stages]
    if not candidates:continue
    j=candidates[0];s=j['spec'];name=s['name'];write(HERE/'specs'/f'{name}.json',s)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    f=(HERE/'logs'/f'{name}.log').open('w');p=subprocess.Popen([PYTHON,'-u',str(HERE/'run.py'),'--spec',str(HERE/'specs'/f'{name}.json')],cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT)
    j.update(state='running',gpu=gpu,pid=p.pid,started_at=now());active[name]=(p,j,f);event('started',name=name,gpu=gpu,pid=p.pid)
  state={'started_at':start,'at':now(),'phase':phase,'jobs':records,'counts':{s:sum(j['state']==s for j in records) for s in ['pending','running','auditing','passed','failed']}}
  write(HERE/'STATE.json',state)
  if not active and not audits and (failure or (phase=='confirmation' and all(j['state']=='passed' for j in records))):
   write(HERE/'SCHEDULER_COMPLETE.json',{'at':now(),'status':'failed' if failure else 'passed','counts':state['counts']});event('scheduler_complete',status='failed' if failure else 'passed');break
  time.sleep(10)
 if failure:raise SystemExit(1)
 invoke('analyze.py');invoke('final_audit.py')
if __name__=='__main__':main()
