import os,time,subprocess,fcntl,traceback,shutil
from common import *
lock=(HERE/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert read(HERE/'READY.json')['status']=='passed'
jobs=read(HERE/'JOBS.json');done=[];failed=[];active={};pending=list(jobs)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
def status(stage):write(HERE/'STATE.json',dict(at=now(),stage=stage,total=len(jobs),done=done,failed=failed,pending=[j['name'] for j in pending],active={str(g):dict(name=v['job']['name'],pid=v['process'].pid,phase=v['phase']) for g,v in active.items()}))
try:
 dep=Path(read(HERE/'SOURCES.json')['wait_for'])
 while not ((dep/'FINAL_AUDIT.json').exists() and (dep/'BASE_AUDIT.json').exists()):
  status('waiting_for_existing_experiments');time.sleep(30)
 assert read(dep/'FINAL_AUDIT.json')['status']=='passed' and read(dep/'BASE_AUDIT.json')['status']=='passed'
 for j in jobs:
  if (HERE/'results'/j['name']).exists():raise RuntimeError('Refuse overwrite: '+j['name'])
 while pending or active:
  for g,v in list(active.items()):
   code=v['process'].poll()
   if code is None:continue
   v['log'].close();name=v['job']['name']
   if code:failed.append(dict(name=name,phase=v['phase'],exit_code=code));del active[g];continue
   if v['phase']=='evaluate':
    log=(HERE/'logs'/f'{name}.audit.log').open('w');proc=subprocess.Popen([PYTHON,'-B',str(HERE/'audit_one.py'),'--name',name],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT);v.update(process=proc,log=log,phase='audit');continue
   done.append(name);del active[g];subprocess.run([PYTHON,'-B',str(HERE/'report.py')],cwd=HERE,env=env,check=True)
  if failed and not active:break
  if shutil.disk_usage(HERE).free<5*2**30:status('disk_guard_waiting');time.sleep(30);continue
  text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  for line in text.strip().splitlines():
   g,used,util=map(int,line.split(','))
   if g in active or used>1024 or util>10 or failed:continue
   eligible=[j for j in pending if j['method']=='base' or 'base__'+j['model'] in done]
   if not eligible:continue
   j=min(eligible,key=lambda j:j['method']!='base');pending.remove(j);name=j['name'];log=(HERE/'logs'/f'{name}.log').open('w')
   proc=subprocess.Popen([PYTHON,'-B',str(HERE/'run_eval.py'),'--name',name],cwd=HERE,env=dict(env,CUDA_VISIBLE_DEVICES=str(g)),stdout=log,stderr=subprocess.STDOUT);active[g]=dict(job=j,process=proc,log=log,phase='evaluate');print(now(),'start',g,name,flush=True)
  status('running');time.sleep(30)
 status('needs_technical_review' if failed else 'complete')
 if not failed:
  assert len(done)==62
  manifest={str(f.relative_to(HERE)):sha(f) for f in (HERE/'results').rglob('*') if f.is_file()}
  write(HERE/'OUTPUT_MANIFEST.json',dict(at=now(),files=manifest));write(HERE/'FINAL_AUDIT.json',dict(at=now(),status='passed',jobs=len(done),responses=124000,manifest_sha256=sha(HERE/'OUTPUT_MANIFEST.json')))
except Exception:
 status('scheduler_error');traceback.print_exc();raise
