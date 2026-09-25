"""Shared resource scheduler; accepts only ready frozen scientific phases."""
import os,sys,time,json,subprocess,signal,fcntl,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;PYTHON='/home/wz/anaconda3/envs/torch24/bin/python'
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def emit(x):print(json.dumps(dict(at=now().isoformat(),**x),ensure_ascii=False),flush=True)
def free_gpus():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a) for line in text.splitlines() for a,b in [line.split(',')] if int(a) in range(1,8) and int(b)>=49152}
def allowance(phase,spec):
 big=spec['model'] in ['qwen25_7b_base','qwen3_8b_base'];medium=spec['model'] in ['qwen25_3b_base','qwen3_17b_base','qwen3_4b_base']
 if spec['smoke']:return 1200
 if spec['task'] in ['cluener','squad2']:return (75 if big else 50 if medium else 30)*60
 return (40 if big else 30 if medium else 20)*60
def main():
 lock=(ROOT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
 assert not (ROOT/'STATE.json').exists(),'Explicit inspected resume is required; no automatic retry.'
 window=read(ROOT/'WINDOW.json');launch_stop=datetime.fromisoformat(window['stop_new_launches_utc']);worker_stop=datetime.fromisoformat(window['finish_workers_by_utc'])
 phases={};jobs={};running={};auditing={};audit_queue=[];halt=set()
 while True:
  for phase in ['core','new_tasks']:
   path=ROOT/phase
   if phase in phases or not (path/'READY.json').exists():continue
   ready=read(path/'READY.json');assert ready['status']=='passed'
   for kind in ['CODE','DATA']:
    mf=path/f'{kind}_FROZEN.json';assert sha(mf)==ready[kind.lower()+'_sha256']
    for rel,h in read(mf)['files'].items():assert sha(path/rel)==h,(phase,rel)
   phases[phase]={'registered_at':now().isoformat(),'ready':ready}
   for stage in ['SMOKE','FORMAL']:
    for record in read(path/f'{stage}_JOBS.json'):
     spec=record['spec'];key=phase+'/'+spec['name'];assert key not in jobs
     jobs[key]=dict(record,phase=phase,stage=stage,status='pending',allowance_seconds=allowance(phase,spec))
   emit({'registered':phase,'jobs':sum(r['phase']==phase for r in jobs.values())})
  for key,r in list(running.items()):
   job=jobs[key];proc=r['process'];code=proc.poll()
   if code is None and now()>=worker_stop:
    os.killpg(proc.pid,signal.SIGTERM);job['deadline_terminated']=True;job['termination_at']=now().isoformat();emit({'deadline_terminate':key})
   if code is not None:
    r['log'].close();del running[key];job['worker_exit_code']=code;job['worker_finished_at']=now().isoformat()
    if code==0:
     assert read(ROOT/job['phase']/'checkpoints'/job['spec']['name']/'COMPLETE.json')['status']=='passed'
     job['status']='awaiting_audit';audit_queue.append(key)
    else:job['status']='deadline_incomplete' if job.get('deadline_terminated') else 'failed';halt.add(job['phase'])
    emit({'worker_finished':key,'exit_code':code,'status':job['status']})
  for key,r in list(auditing.items()):
   code=r['process'].poll()
   if code is not None:
    r['log'].close();del auditing[key];job=jobs[key];job['audit_exit_code']=code;job['finished_at']=now().isoformat();job['status']='passed' if code==0 else 'audit_failed'
    if code:halt.add(job['phase'])
    emit({'audited':key,'status':job['status']})
  while audit_queue and len(auditing)<2:
   key=audit_queue.pop(0);job=jobs[key];path=ROOT/job['phase'];log=(path/'logs'/f"audit_{job['spec']['name']}.log").open('w')
   proc=subprocess.Popen([PYTHON,'-u',str(path/'audit_one.py'),'--name',job['spec']['name']],cwd=path,env=dict(os.environ,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   auditing[key]={'process':proc,'log':log};job['status']='auditing'
  if now()<launch_stop:
   free=free_gpus()-{r['gpu'] for r in running.values()}
   ordered=sorted(jobs.items(),key=lambda x:(x[1]['priority'],x[1]['spec']['seed'],x[1]['spec']['task'],x[1]['spec']['model'],x[0]))
   for gpu in sorted(free):
    candidate=None
    for key,job in ordered:
     if job['status']!='pending' or job['phase'] in halt:continue
     if job['stage']=='FORMAL':
      gate=job['phase']+'/smoke_'+job['spec']['task']+'_'+job['spec']['model']
      if jobs[gate]['status']!='passed':continue
     if (worker_stop-now()).total_seconds()<job['allowance_seconds']:continue
     candidate=(key,job);break
    if candidate is None:break
    key,job=candidate;path=ROOT/job['phase'];spec=job['spec'];sp=path/'specs'/f"{spec['name']}.json";write(sp,spec)
    assert not (path/'checkpoints'/spec['name']).exists(),key
    (path/'logs').mkdir(exist_ok=True);log=(path/'logs'/f"{spec['name']}.log").open('w')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    proc=subprocess.Popen([PYTHON,'-u',str(path/'run.py'),'--spec',str(sp)],cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job.update(status='running',started_at=now().isoformat(),gpu=gpu,pid=proc.pid)
    running[key]={'process':proc,'log':log,'gpu':gpu};emit({'launched':key,'gpu':gpu,'pid':proc.pid})
  else:
   for job in jobs.values():
    if job['status']=='pending':job['status']='not_admitted_time_budget'
  state={'at':now().isoformat(),'phases':phases,'halted_phases':sorted(halt),'jobs':jobs,'counts':{s:sum(r['status']==s for r in jobs.values()) for s in sorted({r['status'] for r in jobs.values()})}}
  write(ROOT/'STATE.json',state)
  if not running and not auditing and not audit_queue and ((len(phases)==2 and not any(r['status']=='pending' for r in jobs.values())) or now()>=launch_stop):break
  time.sleep(10)
 write(ROOT/'SCHEDULER_COMPLETE.json',{'at':now().isoformat(),'status':'finished','counts':state['counts'],'halted_phases':sorted(halt),'all_planned_jobs_completed':all(r['status']=='passed' for r in jobs.values())})
if __name__=='__main__':main()
