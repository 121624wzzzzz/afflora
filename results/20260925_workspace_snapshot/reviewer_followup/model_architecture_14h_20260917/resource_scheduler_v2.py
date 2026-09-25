"""Resource-only supervisor replacement. Scientific specs and priorities stay frozen."""
import os,sys,time,json,subprocess,signal,fcntl
from pathlib import Path
from datetime import datetime
import psutil
from scheduler import ROOT,PYTHON,read,write,sha,now,emit,allowance

MEMORY_MIB={
 'qwen25_05b_base':14336,'qwen3_06b_base':14336,
 'qwen25_15b_base':24576,'qwen3_17b_base':24576,
 'qwen25_3b_base':36864,'qwen3_4b_base':36864,
 'qwen25_7b_base':49152,'qwen3_8b_base':49152}

def free_memory():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a):int(b) for line in text.splitlines() for a,b in [line.split(',')] if int(a) in range(8)}

def adopted_alive(record):
 try:
  p=psutil.Process(record['pid'])
  return p.create_time()==record['create_time'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False

def main():
 lock=(ROOT/'scheduler.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
 for rel,h in read(ROOT/'SCHEDULER_V2_SOURCE.json')['files'].items():assert sha(ROOT/rel)==h,rel
 state=read(ROOT/'STATE_HANDOFF_PRE_V2.json');proof=read(ROOT/'HANDOFF_PROCESS_PROOF.json')
 assert read(ROOT/'STATE.json')==state,'Unexpected state change between supervisors'
 window=read(ROOT/'WINDOW.json');launch_stop=datetime.fromisoformat(window['stop_new_launches_utc']);worker_stop=datetime.fromisoformat(window['finish_workers_by_utc'])
 phases=state['phases'];jobs=state['jobs'];halt=set(state['halted_phases']);running={};auditing={};audit_queue=[]
 for phase in ['core','new_tasks']:
  path=ROOT/phase;ready=read(path/'READY.json')
  for kind in ['CODE','DATA']:
   mf=path/f'{kind}_FROZEN.json';assert sha(mf)==ready[kind.lower()+'_sha256']
   for rel,h in read(mf)['files'].items():assert sha(path/rel)==h,(phase,rel)
 for key,j in jobs.items():
  assert j['status'] not in ['auditing','awaiting_audit'],'Handoff requires an empty audit queue'
  if j['status']=='running':
   r=proof['workers'][key];assert r['pid']==j['pid'];j['supervisor_handoff']=True
   running[key]={'adopted':True,'record':r,'gpu':j['gpu'],'log':None}
 emit({'resource_supervisor':'v2','adopted_workers':len(running),'memory_mib':MEMORY_MIB,'scientific_specs_changed':False})
 while True:
  for key,r in list(running.items()):
   job=jobs[key]
   if r['adopted']:
    alive=adopted_alive(r['record']);code=None;pid=r['record']['pid']
   else:
    code=r['process'].poll();alive=code is None;pid=r['process'].pid
   if alive and now()>=worker_stop and not job.get('deadline_terminated'):
    os.killpg(pid,signal.SIGTERM);job['deadline_terminated']=True;job['termination_at']=now().isoformat();emit({'deadline_terminate':key})
   if not alive:
    if r['log'] is not None:r['log'].close()
    del running[key];job['worker_exit_code']=code;job['worker_finished_at']=now().isoformat()
    job['worker_completion_basis']='complete_artifact_and_independent_audit_after_supervisor_handoff' if r['adopted'] else 'captured_process_exit'
    p=ROOT/job['phase']/'checkpoints'/job['spec']['name']/'COMPLETE.json'
    complete=p.exists() and read(p)['status']=='passed'
    if complete and (r['adopted'] or code==0) and not job.get('deadline_terminated'):
     job['status']='awaiting_audit';audit_queue.append(key)
    else:
     job['status']='deadline_incomplete' if job.get('deadline_terminated') else 'failed';halt.add(job['phase'])
    emit({'worker_finished':key,'exit_code':code,'completion_basis':job['worker_completion_basis'],'status':job['status']})
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
   free=free_memory();occupied={r['gpu'] for r in running.values()}
   ordered=sorted(jobs.items(),key=lambda x:(x[1]['priority'],x[1]['spec']['seed'],x[1]['spec']['task'],x[1]['spec']['model'],x[0]))
   for gpu in sorted(set(free)-occupied):
    candidate=None
    for key,job in ordered:
     if job['status']!='pending' or job['phase'] in halt:continue
     if free[gpu]<MEMORY_MIB[job['spec']['model']]:continue
     if job['stage']=='FORMAL':
      gate=job['phase']+'/smoke_'+job['spec']['task']+'_'+job['spec']['model']
      if jobs[gate]['status']!='passed':continue
     if (worker_stop-now()).total_seconds()<job['allowance_seconds']:continue
     candidate=(key,job);break
    if candidate is None:continue
    key,job=candidate;path=ROOT/job['phase'];spec=job['spec'];sp=path/'specs'/f"{spec['name']}.json";write(sp,spec)
    assert not (path/'checkpoints'/spec['name']).exists(),key
    log=(path/'logs'/f"{spec['name']}.log").open('w')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    proc=subprocess.Popen([PYTHON,'-u',str(path/'run.py'),'--spec',str(sp)],cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job.update(status='running',started_at=now().isoformat(),gpu=gpu,pid=proc.pid,admission_free_mib=free[gpu],admission_required_mib=MEMORY_MIB[spec['model']])
    running[key]={'adopted':False,'process':proc,'log':log,'gpu':gpu};emit({'launched':key,'gpu':gpu,'pid':proc.pid,'free_mib':free[gpu],'required_mib':MEMORY_MIB[spec['model']]})
  else:
   for job in jobs.values():
    if job['status']=='pending':job['status']='not_admitted_time_budget'
  state={'at':now().isoformat(),'scheduler_version':2,'phases':phases,'halted_phases':sorted(halt),'jobs':jobs,'counts':{s:sum(r['status']==s for r in jobs.values()) for s in sorted({r['status'] for r in jobs.values()})}}
  write(ROOT/'STATE.json',state)
  if not running and not auditing and not audit_queue and (not any(r['status']=='pending' for r in jobs.values()) or now()>=launch_stop):break
  time.sleep(10)
 write(ROOT/'SCHEDULER_COMPLETE.json',{'at':now().isoformat(),'status':'finished','scheduler_version':2,'counts':state['counts'],'halted_phases':sorted(halt),'all_planned_jobs_completed':all(r['status']=='passed' for r in jobs.values())})

if __name__=='__main__':main()
