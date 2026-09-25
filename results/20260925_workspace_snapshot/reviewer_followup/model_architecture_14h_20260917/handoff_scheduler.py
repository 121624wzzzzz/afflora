"""One-time checked supervisor handoff; never stops any experimental worker."""
import os,signal,time
import psutil
from scheduler import ROOT,PYTHON,read,write,sha,now
from resource_scheduler_v2 import MEMORY_MIB

def main():
 assert not (ROOT/'SCHEDULER_V2_SOURCE.json').exists()
 pid=int((ROOT/'scheduler.lock').read_text());parent=psutil.Process(pid)
 assert parent.cwd()==str(ROOT) and 'scheduler.py' in parent.cmdline()
 assert not (ROOT/'SCHEDULER_COMPLETE.json').exists()
 os.kill(pid,signal.SIGSTOP);stopped=True;terminated=False
 try:
  for _ in range(100):
   if parent.status()==psutil.STATUS_STOPPED:break
   time.sleep(.01)
  assert parent.status()==psutil.STATUS_STOPPED
  state=read(ROOT/'STATE.json');assert not any(j['status'] in ['auditing','awaiting_audit'] for j in state['jobs'].values())
  expected={j['pid'] for j in state['jobs'].values() if j['status']=='running'}
  actual={p.pid for p in parent.children() if p.status()!=psutil.STATUS_ZOMBIE}
  assert actual==expected,(actual,expected)
  workers={}
  for key,j in state['jobs'].items():
   if j['status']!='running':continue
   p=psutil.Process(j['pid']);args=p.cmdline();path=ROOT/j['phase'];sp=path/'specs'/f"{j['spec']['name']}.json"
   assert str(path/'run.py') in args and str(sp) in args and p.cwd()==str(path),(key,args)
   assert p.environ()['CUDA_VISIBLE_DEVICES']==str(j['gpu'])
   assert read(sp)==j['spec']
   workers[key]={'pid':p.pid,'create_time':p.create_time(),'cmdline':args,'cwd':p.cwd(),'gpu':j['gpu']}
  write(ROOT/'STATE_HANDOFF_PRE_V2.json',state)
  write(ROOT/'HANDOFF_PROCESS_PROOF.json',{'at':now().isoformat(),'old_supervisor_pid':pid,'old_supervisor_cmdline':parent.cmdline(),'workers':workers})
  probes={}
  for phase in ['core','new_tasks']:
   for p in (ROOT/phase/'checkpoints').glob('smoke_*/INITIALIZATION.json'):
    probes[str(p.relative_to(ROOT))]=read(p)['worst_length_memory_probe']
  write(ROOT/'RESOURCE_AMENDMENT_V2.json',{'at':now().isoformat(),'reason':'Shared GPU 7 dropped below the original uniform 48-GiB admission line while GPU 0 had approximately 14 GiB free. Use size-specific admission so small models can fill otherwise unavailable slots.',
    'old_pool':[1,2,3,4,5,6,7],'new_pool':list(range(8)),'old_min_free_mib':49152,'new_min_free_mib':MEMORY_MIB,
    'unchanged':['all model and dataset identities','all scientific code and scoring','all job specs, seeds, training budgets and evaluation batches','fixed question-driven priority order among jobs that fit memory','deadline and duration allowances','independent per-run audit'],
    'selection':'No scores are read for admission. Highest fixed priority among memory-eligible jobs per available GPU.',
    'handoff':'Only the supervisor is replaced; existing workers keep their PID, GPU, stdout and training/evaluation state. Their eventual exit status is not a child wait status in the replacement supervisor and is recorded as null; completion requires COMPLETE.json plus a fresh successful independent auditor. No worker is restarted.',
    'smoke_memory_probes':probes})
  # SIGTERM is pending while stopped; SIGCONT delivers it before user code resumes.
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False;terminated=True
  parent.wait(timeout=10)
  names=['resource_scheduler_v2.py','handoff_scheduler.py','RESOURCE_AMENDMENT_V2.json','STATE_HANDOFF_PRE_V2.json','HANDOFF_PROCESS_PROOF.json']
  write(ROOT/'SCHEDULER_V2_SOURCE.json',{'at':now().isoformat(),'files':{n:sha(ROOT/n) for n in names}})
  os.execv(PYTHON,[PYTHON,'-u',str(ROOT/'resource_scheduler_v2.py')])
 except BaseException:
  if stopped and not terminated:os.kill(pid,signal.SIGCONT)
  raise

if __name__=='__main__':main()
