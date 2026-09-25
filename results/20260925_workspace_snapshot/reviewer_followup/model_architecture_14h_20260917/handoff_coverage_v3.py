"""Checked one-time handoff for the declared coverage scheduling amendment."""
import os,signal,time
import psutil
from scheduler import ROOT,PYTHON,read,write,sha,now

def main():
 assert not (ROOT/'SCHEDULER_V3_SOURCE.json').exists()
 for manifest in ['SCHEDULER_SOURCE.json','SCHEDULER_V2_SOURCE.json']:
  for rel,h in read(ROOT/manifest)['files'].items():assert sha(ROOT/rel)==h,rel
 policy=read(ROOT/'COVERAGE_AMENDMENT_V3.json')
 assert sha(ROOT/'STATE_COVERAGE_REVIEW_V3.json')==policy['review_state_sha256']
 pid=int((ROOT/'scheduler.lock').read_text());parent=psutil.Process(pid)
 assert parent.cwd()==str(ROOT) and str(ROOT/'resource_scheduler_v2.py') in parent.cmdline()
 assert not (ROOT/'SCHEDULER_COMPLETE.json').exists()
 os.kill(pid,signal.SIGSTOP);stopped=True;terminated=False
 try:
  for _ in range(100):
   if parent.status()==psutil.STATUS_STOPPED:break
   time.sleep(.01)
  assert parent.status()==psutil.STATUS_STOPPED
  state=read(ROOT/'STATE.json')
  assert not any(j['status'] in ['auditing','awaiting_audit'] for j in state['jobs'].values()),'Wait for the audit queue to drain, then retry only the supervisor handoff.'
  assert set(state['jobs'])==set(policy['effective_priorities'])
  for row in policy['changes']:
   j=state['jobs'][row['name']]
   assert j['status']=='pending' and j['priority']==row['original_priority']
  expected={j['pid'] for j in state['jobs'].values() if j['status']=='running'}
  actual={p.pid for p in parent.children() if p.status()!=psutil.STATUS_ZOMBIE}
  assert actual==expected,(actual,expected)
  workers={}
  for key,j in state['jobs'].items():
   if j['status']!='running':continue
   p=psutil.Process(j['pid']);args=p.cmdline();path=ROOT/j['phase'];sp=path/'specs'/f"{j['spec']['name']}.json"
   assert str(path/'run.py') in args and str(sp) in args and p.cwd()==str(path)
   assert p.environ()['CUDA_VISIBLE_DEVICES']==str(j['gpu']) and read(sp)==j['spec']
   workers[key]={'pid':p.pid,'create_time':p.create_time(),'cmdline':args,'cwd':p.cwd(),'gpu':j['gpu']}
  write(ROOT/'STATE_HANDOFF_PRE_V3.json',state)
  write(ROOT/'HANDOFF_PROCESS_PROOF_V3.json',{'at':now().isoformat(),'old_supervisor_pid':pid,'old_supervisor_cmdline':parent.cmdline(),'workers':workers})
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False;terminated=True
  parent.wait(timeout=10)
  names=['coverage_scheduler_v3.py','handoff_coverage_v3.py','COVERAGE_AMENDMENT_V3.json','STATE_COVERAGE_REVIEW_V3.json','STATE_HANDOFF_PRE_V3.json','HANDOFF_PROCESS_PROOF_V3.json']
  write(ROOT/'SCHEDULER_V3_SOURCE.json',{'at':now().isoformat(),'files':{n:sha(ROOT/n) for n in names}})
  os.execv(PYTHON,[PYTHON,'-u',str(ROOT/'coverage_scheduler_v3.py')])
 except BaseException:
  if stopped and not terminated:os.kill(pid,signal.SIGCONT)
  raise

if __name__=='__main__':main()
