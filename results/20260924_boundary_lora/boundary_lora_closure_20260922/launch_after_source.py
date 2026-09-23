import time,subprocess,os,fcntl
from common import *
from audit_source import audit
lock=(HERE/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=Path(read(HERE/'SOURCE.json')['study']);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
def status(stage,**kw):write(HERE/'LAUNCH_STATE.json',dict(at=now(),stage=stage,**kw))
while True:
 s=read(old/'STATE.json')
 if s['failed']:raise RuntimeError('Source queue has failures; inspect before continuing')
 if s['stage']=='complete':break
 status('waiting_for_source',source_done=len(s['done']),source_pending=len(s['pending']),source_active=len(s['active']));time.sleep(30)
status('auditing_source');audit(require_complete=True)
subprocess.run([PYTHON,'-B',str(HERE/'review_source.py')],env=env,check=True)
assert read(HERE/'INPUT_AUDIT.json')['status']=='passed'
# Short technical gates do not reuse effectiveness evidence.
write(HERE/'READY.json',dict(status='passed',at=now(),reused=0,source_initial_snapshots_missing=len(read(HERE/'SOURCE_AUDIT.json')['missing_initial_snapshots'])))
status('launching_training')
with (HERE/'logs/scheduler.log').open('a') as f:
 p=subprocess.Popen([PYTHON,'-B',str(HERE/'scheduler.py')],env=env,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
with (HERE/'logs/finalizer.log').open('a') as f:
 q=subprocess.Popen([PYTHON,'-B',str(HERE/'finalize.py')],env=env,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
status('training_started',scheduler_pid=p.pid,finalizer_pid=q.pid)

with (HERE/'logs/resource_watcher.log').open('a') as f:
 subprocess.Popen([PYTHON,'-B',str(HERE/'resource_watcher.py')],env=env,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
