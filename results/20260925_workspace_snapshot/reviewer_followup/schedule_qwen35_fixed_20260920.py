"""One finite GPU queue across four immutable, separately audited size studies."""
import os,json,time,subprocess,fcntl,hashlib
from pathlib import Path
from datetime import datetime

PARENT=Path(__file__).resolve().parent
MASTER=PARENT/'qwen35_fixed_multiscale_20260920'
PYTHON='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
KEYS=['08b','2b','4b','9b']
REQUIRED_FREE_MIB={'08b':32768,'2b':40960,'4b':57344,'9b':69632}
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def event(kind,**kw):
 r=dict(at=now(),event=kind,**kw)
 with (MASTER/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
def memory():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return {int(x.split(',')[0]):int(x.split(',')[1]) for x in r.stdout.splitlines()}
def process(root,script,log,args=(),gpu=None):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_COMPILE_THREADS='1')
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 f=log.open('w');p=subprocess.Popen([PYTHON,'-u',str(root/script),*args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 return p,f
def main():
 MASTER.mkdir(exist_ok=True)
 lock=(MASTER/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (MASTER/'STATE.json').exists()
 start=now();studies={};active={};audits={};gates={};failed=False;turn=0
 for key in KEYS:
  p=PARENT/f'qwen35_fixed_{key}_20260920';assert not (p/'STATE.json').exists()
  assert read(p/'DISPATCHER_IDENTITY.json')['sha256']==sha(Path(__file__))
  for name in ['logs','specs']:(p/name).mkdir(exist_ok=True)
  studies[key]=dict(root=p,phase='waiting_ready',jobs=[])
 while True:
  paused=(MASTER/'CONTROL.json').exists() and read(MASTER/'CONTROL.json').get('pause_admission',False)
  if any((s['root']/'PREPARATION_FAILED.json').exists() for s in studies.values()):failed=True
  for key,s in studies.items():
   p=s['root']
   if not failed and s['phase']=='waiting_ready' and (p/'READY.json').exists():
    assert read(p/'READY.json')['status']=='passed'
    assert read(p/'DISPATCHER_IDENTITY.json')['sha256']==sha(Path(__file__))
    s.update(phase='preflight',jobs=[dict(spec=j,state='pending') for j in read(p/'SMOKE_JOBS.json')+read(p/'SEARCH_JOBS.json')])
    event('size_admitted',model=key)
  for ident,(proc,key,j,f) in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   f.close();del active[ident];j.update(returncode=rc,worker_finished_at=now());p=studies[key]['root'];name=j['spec']['name']
   if rc!=0 or not (p/'checkpoints'/name/'COMPLETE.json').exists():
    j['state']='failed';failed=True;event('worker_failed',model=key,name=name,returncode=rc);continue
   proc,f=process(p,'audit_one.py',p/'logs'/f'{name}.audit.log',['--name',name])
   j.update(state='auditing',audit_pid=proc.pid);audits[ident]=(proc,key,j,f)
  for ident,(proc,key,j,f) in list(audits.items()):
   rc=proc.poll()
   if rc is None:continue
   f.close();del audits[ident];ap=studies[key]['root']/'audits'/f'{j["spec"]["name"]}.json'
   ok=rc==0 and ap.exists() and read(ap)['status']=='passed';j.update(state='passed' if ok else 'failed',audit_returncode=rc,finished_at=now())
   if not ok:failed=True
   event(j['state'],model=key,name=j['spec']['name'])
  for key,(proc,f,kind) in list(gates.items()):
   rc=proc.poll()
   if rc is None:continue
   f.close();del gates[key];s=studies[key];p=s['root']
   if rc!=0:failed=True;s['phase']='gate_failed';event('gate_failed',model=key,gate=kind,returncode=rc);continue
   if kind=='preflight_gate.py':s['phase']='search'
   elif kind=='select_config.py':
    s['jobs'].extend(dict(spec=j,state='pending') for j in read(p/'CONFIRMATION_JOBS.json')+read(p/'BASE_JOBS.json'));s['phase']='confirmation'
   elif kind=='analyze.py':
    proc,f=process(p,'final_audit.py',p/'final_audit.log');gates[key]=(proc,f,'final_audit.py');s['phase']='auditing_final'
   else:s['phase']='complete'
   event('gate_passed',model=key,gate=kind,phase=s['phase'])
  if not failed:
   for key,s in studies.items():
    if key in gates:continue
    p=s['root'];phase=s['phase'];jobs=s['jobs'];script=None
    if phase=='preflight' and all(j['state']=='passed' for j in jobs if j['spec']['stage']=='smoke'):script='preflight_gate.py'
    elif phase=='search' and all(j['state']=='passed' for j in jobs if j['spec']['stage']=='search'):script='select_config.py'
    elif phase=='confirmation' and all(j['state']=='passed' for j in jobs):
     write(p/'SCHEDULER_COMPLETE.json',dict(at=now(),status='passed',counts={'passed':len(jobs),'failed':0}));script='analyze.py'
    if script:
     proc,f=process(p,script,p/(script+'.log'));gates[key]=(proc,f,script);s['phase']='gate_running'
   free=memory();busy={j['gpu'] for _,_,j,_ in active.values()}
   # Technical probes run on these explicitly reserved devices until READY.
   busy.update(gpu for gpu,key in [(7,'08b'),(1,'2b'),(2,'4b'),(3,'9b')] if studies[key]['phase']=='waiting_ready')
   if any(s['phase']=='waiting_ready' for s in studies.values()):busy.add(4)
   for gpu in range(8):
    if paused:break
    if gpu in busy or free.get(gpu,0)<min(REQUIRED_FREE_MIB.values()):continue
    picked=None
    for offset in range(len(KEYS)):
     ix=(turn+offset)%len(KEYS);key=KEYS[ix];s=studies[key]
     if free.get(gpu,0)<REQUIRED_FREE_MIB[key]:continue
     allowed={'preflight':['smoke'],'search':['search'],'confirmation':['confirmation','base']}.get(s['phase'],[])
     candidates=[j for j in s['jobs'] if j['state']=='pending' and j['spec']['stage'] in allowed]
     if candidates:picked=(key,candidates[0]);turn=(ix+1)%len(KEYS);break
    if picked is None:continue
    key,j=picked;p=studies[key]['root'];spec=j['spec'];name=spec['name'];sp=p/'specs'/f'{name}.json';write(sp,spec)
    proc,f=process(p,'run.py',p/'logs'/f'{name}.log',['--spec',str(sp)],gpu)
    j.update(state='running',gpu=gpu,pid=proc.pid,started_at=now(),admission_free_mib=free[gpu],required_free_mib=REQUIRED_FREE_MIB[key]);active[key+':'+name]=(proc,key,j,f);event('started',model=key,name=name,gpu=gpu,pid=proc.pid)
  states={}
  for key,s in studies.items():
   counts={state:sum(j['state']==state for j in s['jobs']) for state in ['pending','running','auditing','passed','failed']}
   state=dict(started_at=start,at=now(),phase=s['phase'],jobs=s['jobs'],counts=counts);write(s['root']/'STATE.json',state)
   states[key]=dict(phase=s['phase'],counts=counts)
  write(MASTER/'STATE.json',dict(started_at=start,at=now(),failed=failed,admission_paused=paused,studies=states))
  if not active and not audits and not gates and (failed or all(s['phase']=='complete' for s in studies.values())):
   write(MASTER/'SCHEDULER_COMPLETE.json',dict(at=now(),status='failed' if failed else 'passed',studies=states));break
  time.sleep(10)
 if failed:raise SystemExit(1)

if __name__=='__main__':main()
