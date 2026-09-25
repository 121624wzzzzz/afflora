"""Persistent staged queue; technical failure blocks its architecture, never selects by test score."""
import os,subprocess,time,traceback,fcntl,shutil,collections
from common import *
from plan import architecture
lock=(HERE/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
assert read(HERE/'READY.json')['status']=='passed'
all_jobs=read(HERE/'SMOKE_JOBS.json')+read(HERE/'PENDING_FORMAL.json')+read(HERE/'TUNING_JOBS.json')
active={};failed=[];done=[];passed={tuple(x) for x in read(HERE/'INHERITED_SMOKES.json')['architectures']};blocked=set();confirmation_added=False
for s in all_jobs:
 ap=HERE/'audits'/f"{s['name']}.json"
 if ap.exists() and read(ap)['status']=='passed':
  done.append(s['name'])
  if s['smoke']:passed.add(architecture(s))
instrumentation_names={s['name'] for s in read(HERE/'SMOKE_JOBS.json')}
pending=[s for s in all_jobs if s['name'] not in done]
# Refuse to overwrite an interrupted fit; report it as a technical failure for inspection.
for s in list(pending):
 if (HERE/'checkpoints'/s['name']).exists():
  failed.append(dict(name=s['name'],phase='preexisting_partial',exit_code=None));pending.remove(s);blocked.add(architecture(s))
def memory():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a):int(b) for a,b in (r.split(',') for r in out.strip().splitlines())}
def required(s):
 return 38000 if s['model'] in ['qwen3_4b_base','qwen25_3b_base','llama32_3b_base'] else 22000 if s['model'] in ['qwen25_15b_base','qwen3_17b_base','llama32_1b_base'] else 14000
def status(stage):
 write(HERE/'STATE.json',dict(at=now(),stage=stage,done=done,failed=failed,pending=[s['name'] for s in pending],blocked_architectures=[list(k) for k in blocked],active={str(g):dict(name=v['spec']['name'],pid=v['p'].pid,phase=v['phase']) for g,v in active.items()},reused=read(HERE/'READY.json')['reused'],confirmation_added=confirmation_added,free_disk_GiB=shutil.disk_usage(HERE/'checkpoints').free/2**30))
def add_confirmation():
 global confirmation_added
 tuning=read(HERE/'TUNING_JOBS.json')
 if not all(s['name'] in done for s in tuning):return
 groups=collections.defaultdict(lambda:collections.defaultdict(list))
 for s in tuning:
  key=architecture(s);groups[key][s['boundary_lr_ratio']].append(read(HERE/'evaluations'/s['name']/'dev/SUMMARY.json')['primary'])
 choices={k:max(v,key=lambda r:(sum(v[r])/len(v[r]),-abs(r-1),-r)) for k,v in groups.items()}
 selected=[]
 for s in read(HERE/'CONFIRMATION_TEMPLATES.json'):
  s=dict(s);s['boundary_lr_ratio']=choices[architecture(s)];s['name']+=f"_selected{ s['boundary_lr_ratio']:g}";selected.append(s)
 write(HERE/'SELECTION_FROZEN.json',dict(at=now(),criterion='mean dev primary over three fixed seeds; ties favor ratio closest to 1',candidates=[.015625,.0625,.25,1.,2.,4.],selection=[dict(architecture=list(k),ratio=r,dev_scores=dict(groups[k])) for k,r in choices.items()],jobs=selected))
 for s in selected:
  if (HERE/'audits'/f"{s['name']}.json").exists():done.append(s['name'])
  else:pending.append(s)
 confirmation_added=True
try:
 while pending or active or not confirmation_added:
  for gpu,v in list(active.items()):
   code=v['p'].poll()
   if code is None:continue
   v['log'].close();s=v['spec'];n=s['name']
   if code:
    failed.append(dict(name=n,phase=v['phase'],exit_code=code));blocked.add(architecture(s));del active[gpu];continue
   if v['phase']=='train':
    f=(HERE/'logs'/f'{n}.audit.log').open('w');p=subprocess.Popen([PYTHON,'-B',str(HERE/'audit_one.py'),'--name',n],env=env,cwd=HERE,stdout=f,stderr=subprocess.STDOUT);v.update(p=p,log=f,phase='audit');continue
   done.append(n)
   if s['smoke']:passed.add(architecture(s))
   del active[gpu]
   subprocess.run([PYTHON,'-B',str(HERE/'report.py')],env=env,cwd=HERE,check=True)
  if not confirmation_added:add_confirmation()
  if shutil.disk_usage(HERE/'checkpoints').free<40*2**30 or shutil.disk_usage(HERE).free<5*2**30:
   status('disk_guard_waiting');time.sleep(30);continue
  mem=memory()
  for gpu,mb in sorted(mem.items(),key=lambda x:-x[1]):
   if gpu in active:continue
   eligible=[s for s in pending if architecture(s) not in blocked and mb>=required(s) and (s['smoke'] or architecture(s) in passed)]
   if not eligible:continue
   # Finish the first direct-baseline block promptly, while using spare cards for small-model smokes.
   def priority(s):return (0 if s['smoke'] else 1 if s['stage']=='tuning' else 2 if s['stage']=='confirmation' else 3,-required(s))
   s=min(eligible,key=priority);pending.remove(s);n=s['name'];write(HERE/'specs'/f'{n}.json',s)
   f=(HERE/'logs'/f'{n}.log').open('w');p=subprocess.Popen([PYTHON,'-B',str(HERE/('run_eval.py' if s.get('evaluation_only') else 'run.py')),'--spec',str(HERE/'specs'/f'{n}.json')],env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
   active[gpu]=dict(p=p,log=f,spec=s,phase='train');print(now(),'start',gpu,n,flush=True)
  if not active and pending and all(architecture(s) in blocked for s in pending):break
  if not active and not pending and failed:break
  status('running');time.sleep(30)
 status('complete' if not failed else 'needs_technical_review')
except Exception:
 status('scheduler_error');traceback.print_exc();raise
