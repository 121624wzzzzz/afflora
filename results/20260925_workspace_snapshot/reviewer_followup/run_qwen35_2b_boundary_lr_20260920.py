"""Bounded two-fit posthoc diagnostic; never modifies parent study."""
import os,json,hashlib,subprocess,time,datetime,traceback
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'qwen35_2b_boundary_lr_20260920';BASE=P/'qwen35_fixed_2b_20260920'
PY='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def launch(script,args,log,gpu):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_COMPILE_THREADS='1',CUDA_VISIBLE_DEVICES=str(gpu))
 with log.open('w') as f:return subprocess.Popen([PY,'-u',str(R/script),*args],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
def compare(s):
 n=s['name'];old=BASE/'checkpoints'/f"search_wikisql_hidden_both_c5_s{s['seed']}";new=R/'checkpoints'/n
 a=read(old/'INITIALIZATION.json');b=read(new/'INITIALIZATION.json')
 assert a['initialization_sha256']==b['initialization_sha256']
 assert sha(old/'TRAIN_ORDER.json')==sha(new/'TRAIN_ORDER.json')
 t=read(new/'TRAINING.json');ot=read(old/'TRAINING.json')
 assert t['history'][0]['loss']==ot['history'][0]['loss']
 for record in t['history']:
  assert record['group_lrs']['input']==record['group_lrs']['output']==record['group_lrs']['hidden']*.25
 v=read(R/'evaluations'/n/'dev/SUMMARY.json');ov=read(BASE/'evaluations'/old.name/'dev/SUMMARY.json')
 return dict(seed=s['seed'],old_score=ov['primary'],new_score=v['primary'],delta=v['primary']-ov['primary'],initialization_order_first_loss_identical=True,old_max_grad=max(x['grad_norm'] for x in ot['history']),new_max_grad=max(x['grad_norm'] for x in t['history']),new_first8=t['history'][:8])
def main():
 assert not (R/'DIAGNOSTIC_STATE.json').exists(),'No implicit restart'
 for p,h in read(R/'BASELINE_CHECK.json')['reuse_inputs'].items():assert sha(Path(p))==h,p
 jobs=[dict(spec=s,state='pending') for s in read(R/'SEARCH_JOBS.json')];active={};results=[]
 while True:
  for name,(proc,j) in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   del active[name]
   if rc: j.update(state='failed',returncode=rc);continue
   if j['state']=='training':
    j['state']='auditing';active[name]=(launch('audit_one.py',['--name',name],R/'logs'/f'{name}.audit.log',j['gpu']),j)
   else:
    try:results.append(compare(j['spec']));j['state']='passed'
    except Exception:j.update(state='failed',error=traceback.format_exc())
  if not any(j['state']=='failed' for j in jobs):
   free_disk=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize/2**30
   if free_disk>=80:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    used={j['gpu'] for _,j in active.values()}
    available=[int(a) for a,b in (line.split(',') for line in raw.splitlines()) if int(b)>40960 and int(a) not in used]
    for j in jobs:
     if j['state']!='pending' or not available or len(active)>=2:continue
     gpu=available.pop(0);j.update(state='training',gpu=gpu,started=now());name=j['spec']['name']
     proc=launch('run.py',['--spec',str(R/'specs'/f'{name}.json')],R/'logs'/f'{name}.log',gpu);j['pid']=proc.pid;active[name]=(proc,j)
  write(R/'DIAGNOSTIC_STATE.json',dict(at=now(),jobs=jobs,results=results))
  if all(j['state']=='passed' for j in jobs):
   write(R/'RESULTS.json',dict(at=now(),scope='Posthoc paired development diagnostic, two seeds; no significance or isolated architecture claim.',results=results));break
  if any(j['state']=='failed' for j in jobs) and not active:break
  time.sleep(15)
if __name__=='__main__':
 try:main()
 except Exception:
  write(R/'DRIVER_ERROR.json',dict(at=now(),error=traceback.format_exc()));raise
