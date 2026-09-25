"""Run a fixed four-fit additional-rank learning-rate diagnostic with paired provenance and per-fit audits."""
import os,json,hashlib,subprocess,time,datetime,traceback,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'qwen35_2b_extra_rank_lr_20260920';BASE=P/'qwen35_fixed_2b_20260920';PY='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def launch(j,script,args,suffix):
 root=Path(j['root']);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_COMPILE_THREADS='1',CUDA_VISIBLE_DEVICES=str(j['gpu']))
 with (root/'logs'/f"{j['spec']['name']}{suffix}.log").open('w') as log:
  return subprocess.Popen([PY,'-u',str(root/script),*args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
def compare(j):
 root=Path(j['root']);s=j['spec'];n=s['name'];new=root/'checkpoints'/n;init=read(new/'INITIALIZATION.json');tr=read(new/'TRAINING.json');scores={}
 for arm in ['hidden','hidden_budget','hidden_both']:
  oldname=f"search_wikisql_{arm}_{s['candidate']}_s{s['seed']}";old=BASE/'checkpoints'/oldname;a=read(old/'INITIALIZATION.json')
  assert init['shared_hidden_initialization_sha256']==a['shared_hidden_initialization_sha256']
  assert sha(new/'TRAIN_ORDER.json')==sha(old/'TRAIN_ORDER.json')
  assert tr['history'][0]['loss']==read(old/'TRAINING.json')['history'][0]['loss']
  scores[arm]=read(BASE/'evaluations'/oldname/'dev/SUMMARY.json')['primary']
 audit=read(root/'audits'/f'{n}.json');f=root/'evaluations'/n/'dev/SUMMARY.json';assert audit['status']=='passed' and sha(f)==audit['summary_sha256']['dev'];scores['new_budget']=read(f)['primary']
 assert init['trainable_parameters']==8542720
 matched=P/'qwen35_2b_rank_diagnostic_20260920/mlp_narrow_r10';mn=f"diagnostic_mlp_narrow_r10_{s['candidate']}_s{s['seed']}";mi=read(matched/'checkpoints'/mn/'INITIALIZATION.json')
 assert init['initialization_sha256']==mi['initialization_sha256']
 assert sha(new/'TRAIN_ORDER.json')==sha(matched/'checkpoints'/mn/'TRAIN_ORDER.json')
 assert tr['history'][0]['loss']==read(matched/'checkpoints'/mn/'TRAINING.json')['history'][0]['loss']
 scores['matched_rank10_full_lr']=read(matched/'evaluations'/mn/'dev/SUMMARY.json')['primary']
 return dict(variant=root.name,seed=s['seed'],lr=s['lr'],scores=scores,shared_rank8_initialization_order_first_loss_identical=True,max_grad=max(x['grad_norm'] for x in tr['history']),clipped_steps=sum(x['joint_clip_factor']<1 for x in tr['history']))
def report(results):
 lines=['# 新增 rank 学习率诊断阶段结果','', '事后 WikiSQL 开发集诊断；配对种子，严格等参。每行保留全部已完成配置，不进行结果筛选。','', '|分配|LR|种子|普通|原等参|新等参|同 rank10 原学习率|','|---|---:|---:|---:|---:|---:|---:|']
 for r in sorted(results,key=lambda x:(x['variant'],x['lr'],x['seed'])):
  s=r['scores'];lines.append(f"|{r['variant']}|{r['lr']:g}|{r['seed']}|{s['hidden']:.4f}|{s['hidden_budget']:.4f}|{s['new_budget']:.4f}|{s['matched_rank10_full_lr']:.4f}|")
 lines+=['','新增 rank 使用共有 rank 学习率的四分之一。降低 E/U 学习率的独立诊断见 ../qwen35_2b_boundary_lr_20260920/REPORT_ZH.md；不能把原高 LR 叠加失稳当作该方法充分调参后的表现。两种子不作显著性或普遍优势声明。']
 (R/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
def main():
 assert not (R/'STATE.json').exists(),'No implicit restart'
 for p,h in read(R/'REUSE_CHECK.json')['inputs'].items():assert sha(Path(p))==h,p
 jobs=read(R/'JOBS.json');active={};results=[]
 while True:
  for n,(proc,j) in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   del active[n]
   if rc:j.update(state='failed',returncode=rc);continue
   if j['state']=='training':
    j['state']='auditing';active[n]=(launch(j,'audit_one.py',['--name',n],'.audit'),j)
   else:
    try:results.append(compare(j));j['state']='passed';report(results)
    except Exception:j.update(state='failed',error=traceback.format_exc())
  disk=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize/2**30
  if not any(j['state']=='failed' for j in jobs) and disk>=80:
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
   used={j['gpu'] for _,j in active.values()};available=[int(a) for a,b in (line.split(',') for line in raw.splitlines()) if int(b)>40960 and int(a) not in used]
   for j in jobs:
    if j['state']!='pending' or not available or len(active)>=2:continue
    j.update(state='training',gpu=available.pop(0),started=now());n=j['spec']['name'];proc=launch(j,'run.py',['--spec',str(Path(j['root'])/'specs'/f'{n}.json')],'');j['pid']=proc.pid;active[n]=(proc,j)
  write(R/'STATE.json',dict(at=now(),disk_free_gib=disk,jobs=jobs,results=results))
  if all(j['state']=='passed' for j in jobs):write(R/'RESULTS.json',dict(at=now(),status='passed',results=results));break
  if any(j['state']=='failed' for j in jobs) and not active:break
  time.sleep(30)
if __name__=='__main__':
 try:main()
 except Exception:write(R/'DRIVER_ERROR.json',dict(at=now(),error=traceback.format_exc()));raise
