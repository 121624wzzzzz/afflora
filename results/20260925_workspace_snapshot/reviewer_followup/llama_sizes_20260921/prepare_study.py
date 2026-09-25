"""Prepare verified reuse and a new exact-budget three-size Llama matrix."""
import os,json,hashlib,subprocess,shutil,datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parent;ROOT=R/'study';OLD=R.parent/'llama_transfer_20260918';PY='/home/wz/anaconda3/envs/torch24/bin/python'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(script,*args):
 with (ROOT/'logs'/f'{script}.prepare.log').open('w') as log:
  subprocess.run([PY,'-B',str(ROOT/script),*args],env=ENV,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
assert (R/'MODEL_1B.json').exists();assert not (ROOT/'READY.json').exists()
models=read(OLD/'models.json');models={'llama32_1b_base':read(R/'MODEL_1B.json'),**models};seal=read(OLD/'SEAL_MANIFEST.json')['files'];checked={}
def verify_old(p):
 rel=str(p.relative_to(OLD));h=sha(p);assert h==seal[rel]['sha256'],rel;checked[rel]=h
for f in [OLD/'models.json',OLD/'MODEL_IDENTITY_AUDIT.json',OLD/'CODE_FROZEN.json',OLD/'DATA_FROZEN.json']:verify_old(f)
for rel,h in read(OLD/'DATA_FROZEN.json')['files'].items():assert sha(OLD/rel)==h,rel
for alias,cfg in models.items():
 for path,h in cfg['files'].items():assert sha(Path(path))==h,path
write(ROOT/'models.json',models);write(ROOT/'MODEL_IDENTITY_AUDIT.json',dict(at=datetime.datetime.now().astimezone().isoformat(),status='passed',models={k:dict(repo=v['repo'],revision=v['revision'],files=v['files']) for k,v in models.items()},scope='1B official Meta git/LFS identities; existing3B/8B files rehashed against sealed official identity records.'))
run('prepare.py');run('test_scoring.py')
for alias in ['llama32_3b_base','llama31_8b_base']:
 for task in ['cluener','wikisql']:
  for split in ['train','dev','test']:
   f=f'tokens/{task}_{alias}_{split}.json';verify_old(OLD/f);assert sha(ROOT/f)==sha(OLD/f),f
inventory=read(ROOT/'FORMAL_JOBS.json');fresh=[];reused=[]
for folder in ['checkpoints','evaluations']:(ROOT/folder).mkdir(exist_ok=True)
for j in inventory:
 s=j['spec'];reuse=s['model']=='llama31_8b_base' or (s['model']=='llama32_3b_base' and s['arm']!='hidden_budget')
 if not reuse:fresh.append(j);continue
 n=s['name'];assert read(OLD/'checkpoints'/n/'spec.json')==s
 for folder in ['checkpoints','evaluations']:
  src=OLD/folder/n
  for f in src.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts:verify_old(f)
  (ROOT/folder/n).symlink_to(src,target_is_directory=True)
 verify_old(OLD/'audits'/f'{n}.json');j['reused']=True;reused.append(j)
assert len(fresh)==26 and len(reused)==34
# All three1B arms get technical probes;3B only the changed strict-budget arm.
smokes=[j for j in read(ROOT/'SMOKE_JOBS.json') if j['spec']['model']=='llama32_1b_base' or (j['spec']['model']=='llama32_3b_base' and j['spec']['arm']=='hidden_budget')]
for task in ['cluener','wikisql']:
 j=next(j for j in smokes if j['spec']['model']=='llama32_1b_base' and j['spec']['task']==task and j['spec']['arm']=='hidden_both');k=json.loads(json.dumps(j));k['spec']['arm']='hidden';k['spec']['name']=k['spec']['name'].replace('hidden_both','hidden');smokes.append(k)
assert len(smokes)==8
write(ROOT/'FORMAL_JOBS.json',fresh);write(ROOT/'INVENTORY.json',inventory);write(ROOT/'SMOKE_JOBS.json',smokes);write(ROOT/'REUSED_JOBS.json',reused)
def reaudit(j):
 n=j['spec']['name'];log=ROOT/'logs'/f'{n}.reuse_audit.log'
 with log.open('w') as f:subprocess.run([PY,'-B',str(ROOT/'audit_one.py'),'--name',n],env=ENV,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 return n
with ThreadPoolExecutor(max_workers=4) as pool:
 for n in pool.map(reaudit,reused):print('reuse audited',n,flush=True)
write(ROOT/'REUSE_AUDIT.json',dict(status='passed',parent_manifest_sha256=sha(OLD/'SEAL_MANIFEST.json'),verified_parent_files=checked,reused_jobs=[j['spec']['name'] for j in reused],fresh_formal_jobs=len(fresh),scope='Full original checkpoint/output hashes, re-tokenization and independent output rescoring; no old fits treated as new independent replicates.'))
# Include provenance of reuse in frozen source input set through SOURCE_REUSE.
reuse=read(ROOT/'SOURCE_REUSE.json');reuse['reuse_audit_sha256']=sha(ROOT/'REUSE_AUDIT.json');write(ROOT/'SOURCE_REUSE.json',reuse)
run('freeze.py');run('analyze.py')
print('READY',len(fresh),'new formal jobs',len(smokes),'smokes',len(reused),'verified reused',flush=True)
