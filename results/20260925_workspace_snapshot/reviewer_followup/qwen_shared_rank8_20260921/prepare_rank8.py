import sys,shutil,json,hashlib,subprocess,os
from pathlib import Path
R=Path(__file__).resolve().parent;O=R.parent/'qwen_shared_20260921/study';S=R/'study';assert not S.exists();S.mkdir()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert read(O/'FINAL_AUDIT.json')['status']=='passed'
for rel,h in read(O/'CODE_FROZEN.json')['files'].items():assert sha(O/rel)==h,rel
for rel,h in read(O/'DATA_FROZEN.json')['files'].items():assert sha(O/rel)==h,rel
for p in list(O.glob('*.py'))+list((O/'source').glob('*.py')):
 q=S/p.relative_to(O);q.parent.mkdir(exist_ok=True);shutil.copy2(p,q)
for n in ['data','tokens','raw']:(S/n).symlink_to(O/n,target_is_directory=True)
for n in ['checkpoints','evaluations','audits','specs','logs']:(S/n).mkdir()
for n in ['models.json','REUSE_AUDIT.json','DATA_FROZEN.json','BUDGET_PLAN.json','SOURCE_REUSE.json']:shutil.copy2(O/n,S/n)
models=['qwen25_15b_base','qwen3_06b_base'];configs=read(S/'models.json')
for m in models:
 for p,h in configs[m]['files'].items():assert sha(Path(p))==h,p
write(S/'MODEL_IDENTITY_AUDIT.json',dict(status='passed',models={m:configs[m]['files'] for m in models}))
for n in ['common.py','modeling.py','run.py','audit_one.py']:
 p=S/n;p.write_text(p.read_text().replace("'hidden_shared16','hidden_shared32'","'hidden_shared8','hidden_shared16','hidden_shared32'"))
p=S/'test_shared.py';p.write_text(p.read_text().replace('for rank in [16,32]:','for rank in [8]:'))
budget=read(S/'BUDGET_PLAN.json')
for m in models:budget[m]['hidden_shared8']=budget[m]['hidden']+17*configs[m]['hidden_size']
write(S/'BUDGET_PLAN.json',budget)
jobs=[];smokes=[]
for s in read(O/'FORMAL_JOBS.json'):
 if s['model'] not in models or s['arm']!='hidden_shared16':continue
 s=dict(s,arm='hidden_shared8',name=s['name'].replace('shared16','shared8'));jobs.append(s)
 if s['seed'] in [6100,7100]:smokes.append(dict(s,smoke=True,name=s['name']+'_smoke'))
assert len(jobs)==12 and len(smokes)==4
write(S/'FORMAL_JOBS.json',jobs);write(S/'SMOKE_JOBS.json',smokes)
# Rehash all consumed completed r16/r32 results and adapter checkpoints.
prior=[]
for g in read(O/'RESULTS.json')['groups']:
 for pair in g['paired_seeds']:
  for rank in [16,32]:
   n=f"{g['task']}_{g['model']}_hidden_shared{rank}_s{pair['seed']}";ck=O/'checkpoints'/n;a=read(O/'audits'/f'{n}.json');tr=read(ck/'TRAINING.json');assert a['status']=='passed' and sha(ck/'adapter.safetensors')==tr['adapter_sha256']
   p=O/'evaluations'/n/'test/SUMMARY.json';assert sha(p)==a['summary_sha256']['test'];s=read(p);assert sha(p.parent/'responses.jsonl')==s['responses_sha256'];prior.append(dict(model=g['model'],task=g['task'],seed=pair['seed'],rank=rank,primary=s['primary'],summary=str(p),summary_sha256=sha(p)))
write(S/'PRIOR_SHARED_AUDIT.json',dict(status='passed',runs=prior,source_final_audit_sha256=sha(O/'FINAL_AUDIT.json')))
(S/'PROTOCOL.md').write_text('Shared aLoRA rank8 extension, user requested 2026-09-21. Only tied Qwen2.5-1.5B and Qwen3-0.6B. Two tasks, three existing paired seeds, 12 formal fits and 4 smoke gates. Hidden LoRA remains r8 alpha16; shared boundary r8 alpha64, alpha/rank8, E bias retained, U uses transpose and common logit shift. Extra boundary budget17d vs r16 33d and r32 65d. All data, optimizer, LR2e-4, 64 steps and evaluation match prior frozen protocol. No untied shared jobs. Retain every seed regardless of gain sign. Each configuration smoke must pass independent audit before formal training. Prior results and checkpoints rehashed.\n')
print('prepared',flush=True)
