from pathlib import Path
import sys
R=Path(__file__).resolve().parent;S=R/'study';sys.path.insert(0,str(S));from common import *
assert not (S/'CODE_FROZEN.json').exists()
configs=read(S/'models.json');audit=read(S/'MODEL_IDENTITY_AUDIT.json');models=['qwen25_7b_base','qwen3_8b_base']
for m in models:
 assert configs[m]['tie_word_embeddings'] is False
 for path,h in configs[m]['files'].items():assert sha(Path(path))==h,path
 audit['models'][m]=configs[m]['files']
write(S/'MODEL_IDENTITY_AUDIT.json',audit)
p=S/'modeling.py';s=p.read_text();assert 'model.config.hidden_size,16,128,0' in s;s=s.replace('model.config.hidden_size,16,128,0','model.config.hidden_size,8,64,0');s=s.replace('expect=(33*d','expect=(17*d').replace('+(32*d if arm','+(16*d if arm');p.write_text(s)
p=S/'run.py';s=p.read_text();needle="if spec['arm'] in ['hidden_shared8','hidden_shared16','hidden_shared32']:\n        old=";assert needle in s;s=s.replace(needle,"if spec['arm'].startswith('hidden'):\n        old=",1);p.write_text(s)
bud=read(S/'BUDGET_PLAN.json');jobs=read(S/'FORMAL_JOBS.json');smokes=read(S/'SMOKE_JOBS.json')
for m in models:
 d=configs[m]['hidden_size']
 for arm,extra in [('hidden_input',17*d),('hidden_output',16*d),('hidden_both',33*d)]:bud[m][arm]=bud[m]['hidden']+extra
for idx in range(3):
 for m in models:
  for task,start,micro in [('cluener',6100,2),('wikisql',7100,4)]:
   for arm in ['hidden_input','hidden_output','hidden_both']:
    n=f'{task}_{m}_{arm}_r8_s{start+idx}';s=dict(name=n,model=m,task=task,arm=arm,seed=start+idx,lr=2e-4,microbatch=micro,smoke=False);jobs.append(s)
    if idx==0:smokes.append(dict(s,name=n+'_smoke',smoke=True))
assert len(jobs)==48 and len(smokes)==16
write(S/'BUDGET_PLAN.json',bud);write(S/'FORMAL_JOBS.json',jobs);write(S/'SMOKE_JOBS.json',smokes)
p=S/'PROTOCOL.md';p.write_text(p.read_text()+'\nUser scope extension: also untied Qwen2.5-7B and Qwen3-8B, independently trained E-only/U-only/dual boundaries on top of H. Each active boundary has rank8/alpha64; E has bias and U does not. No shared transform on untied models. Additional36 formal and12 smoke fits; total48 formal plus16 smoke. Hidden rank8 unchanged. Single-side budgets17d and16d; dual33d. Untied sides remain separate parameters; prior r16 contrasts reused from the verified REUSE_AUDIT.\n')
print('48 formal fits and16 smoke gates prepared',flush=True)
