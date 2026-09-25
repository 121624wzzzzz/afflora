"""Fixed before Gemma responses: same LR grid, two selection seeds, five confirmation seeds."""
from common import *
SEARCH_SEEDS=[7400,7401];CONFIRM_SEEDS=list(range(7500,7505))
LRS=[2e-4,1e-4,4e-4,5e-5,3e-4,8e-4]
CANDIDATES=[{'id':f'c{i}','lr':lr} for i,lr in enumerate(LRS)]
def spec(stage,task,arm,candidate,seed):
 c=CANDIDATES[candidate] if arm!='base' else {'id':'base','lr':0.}
 return dict(name=f'{stage}_{task}_{arm}_{c["id"]}_s{seed}',stage=stage,task=task,model=MODEL,arm=arm,seed=seed,
             candidate=c['id'],lr=c['lr'],boundary_lr_ratio=1.,microbatch=TASK_SETTINGS[task]['microbatch'],smoke=stage=='smoke',
             eval_splits=['confirm'] if stage in ['confirmation','base'] else ['dev'])
def make():
 search=[spec('search',task,arm,c,seed) for seed in SEARCH_SEEDS for c in range(6) for task in TASKS for arm in ARMS]
 smokes=[spec('smoke',task,arm,c,7399) for task in TASKS for arm,c in [('hidden',0),('hidden_budget',0),('hidden_both',0),('hidden_both',3),('hidden_both',5)]]
 base=[spec('base',task,'base',0,7398) for task in TASKS]
 for name,value in [('SEARCH_JOBS.json',search),('SMOKE_JOBS.json',smokes),('BASE_JOBS.json',base),('CANDIDATES.json',CANDIDATES)]:write(HERE/name,value)
def validate_spec(s):
 filename={'search':'SEARCH_JOBS.json','smoke':'SMOKE_JOBS.json','confirmation':'CONFIRMATION_JOBS.json','base':'BASE_JOBS.json'}[s['stage']]
 assert s in read(HERE/filename)
 if s['stage'] in ['confirmation','base']:
  gate=read(HERE/'SELECTION.json');assert gate['status']=='frozen'
  assert gate['confirmation_jobs_sha256']==sha(HERE/'CONFIRMATION_JOBS.json')
  assert all(read(HERE/'audits'/f'{j["name"]}.json')['status']=='passed' for j in read(HERE/'SEARCH_JOBS.json'))
if __name__=='__main__':make()
