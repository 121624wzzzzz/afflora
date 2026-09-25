"""Fixed candidate order, seeds, role mapping and allowed worker specifications."""
from common import *

MODEL='llama31_8b_base'
SEARCH_SEEDS=[7200,7201]
CONFIRM_SEEDS=list(range(7300,7305))
CANDIDATES={arm:[{'id':f'c{i}','lr':lr,'boundary_lr_ratio':1.} for i,lr in enumerate([2e-4,1e-4,4e-4,5e-5,3e-4,8e-4])] for arm in ['hidden','hidden_budget']}
CANDIDATES['hidden_both']=[{'id':f'c{i}','lr':lr,'boundary_lr_ratio':ratio} for i,(lr,ratio) in enumerate([(2e-4,1.),(2e-4,.25),(1e-4,1.),(1e-4,.25),(4e-4,1.),(4e-4,.25)])]

def spec(stage,arm,candidate,seed,smoke=False):
    c=CANDIDATES[arm][candidate]
    return dict(name=f'{stage}_{arm}_{c["id"]}_s{seed}',stage=stage,task='wikisql',model=MODEL,arm=arm,seed=seed,
                candidate=c['id'],lr=c['lr'],boundary_lr_ratio=c['boundary_lr_ratio'],microbatch=4,smoke=smoke,
                eval_splits=[] if stage=='compatibility' else ['confirm'] if stage=='confirmation' else ['dev'])

def make():
    search=[spec('search',arm,c,seed) for seed in SEARCH_SEEDS for c in range(6) for arm in ARMS]
    smokes=[spec('smoke',a,c,7199,True) for a,c in [('hidden',3),('hidden',5),('hidden_budget',5),('hidden_both',0),('hidden_both',3),('hidden_both',5)]]
    compatibility=[spec('compatibility',a,0,7100) for a in ['hidden_budget','hidden_both']]
    for name,value in [('SEARCH_JOBS.json',search),('SMOKE_JOBS.json',smokes),('COMPATIBILITY_JOBS.json',compatibility),('CANDIDATES.json',CANDIDATES)]:write(HERE/name,value)

def validate_spec(s):
    filename={'search':'SEARCH_JOBS.json','smoke':'SMOKE_JOBS.json','compatibility':'COMPATIBILITY_JOBS.json','confirmation':'CONFIRMATION_JOBS.json'}[s['stage']]
    assert s in read(HERE/filename)
    if s['stage']=='confirmation':
        gate=read(HERE/'SELECTION.json');assert gate['status']=='frozen'
        assert gate['confirmation_jobs_sha256']==sha(HERE/filename)
        assert all(read(HERE/'audits'/f'{j["name"]}.json')['status']=='passed' for j in read(HERE/'SEARCH_JOBS.json'))

if __name__=='__main__':make()
