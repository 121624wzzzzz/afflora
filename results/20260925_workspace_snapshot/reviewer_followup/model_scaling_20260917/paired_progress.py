"""Read-only progress: print only complete three-arm seed groups, no inference."""
from common import *

for task in TASKS:
    for model in MODELS:
        complete=[]
        for seed in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+5):
            names=[f'{task}_{model}_{a}_s{seed}' for a in ['hidden','hidden_budget','hidden_both']]
            if not all((HERE/'checkpoints'/n/'COMPLETE.json').exists() for n in names):continue
            assert all(read(HERE/'checkpoints'/n/'COMPLETE.json')['status']=='passed' for n in names)
            vals=[read(HERE/'evaluations'/n/'test'/'SUMMARY.json')['primary'] for n in names]
            complete.append({'seed':seed,'ordinary':vals[0],'budget':vals[1],'stack':vals[2],
                             'delta_ordinary':vals[2]-vals[0],'delta_budget':vals[2]-vals[1]})
        print(canonical({'task':task,'model':model,'complete_paired_seeds':len(complete),'planned':5,
                         'status':'descriptive per-seed view; consult final audits and TEST_ANALYSIS.json for inference','scores':complete}))
