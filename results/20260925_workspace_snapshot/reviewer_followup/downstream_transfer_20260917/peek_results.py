"""Read-only interim means; complete seed triplets only, no inferential claims."""
from common import *

for task in TASKS:
    for model in MODELS:
        results=[]
        for seed in range(7100,7105):
            paths=[HERE/'evaluations'/f'{task}_{model}_{arm}_s{seed}'/'test/SUMMARY.json' for arm in ['hidden','hidden_budget','hidden_both']]
            if all(p.exists() for p in paths):results.append([read(p)['primary'] for p in paths])
        if results:
            n=len(results);means=[sum(v[i] for v in results)/n for i in range(3)]
            print(task,model,'paired_seeds',n,'hidden/budget/both',[round(v,3) for v in means],
                'deltas_vs_hidden',[round(v[2]-v[0],3) for v in results],
                'deltas_vs_budget',[round(v[2]-v[1],3) for v in results])
