"""Read-only completed seed blocks; never fills missing cells or infers significance."""
from reporting_common import *
for phase in TASKS:
 data=collect(phase)
 arms=ARMS if phase=='core' else ['base','hidden','hidden_budget','hidden_both']
 for task in TASKS[phase]:
  for model in MODELS:
   n=5 if phase=='core' and model in ['qwen25_7b_base','qwen3_8b_base'] else 3;complete=[]
   for seed in range(SEEDS[task],SEEDS[task]+n):
    keys=[(task,model,arm,None if arm=='base' else seed) for arm in arms]
    if all(k in data for k in keys):complete.append({'seed':seed,'scores':{ARM_LABELS[k[2]]:data[k]['primary'] for k in keys}})
   if complete:print(json.dumps({'phase':phase,'task':task,'model':model,'complete_paired_seeds':len(complete),'planned':n,'rows':complete},ensure_ascii=False))
