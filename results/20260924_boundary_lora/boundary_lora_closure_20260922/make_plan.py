from common import *
from plan import spec,variant,architecture
formal=[];tuning=[];templates=[]
models=['qwen3_06b_base','qwen25_15b_base','qwen25_7b_base','qwen3_8b_base','llama31_8b_base']
for model in models:
 place='shared' if model in models[:2] else 'u'
 for task in TASKS:
  start=TASK_SETTINGS[task]['seed_start']
  for seed in range(start+200000,start+200005):formal.append(spec(model,task,variant('none','none',0),seed,'confirmation'))
  for method,rank in [('vocab',1),('affine',16)]:
   for ratio in [.0625,.25,1.,4.]:
    for seed in [start+100000,start+100001]:tuning.append(spec(model,task,variant(method,place,rank),seed,'tuning',ratio))
   for seed in range(start+200000,start+200005):templates.append(spec(model,task,variant(method,place,rank),seed,'confirmation'))
# Complete paired H references for the previous eight tuned model/task cells.
for model in ['qwen25_7b_base','qwen3_8b_base','llama31_8b_base','qwen3_4b_base']:
 for task in TASKS:
  for seed in range(TASK_SETTINGS[task]['seed_start']+200,TASK_SETTINGS[task]['seed_start']+205):formal.append(spec(model,task,variant('none','none',0),seed,'prior_h_baseline'))
smokes=[]
for model,v in [('qwen3_06b_base',variant('none','none',0)),('qwen3_06b_base',variant('affine','shared',16)),('qwen3_06b_base',variant('vocab','shared',1)),('qwen25_7b_base',variant('affine','u',16))]:
 s=spec(model,'cluener',v,306100,'smoke');s['smoke']=True;smokes.append(s)
write(HERE/'FORMAL_JOBS.json',formal);write(HERE/'PENDING_FORMAL.json',formal);write(HERE/'SMOKE_JOBS.json',smokes);write(HERE/'TUNING_JOBS.json',tuning);write(HERE/'CONFIRMATION_TEMPLATES.json',templates)
write(HERE/'PLAN_COUNTS.json',dict(h_baselines=len(formal),dev_only_tuning=len(tuning),fresh_confirmation=len(templates),instrumentation_smokes=len(smokes),training_total=len(formal+tuning+templates+smokes),checkpoint_GiB=sum(s['expected_total_parameters']*8 for s in formal+tuning+templates+smokes)/2**30))
print(read(HERE/'PLAN_COUNTS.json'))
