from common import *
from plan import spec,variant,architecture
jobs=[];tuning=[];templates=[];smokes=[]
model='qwen3_4b_base'
for task in TASKS:
 start=306100 if task=='cluener' else 307100
 variants=[variant('affine','shared',16),variant('affine','shared',30),variant('vocab','shared',1),variant('vocab','shared',2)]
 for v in variants:
  for ratio in [.015625,.0625,.25,1.,2.,4.]:
   for seed in range(start,start+3):tuning.append(spec(model,task,v,seed,'tuning',ratio))
  for seed in range(start+100000,start+100005):templates.append(spec(model,task,v,seed,'confirmation'))
 for seed in range(start+100000,start+100005):jobs.append(spec(model,task,variant('none','none',0),seed,'confirmation'))
 for v in variants+[variant('none','none',0)]:
  s=spec(model,task,v,start+200000,'smoke');s.update(name='smoke_'+s['name'],smoke=True);smokes.append(s)
for name,value in [('FORMAL_JOBS.json',jobs),('PENDING_FORMAL.json',jobs),('TUNING_JOBS.json',tuning),('CONFIRMATION_TEMPLATES.json',templates),('SMOKE_JOBS.json',smokes)]:write(HERE/name,value)
all_jobs=jobs+tuning+templates+smokes
assert len(all_jobs)==204 and len({s['name'] for s in all_jobs})==204
assert all(s['eval_splits']==['dev'] for s in tuning)
write(HERE/'PLAN_COUNTS.json',dict(models=1,new_formal=len(jobs),dev_only_tuning=len(tuning),fresh_confirmation=len(templates),smokes=len(smokes),training_total=len(all_jobs),reused_results=0,checkpoint_GiB=sum(s['expected_total_parameters']*8 for s in all_jobs)/2**30))
print(read(HERE/'PLAN_COUNTS.json'))
