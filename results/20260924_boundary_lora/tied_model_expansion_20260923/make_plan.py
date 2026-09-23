from common import *
from plan import spec,variant
models=list(read(HERE/'models.json'));formal=[];tuning=[];confirmation=[]
for m in models:
 for task in TASKS:
  seed=TASK_SETTINGS[task]['seed_start']
  variants=[(variant('none','none',0),5),(variant('affine','shared',8),5),(variant('affine','shared',16),5),(variant('affine','shared',32),5),(variant('vocab','shared',1),5),(variant('vocab','shared',2),3),(variant('affine','shared',16,False),3)]
  for v,n in variants:
   for s in range(seed,seed+n):formal.append(spec(m,task,v,s,'common'))
  for s in range(seed+200000,seed+200005):formal.append(spec(m,task,variant('none','none',0),s,'confirmation'))
  for method,rank in [('vocab',1),('affine',16)]:
   for ratio in [.0625,.25,1.,4.]:
    for s in [seed+100000,seed+100001]:tuning.append(spec(m,task,variant(method,'shared',rank),s,'tuning',ratio))
   for s in range(seed+200000,seed+200005):confirmation.append(spec(m,task,variant(method,'shared',rank),s,'confirmation'))
write(HERE/'FORMAL_JOBS.json',formal);write(HERE/'ALL_TUNING_JOBS.json',tuning);write(HERE/'ALL_CONFIRMATION_TEMPLATES.json',confirmation)
print('coverage',len(models),'models;',len(formal),'common/H;',len(tuning),'tuning;',len(confirmation),'confirmation')
