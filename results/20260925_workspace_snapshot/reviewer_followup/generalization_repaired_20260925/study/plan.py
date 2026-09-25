"""Fixed contrasts; no test-result-dependent additions or stopping."""
import random,collections
from common import *

def variant(method,place,rank,bias=True):return dict(method=method,placement=place,rank=rank,bias=bias if method=='affine' and place in ['e','eu','shared'] else False)
def spec(model,task,v,seed,stage='common',ratio=1.,subset=None):
 cfg=read(HERE/'models.json')[model];c=read(Path(cfg['path'])/'config.json');d=c['hidden_size'];vocab=c['vocab_size'];hd=c.get('head_dim',d//c['num_attention_heads']);q=c['num_attention_heads']*hd;k=c['num_key_value_heads']*hd
 h=8*c['num_hidden_layers']*((d+q)+2*(d+k)+(q+d)+3*(d+c['intermediate_size']))
 extra=(v['rank']*(vocab+d) if v['method']=='vocab' else 2*v['rank']*d)*(2 if v['placement']=='eu' else 1) if v['method']!='none' else 0
 extra+=d if v['bias'] else 0
 name=f"{stage}_{task}_{model}_{v['method']}_{v['placement']}_r{v['rank']}_b{int(v['bias'])}_lr{ratio:g}_s{seed}"
 s=dict(name=name,model=model,task=task,arm='hidden',seed=seed,lr=2e-4,microbatch=TASK_SETTINGS[task]['microbatch'],smoke=False,stage=stage,boundary_lr_ratio=ratio,expected_total_parameters=h+extra,expected_boundary_parameters=extra,**v)
 if stage=='tuning':s['eval_splits']=['dev']
 if subset is not None:s['subset_indices']=subset
 return s

def main():
 jobs=[];tuning=[];templates=[]
 untied=['qwen25_7b_base','qwen3_8b_base','llama31_8b_base'];tied=['qwen3_4b_base','qwen25_15b_base','qwen3_06b_base']
 # Put first 36 new ordinary rank1 Qwen fits first; full main contrasts fixed at five paired seeds.
 for model in untied+tied:
  for task in TASKS:
   start=TASK_SETTINGS[task]['seed_start'];places=['e','u','eu'] if model in untied else ['shared']
   variants=[(variant('none','none',0),5)]
   for place in places:
    variants += [(variant('vocab',place,1),5),(variant('affine',place,16),5)]
    if place in ['eu','shared']:
     variants += [(variant('vocab',place,2),3),(variant('affine',place,8),5 if model in tied else 3),(variant('affine',place,32),5 if model in tied else 3)]
    elif model!='llama31_8b_base':variants += [(variant('affine',place,8),3)]
    if place!='u':variants += [(variant('affine',place,16,False),3)]
   for v,n in variants:
    for seed in range(start,start+n):jobs.append(spec(model,task,v,seed))
   # Same 3 boundary-LR candidates and 2 dev-only seeds for both method families.
   if model in ['qwen25_7b_base','qwen3_8b_base','qwen3_4b_base','llama31_8b_base']:
    place='eu' if model in untied else 'shared'
    for method,rank in [('vocab',1),('affine',16)]:
     v=variant(method,place,rank)
     for ratio in [.25,1.,4.]:
      for seed in [start+100,start+101]:tuning.append(spec(model,task,v,seed,'tuning',ratio))
     for seed in range(start+200,start+205):templates.append(spec(model,task,v,seed,'confirmation'))
 # Two nonoverlapping 1024-example halves of the existing training pool; same heldout dev/test.
 order=list(range(2048));random.Random(92817).shuffle(order)
 for task in TASKS:
  start=TASK_SETTINGS[task]['seed_start']
  for half in range(2):
   for v in [variant('none','none',0),variant('vocab','shared',1),variant('affine','shared',16)]:
    for seed in range(start+300,start+303):jobs.append(spec('qwen3_06b_base',task,v,seed,f'subset{half}',subset=sorted(order[half*1024:(half+1)*1024])))
 jobs.sort(key=lambda s:(0 if s['method']=='vocab' and s['rank']==1 and s['model'] in untied[:2] and s['seed']%100<3 else 1,s['stage']!='common'))
 # Smokes cover every model/task/method/placement/rank/bias and full evaluation memory shape.
 smoke=[];seen=set()
 for s in jobs+tuning:
  key=architecture(s)
  if key in seen:continue
  seen.add(key);z=dict(s,name='smoke_'+s['name'],smoke=True,stage='smoke');z.pop('subset_indices',None);z.pop('eval_splits',None);smoke.append(z)
 write(HERE/'FORMAL_JOBS.json',jobs);write(HERE/'TUNING_JOBS.json',tuning);write(HERE/'CONFIRMATION_TEMPLATES.json',templates);write(HERE/'SMOKE_JOBS.json',smoke)
 write(HERE/'PLAN_COUNTS.json',dict(formal_before_reuse=len(jobs),tuning=len(tuning),confirmation=len(templates),smokes=len(smoke),checkpoint_upper_bound_GiB=sum(s['expected_total_parameters']*8 for s in jobs+tuning+templates+smoke)/2**30,by_model=dict(collections.Counter(s['model'] for s in jobs))))
 print(read(HERE/'PLAN_COUNTS.json'))
def architecture(s):return tuple(s[k] for k in ['model','task','method','placement','rank','bias'])
if __name__=='__main__':main()
