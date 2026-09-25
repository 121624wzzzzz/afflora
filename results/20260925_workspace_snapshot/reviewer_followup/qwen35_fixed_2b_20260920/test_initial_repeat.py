"""Same first batch and shared initialization across all arms, before any fit."""
import gc,torch
from common import *
from modeling import build
from run import preflight
from initial_repeat import initial_repeat

def main():
 assert not (HERE/'CODE_FROZEN.json').exists();records=[]
 for task in TASKS:
  losses=[]
  for arm in ARMS:
   s={'seed':7600,'model':MODEL,'arm':arm};model,audit=build(s)
   data=read(HERE/f'tokens/{task}_{MODEL}_train.json')
   pf=preflight(model,s,data);r=initial_repeat(model,data,7600,2)
   records.append(dict(task=task,arm=arm,preflight=pf,repeat=r,initialization=audit['shared_hidden_initialization_sha256']))
   write(HERE/'INITIAL_REPEAT_PROGRESS.json',{'records':records});assert r['status']=='passed',r
   from reference_check import check_reference
   records[-1]['cross_process_initial_reference']=check_reference(r,task,7600)
   if arm=='hidden':assert r['gradient_sha256']==read(HERE/'INITIAL_REFERENCES.json')['records'][f'{task}:7600']['repeat']['gradient_sha256']
   losses.append(r['losses'][0]);print(canonical(records[-1]),flush=True)
   del model;gc.collect();torch.cuda.empty_cache()
  assert max(losses)==min(losses),(task,losses)
  assert len({r['initialization'] for r in records if r['task']==task})==1
 write(HERE/'INITIAL_REPEAT_GATE.json',{'at':now(),'status':'passed','records':records,
  'cross_arm_first_loss_tolerance':0,'independent_process_reference_required':True,'scope':'Two zero-update first-batch backward passes per task/arm, seed7600. No development/confirmation predictions.'})
if __name__=='__main__':main()
