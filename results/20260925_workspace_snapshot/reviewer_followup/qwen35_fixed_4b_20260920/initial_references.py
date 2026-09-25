"""Training-only, zero-update first-batch references for every admitted seed."""
import gc,torch
from common import *
from modeling import build,fixed_runtime_audit
from run import preflight
from initial_repeat import initial_repeat
SEEDS=[7599,7600,7601,7700,7701,7702,7703,7704]
def main():
 assert not (HERE/'INITIAL_REFERENCES.json').exists();records={}
 for seed in SEEDS:
  spec=dict(seed=seed,model=MODEL,arm='hidden');model,init=build(spec)
  for task in TASKS:
   data=read(HERE/f'tokens/{task}_{MODEL}_train.json');pf=preflight(model,spec,data)
   repeat=initial_repeat(model,data,seed,2);assert repeat['status']=='passed' and repeat['gradients_bitwise_equal']
   assert repeat['microbatch_losses'][0]==repeat['microbatch_losses'][1]
   records[f'{task}:{seed}']=dict(task=task,seed=seed,reference_arm='hidden',repeat=repeat,preflight=pf,initialization=init)
   write(HERE/'INITIAL_REFERENCES_PROGRESS.json',dict(records=records));print(task,seed,repeat['losses'][0],flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 write(HERE/'INITIAL_REFERENCES.json',dict(at=now(),status='passed',records=records,numerical_policy=fixed_runtime_audit(),
  scope='Training examples only; no optimizer updates or dev/confirm scoring. All arms must exactly match the 16 reference microbatch losses before any update.'))
if __name__=='__main__':main()
