"""Zero-update, process-local kernel-configuration intervention; frozen studies untouched."""
import argparse,sys,importlib,json,hashlib,time
from pathlib import Path
import torch
from triton.runtime.autotuner import Autotuner

PARENT=Path(__file__).resolve().parent
def config(c):return dict(kwargs=c.kwargs,num_warps=c.num_warps,num_stages=c.num_stages,num_ctas=c.num_ctas)
def unwrap(k):
 while not isinstance(k,Autotuner):k=k.fn
 return k
def caches():
 result={};seen=set()
 for name,module in list(sys.modules.items()):
  if not name.startswith('fla.') or module is None:continue
  for attr,obj in vars(module).items():
   visited=set()
   while hasattr(obj,'fn') and id(obj) not in visited:
    visited.add(id(obj))
    if isinstance(obj,Autotuner):
     if id(obj) not in seen and obj.cache:
      seen.add(id(obj));result[name+'.'+attr]=[dict(key=str(k),config=config(c)) for k,c in obj.cache.items()]
     break
    obj=obj.fn
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kernel',default='fla.ops.common.chunk_o:chunk_fwd_kernel_o');ap.add_argument('--out',required=True);args=ap.parse_args()
 root=PARENT/'qwen35_08b_20260919';out=PARENT/'qwen35_numerics_20260920'/args.out;out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(root))
 from common import read,write,sha,MODEL,canonical,htext,now
 from modeling import build,batch,loss,adapter_state,digest,frozen_digest
 from run import preflight
 from initial_repeat import initial_repeat
 source=root/'checkpoints/search_wikisql_hidden_budget_c1_s7600'
 spec=read(source/'spec.json');init=read(source/'INITIALIZATION.json');data=read(root/f'tokens/wikisql_{MODEL}_train.json')
 model,audit=build(spec);assert digest(adapter_state(model))==init['initialization_sha256']
 before=frozen_digest(model);assert before==init['frozen_before'];pf=preflight(model,spec,data)
 local=initial_repeat(model,data,spec['seed'],2);assert local['status']=='passed'
 module,name=args.kernel.split(':');kernel=unwrap(getattr(importlib.import_module(module),name));original=list(kernel.configs)
 baseline_caches=caches();order=torch.randperm(len(data),generator=torch.Generator().manual_seed(spec['seed'])).tolist()[:32]
 assert htext(canonical(order))==local['order_sha256'];denom=sum(len(data[i]['target_ids']) for i in order)
 measurements=[]
 for i,c in enumerate(original):
  kernel.configs=[c];kernel.cache.clear();values=[];started=time.monotonic()
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   model.train();model.zero_grad(set_to_none=True)
   for start in range(0,32,2):
    ids,mask,labels=batch([data[n] for n in order[start:start+2]])
    with torch.autocast('cuda',dtype=torch.bfloat16):_,ce,_=loss(model,ids,mask,labels);value=ce.sum()/denom
    assert torch.isfinite(value);values.append(float(value.detach()))
   del value,ce
  row=dict(index=i,config=config(c),loss=sum(values),microbatch_losses=values,seconds=time.monotonic()-started)
  measurements.append(row);write(out/'PROGRESS.json',dict(kernel=args.kernel,measurements=measurements));print(json.dumps(row),flush=True)
 assert before==frozen_digest(model) and digest(adapter_state(model))==init['initialization_sha256']
 result=dict(at=now(),status='passed',scope='Same model, same batch, no optimizer updates; change one process-local Triton autotuner configuration only. No on-disk library/study changes.',
  source_files={str(source/n):sha(source/n) for n in ['spec.json','INITIALIZATION.json','INITIAL_REPEAT.json']},
  kernel=args.kernel,baseline_local_repeat=local,baseline_kernel_caches=baseline_caches,measurements=measurements,
  source_initial_repeat=read(source/'INITIAL_REPEAT.json'),preflight=pf,base_unchanged=True,adapters_unchanged=True)
 write(out/'RESULTS.json',result);print('probe complete',flush=True)

if __name__=='__main__':main()
