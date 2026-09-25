"""Eight-step training-only reproducibility pilot under an explicit numerical policy."""
import argparse,sys,math,json,time
from pathlib import Path
import torch
from safetensors.torch import save_file
PARENT=Path(__file__).resolve().parent
sys.path.insert(0,str(PARENT/'qwen35_numerics_20260920'))
from fixed_runtime import install,audit as runtime_audit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['hidden','hidden_budget','hidden_both'],required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 out=PARENT/'qwen35_numerics_20260920'/args.out;out.mkdir(parents=True,exist_ok=False);install()
 root=PARENT/'qwen35_08b_20260919';sys.path.insert(0,str(root))
 from common import read,write,MODEL,now
 from modeling import build,batch,loss,adapter_state,digest,frozen_digest,is_adapter
 from run import preflight,parameter_group
 from initial_repeat import initial_repeat
 spec=dict(model=MODEL,arm=args.arm,seed=7600);model,initial=build(spec);data=read(root/f'tokens/wikisql_{MODEL}_train.json')
 pf=preflight(model,spec,data);repeat=initial_repeat(model,data,spec['seed'],2);assert repeat['status']=='passed'
 before=frozen_digest(model);start_state=digest(adapter_state(model));order=torch.randperm(len(data),generator=torch.Generator().manual_seed(7600)).tolist()
 named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];opt=torch.optim.AdamW([p for n,p in named],lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
 records=[];started=time.monotonic()
 for step in range(8):
  model.train();opt.zero_grad(set_to_none=True);indices=order[step*32:(step+1)*32];denom=sum(len(data[i]['target_ids']) for i in indices);total=0.
  factor=(step+1)/2 if step<2 else .5*(1+math.cos(math.pi*(step-2)/62))
  for g in opt.param_groups:g['lr']=2e-4*factor
  for begin in range(0,32,2):
   ids,mask,labels=batch([data[i] for i in indices[begin:begin+2]])
   with torch.autocast('cuda',dtype=torch.bfloat16):_,ce,_=loss(model,ids,mask,labels);v=ce.sum()/denom
   assert torch.isfinite(v);v.backward();total+=float(v.detach())
  assert all(p.grad is None for n,p in model.named_parameters() if not is_adapter(n))
  gradients=digest({n:p.grad for n,p in named});norm=torch.nn.utils.clip_grad_norm_([p for n,p in named],1.,error_if_nonfinite=True);opt.step()
  record=dict(step=step+1,loss=total,grad_norm=float(norm),gradients_sha256=gradients,adapter_sha256=digest(adapter_state(model)),seconds=time.monotonic()-started)
  records.append(record);write(out/'PROGRESS.json',dict(arm=args.arm,records=records));print(json.dumps(record),flush=True)
 assert frozen_digest(model)==before
 save_file(adapter_state(model),str(out/'adapter.safetensors'))
 result=dict(at=now(),status='passed',scope='Training-only numerical pilot: same 256 training examples, eight updates, no development or confirmation scoring; pilot checkpoints never reused for effects.',
  spec=spec,preflight=pf,initial_repeat=repeat,initial_adapter_sha256=start_state,frozen_base_sha256=before,records=records,numerical_policy=runtime_audit())
 write(out/'RESULTS.json',result);print('fixed-policy pilot complete',flush=True)
if __name__=='__main__':main()
