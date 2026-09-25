"""Pre-fit synthetic equivalence and gradient checks for native accelerated recurrence."""
import time,torch,inspect
from common import *
from modeling import native

def error(a,b):
 a=a.float();b=b.float();return {'max_abs':float((a-b).abs().max()),'relative_rms':float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}
def main():
 assert not (HERE/'CODE_FROZEN.json').exists();torch.set_num_threads(4);torch.manual_seed(91723);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 records=[]
 for length in [64,131,671]:
  vals=[torch.randn(2,length,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
  vals += [-torch.rand(2,length,4,device='cuda'),torch.rand(2,length,4,device='cuda',dtype=torch.bfloat16)]
  ref=[v.detach().clone().requires_grad_() for v in vals];fast=[v.detach().clone().requires_grad_() for v in vals]
  go=torch.randn_like(vals[2]);kwargs=dict(output_final_state=True,use_qk_l2norm_in_kernel=True)
  start=time.monotonic();y,hs=native.chunk_gated_delta_rule(*fast,**kwargs);(y*go).float().sum().backward();torch.cuda.synchronize();first=time.monotonic()-start
  z,rs=native.torch_chunk_gated_delta_rule(*ref,**kwargs);(z*go).float().sum().backward();torch.cuda.synchronize()
  out=error(y,z);state=error(hs,rs);grads=[error(a.grad,b.grad) for a,b in zip(fast,ref)]
  assert out['relative_rms']<.03 and state['relative_rms']<.03,(length,out,state)
  assert all(e['relative_rms']<.05 for e in grads),(length,grads)
  assert all(torch.isfinite(v.grad).all() for v in fast+ref)
  record={'length':length,'output':out,'state':state,'gradients':dict(zip(['q','k','v','g','beta'],grads)),'initial_compile_and_run_seconds':first}
  records.append(record);print(canonical(record),flush=True)
 write(HERE/'ACCELERATED_KERNEL_AUDIT.json',{'at':now(),'status':'passed','scope':'Synthetic BF16 native FLA versus native PyTorch recurrence; no task score.',
  'thresholds':{'output_and_state_relative_rms':.03,'gradient_relative_rms':.05},'records':records})
if __name__=='__main__':main()
