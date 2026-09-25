import sys,torch
from torch import nn
from common import *
sys.path.insert(0,str(HERE/'source'))
from affine_adapter import LowRankAffineMap
from shared_boundary import SharedTransposeHead
from safetensors.torch import save_file,load_file
import tempfile

def main():
 torch.manual_seed(991);records=[]
 for rank in [16,32]:
  d=40;v=71;aff=LowRankAffineMap(d,rank,rank*8,0,True).double();emb=nn.Embedding(v,d).double();head=nn.Linear(d,v,bias=False).double();head.weight=emb.weight;head.requires_grad_(False)
  counts={'output':0};wrapper=SharedTransposeHead(head,aff,counts);h=torch.randn(7,d,dtype=torch.double)
  assert torch.equal(wrapper(h),head(h))
  with torch.no_grad():aff.up.weight.normal_(0,.03);aff.bias.normal_(0,.02)
  merged=aff(emb.weight);expected=h@merged.T;actual=wrapper(h);err=float((actual-expected).abs().max().detach());assert err<1e-12
  params=list(aff.parameters());assert sum(p.numel() for p in params)==(2*rank+1)*d
  # Input and output losses each reach the same parameters; the joint derivative adds.
  a=aff(emb.weight[:7]).square().mean();b=wrapper(h).square().mean()
  ga=torch.autograd.grad(a,params,retain_graph=True);gb=torch.autograd.grad(b,params,retain_graph=True);gc=torch.autograd.grad(a+b,params)
  assert all(x.abs().sum()>0 and y.abs().sum()>0 for x,y in zip(ga,gb))
  assert all(torch.allclose(z,x+y,atol=1e-12,rtol=1e-12) for x,y,z in zip(ga,gb,gc))
  owner=nn.Module();owner.add_module('boundary_input',aff);owner.add_module('lm_head',wrapper)
  assert sum(p.numel() for p in owner.parameters() if p.requires_grad)==(2*rank+1)*d
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'adapter.safetensors';state={n:p.detach().contiguous() for n,p in aff.named_parameters()};save_file(state,str(path))
   with torch.no_grad():
    for p in params:p.zero_()
   aff.load_state_dict(load_file(str(path)))
   assert torch.equal(wrapper(h),actual)
  records.append(dict(rank=rank,merge_max_abs_error=err,parameters=(2*rank+1)*d,gradient_sum=True,reload=True))
 write(HERE/'SHARED_ALGEBRA_AUDIT.json',dict(status='passed',tests=records));print(records)
if __name__=='__main__':main()
