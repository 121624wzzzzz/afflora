import torch
from torch import nn
from vocab_boundary import VocabLoRA,VocabHead
from common import *
from shared_boundary import SharedTransposeHead
import sys
sys.path.insert(0,str(HERE/'source'))
from affine_adapter import LowRankAffineMap

def main():
 torch.manual_seed(17);checks=[]
 for method in ['vocab','affine']:
  for side in ['input','output','shared']:
   for bias in ([False,True] if method=='affine' else [False]):
    v,d,r=19,7,3;w=torch.randn(v,d,dtype=torch.double);x=torch.randn(5,d,dtype=torch.double,requires_grad=True);ids=torch.tensor([0,3,3,18])
    a=(VocabLoRA(v,d,r,'input' if side!='output' else 'output') if method=='vocab' else LowRankAffineMap(d,r,8*r,0,bias)).double()
    head=nn.Linear(d,v,bias=False).double();head.weight.data.copy_(w);head.weight.requires_grad_(False)
    wrapper=VocabHead(head,a,{'output':0}) if method=='vocab' else SharedTransposeHead(head,a,{'output':0})
    if method=='vocab':assert a.embedding(ids).abs().max()==0 and a.logits(x).abs().max()==0
    else:assert torch.equal(a(x),x)
    # First-step update reaches the zero factor; nonzero-factor gradient is zero initially.
    y=a.embedding(ids).sum()+a.logits(x).sum() if method=='vocab' else a(x).sum()
    y.backward();zero=a.left if method=='vocab' else a.up.weight;nonzero=a.right if method=='vocab' else a.down.weight
    assert zero.grad.abs().sum()>0 and nonzero.grad.abs().sum()==0
    a.zero_grad();x.grad=None
    with torch.no_grad():
     for p in a.parameters():p.normal_(0,.1)
    if method=='vocab':
     effective=w+a.scale*a.left@a.right;out=wrapper(x);emb=w[ids]+a.embedding(ids)
    elif side=='output':
     effective=w+a.scale*w@a.up.weight@a.down.weight
     out=head(a(x));emb=None
     if bias:effective=None # output bias isn't part of the proposed model; handled directly below
    else:
     effective=w+a.scale*w@a.down.weight.T@a.up.weight.T
     if bias:effective=effective+a.bias
     out=wrapper(x);emb=a(w[ids])
    if effective is None:continue
    reference=x@effective.T;torch.testing.assert_close(out,reference,rtol=1e-11,atol=1e-11)
    if emb is not None:torch.testing.assert_close(emb,effective[ids],rtol=1e-11,atol=1e-11)
    params=list(a.parameters())+[x]
    actual=torch.autograd.grad(out.square().sum(),params,retain_graph=True)
    expected=torch.autograd.grad(reference.square().sum(),params)
    for u,z in zip(actual,expected):torch.testing.assert_close(u,z,rtol=1e-10,atol=1e-10)
    checks.append(dict(method=method,side=side,bias=bias,status='passed'))
 write(HERE/'ALGEBRA_AUDIT.json',dict(status='passed',checks=checks,at=now()))
 print('algebra, zero update, first gradients, dense forward/backward: passed')
if __name__=='__main__':main()
