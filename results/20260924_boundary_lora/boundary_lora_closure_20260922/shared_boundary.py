"""One registered affine module, used at E and transposed at U.
For untied base weights this shares the transform, not the original matrices.
"""
import weakref
import torch
from torch import nn
from torch.nn import functional as F
class SharedTransposeHead(nn.Module):
    def __init__(self,base_head,affine,counts):
        super().__init__();self.base_head=base_head
        object.__setattr__(self,'_affine_ref',weakref.ref(affine));self.counts=counts
    @property
    def weight(self):return self.base_head.weight
    @property
    def bias(self):return self.base_head.bias
    def forward(self,hidden):
        self.counts['output']+=1;aff=self._affine_ref();assert aff is not None
        x=hidden.to(aff.down.weight.dtype)
        delta=F.linear(F.linear(x,aff.up.weight.T),aff.down.weight.T)
        logits=self.base_head((x+aff.scale*delta).to(self.weight.dtype))
        if aff.bias is not None:
            logits=logits+(x*(aff.bias_scale*aff.bias)).sum(-1,keepdim=True).to(logits.dtype)
        return logits
