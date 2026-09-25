"""Direct free vocabulary factors. Shared update is exactly one V x d matrix."""
import weakref,math
import torch
from torch import nn
from torch.nn import functional as F
class VocabLoRA(nn.Module):
    def __init__(self,v,d,r,side):
        super().__init__();self.scale=8.;self.active_scale=8.;self.bias_scale=1.
        self.left=nn.Parameter(torch.empty(v,r));self.right=nn.Parameter(torch.empty(r,d))
        # Embedding: PEFT convention, vocabulary factor zero / feature factor normal.
        # Output: linear LoRA convention, vocabulary factor zero / feature factor Kaiming.
        nn.init.zeros_(self.left)
        if side=='input':nn.init.normal_(self.right)
        else:nn.init.kaiming_uniform_(self.right,a=math.sqrt(5))
    def embedding(self,ids):return (F.embedding(ids,self.left)@self.right)*self.scale
    def logits(self,x):return F.linear(F.linear(x.to(self.right.dtype),self.right),self.left)*self.scale
class VocabHead(nn.Module):
    def __init__(self,base,adapter,counts):
        super().__init__();self.base_head=base;self.counts=counts
        object.__setattr__(self,'_adapter_ref',weakref.ref(adapter))
    @property
    def weight(self):return self.base_head.weight
    @property
    def bias(self):return self.base_head.bias
    def forward(self,x):
        self.counts['output']+=1;y=self.base_head(x.to(self.weight.dtype))
        return y+self._adapter_ref().logits(x).to(y.dtype)
