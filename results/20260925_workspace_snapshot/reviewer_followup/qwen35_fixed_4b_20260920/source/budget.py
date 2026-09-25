"""Allocate the exact boundary budget without changing shared rank-8 tensors."""
import hashlib
import torch
from torch import nn
from architecture import architecture_plan

def hidden_digest(model,base_rank=None):
 h=hashlib.sha256()
 for name,value in sorted(model.named_parameters()):
  if 'lora_' not in name:continue
  if base_rank is not None:value=value[:base_rank,:] if '.lora_A.' in name else value[:,:base_rank]
  h.update(name.encode());h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def add_hidden_budget(model,base_rank,seed):
 plan=architecture_plan(model.config);before=hidden_digest(model);cfg=model.peft_config['default'];assert not cfg.rank_pattern and not cfg.alpha_pattern
 modules=dict(model.named_modules());spent=0
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(seed+3_000_017)
  for name,new_rank in plan['rank_pattern'].items():
   layer=modules[name];a,b=layer.lora_A['default'],layer.lora_B['default'];assert a.weight.device.type=='cpu' and layer.r['default']==base_rank
   na=nn.Linear(a.in_features,new_rank,bias=False,dtype=a.weight.dtype);nb=nn.Linear(new_rank,b.out_features,bias=False,dtype=b.weight.dtype)
   with torch.no_grad():na.weight[:base_rank].copy_(a.weight);nb.weight.zero_();nb.weight[:,:base_rank].copy_(b.weight)
   layer.lora_A['default']=na;layer.lora_B['default']=nb;layer.r['default']=new_rank;layer.lora_alpha['default']=2*new_rank;layer.scaling['default']=2.
   cfg.rank_pattern[name]=new_rank;cfg.alpha_pattern[name]=2*new_rank;spent+=a.in_features+b.out_features
 assert hidden_digest(model,base_rank)==before and spent==plan['extra']
 return {'target_extra_parameters':plan['extra'],'actual_extra_parameters':spent,'rank_pattern':plan['rank_pattern'],'shared_initialization_sha256':before}
