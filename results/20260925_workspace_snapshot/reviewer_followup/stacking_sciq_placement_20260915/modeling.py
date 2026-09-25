"""Common initialization and the same affine modules at different positions."""
import sys,hashlib
import torch
from torch import nn
from transformers import AutoModelForCausalLM,set_seed
from peft import LoraConfig,get_peft_model,TaskType
from safetensors.torch import load_file,save_file
from common import *
sys.path.insert(0,str(HERE/'source'))
from affine_adapter import LowRankAffineMap
from budget import add_hidden_budget,hidden_digest

def adapter_state(model):
    return {n:p.detach().float().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
def digest(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):h.update(n.encode());h.update(v.float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build(spec):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_seed(spec['seed'])
    cfg=read(HERE/'models.json')[spec['model']]
    base=AutoModelForCausalLM.from_pretrained(cfg['path'],dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True)
    base.config.use_cache=False
    model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,
        lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'],bias='none'))
    budget={}
    if spec['arm']=='hidden_budget':budget=add_hidden_budget(model,8,spec['seed'])
    hooks=[];counts={'input':0,'output':0}
    if spec['arm'] in ['both','interior']:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec['seed']+1000003)
            maps=nn.ModuleDict({k:LowRankAffineMap(base.config.hidden_size,16,128,0,k=='input') for k in ['input','output']})
            with torch.no_grad():
                maps['output'].down.weight.copy_(maps['input'].down.weight)
                maps['output'].up.weight.copy_(maps['input'].up.weight)
        base.add_module('placement_maps',maps)
        def post(which):
            def hook(module,args,output):
                counts[which]+=1
                if isinstance(output,tuple):return (maps[which](output[0]),)+output[1:]
                return maps[which](output)
            return hook
        if spec['arm']=='both':
            hooks.append(base.get_input_embeddings().register_forward_hook(post('input')))
            def before_head(module,args):
                counts['output']+=1
                return (maps['output'](args[0]).to(module.weight.dtype),)+args[1:]
            hooks.append(base.lm_head.register_forward_pre_hook(before_head))
        else:
            depth=len(base.model.layers);positions=[depth//4-1,3*depth//4-1]
            for key,pos in zip(['input','output'],positions):
                hooks.append(base.model.layers[pos].register_forward_hook(post(key)))
    for p in model.parameters():
        if p.requires_grad:p.data=p.data.float()
    state=adapter_state(model)
    hidden=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n)
    affine=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'placement_maps.' in n)
    assert sum(v.numel() for v in state.values())==hidden+affine
    assert affine==(65*base.config.hidden_size if spec['arm'] in ['both','interior'] else 0)
    if budget:assert budget['actual_extra_parameters']==65*base.config.hidden_size
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    audit={'shared_hidden_init_sha256':hidden_digest(model,8),'all_adapter_init_sha256':digest(state),
        'affine_init_sha256':digest({n:v for n,v in state.items() if 'placement_maps.' in n}),
        'hidden_parameters':hidden,'affine_parameters':affine,'total_parameters':hidden+affine,'budget':budget,
        'positions':([len(base.model.layers)//4-1,3*len(base.model.layers)//4-1] if spec['arm']=='interior' else None)}
    # Keep handles and counters as ordinary metadata, outside the module tree.
    model._placement_handles=hooks;model._placement_counts=counts
    return model,audit

def batch(items,meta,train=False):
    seq=[x['input_ids']+([meta['label_ids'][x['gold']]] if train else []) for x in items]
    lengths=[len(s) for s in seq];width=max(lengths)
    ids=torch.tensor([s+[meta['pad_id']]*(width-len(s)) for s in seq],device='cuda')
    mask=torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seq],device='cuda')
    return ids,mask,torch.tensor(lengths,device='cuda')

def forward_selected(model,ids,mask,lengths,train=False):
    base=model.get_base_model()
    hidden=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    if train:
        pos=torch.stack([lengths-2,lengths-1],dim=1)
        selected=hidden[torch.arange(len(lengths),device=ids.device)[:,None],pos]
    else:selected=hidden[torch.arange(len(lengths),device=ids.device),lengths-1]
    return base.lm_head(selected).float()

def load_checkpoint(cp,fp32=True):
    spec=read(cp/'spec.json');model,audit=build(spec)
    state=load_file(str(cp/'adapter.safetensors'))
    expected=adapter_state(model);assert state.keys()==expected.keys()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad:
                assert state[n].shape==p.shape and torch.isfinite(state[n]).all()
                p.copy_(state[n])
    assert digest(adapter_state(model))==digest(state)
    model=model.cuda().eval()
    if fp32:model.float()
    return model,spec,audit
