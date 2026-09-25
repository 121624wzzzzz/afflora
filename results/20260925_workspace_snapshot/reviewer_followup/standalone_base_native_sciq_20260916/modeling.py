import sys,hashlib
import torch
from transformers import AutoModelForCausalLM,set_seed
from safetensors.torch import load_file
from peft import LoraConfig,get_peft_model,TaskType
from peft.tuners.tuners_utils import BaseTunerLayer
from settings import *
sys.path.insert(0,str(HERE/'source'))
from affine_adapter import LowRankAffineMap

PREFIX='standalone_affine.'
def adapter_state(model):return {n:p.detach().float().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
def digest(state):
    h=hashlib.sha256()
    for n,p in sorted(state.items()):
        q=p.detach().cpu().contiguous()
        h.update(n.encode());h.update(str(q.dtype).encode());h.update(str(tuple(q.shape)).encode());h.update(q.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def frozen_digest(model):
    h=hashlib.sha256()
    original={n.replace('.base_layer.','.'):p for n,p in model.named_parameters() if not is_adapter_name(n)}
    for n,p in sorted(original.items()):
        assert not p.requires_grad,n
        q=p.detach().cpu().contiguous()
        h.update(n.encode());h.update(str(q.dtype).encode());h.update(str(tuple(q.shape)).encode());h.update(q.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def is_adapter_name(n):
    return n.startswith(PREFIX) or '.lora_A.' in n or '.lora_B.' in n
def set_adapter_enabled(model,enabled):
    if hasattr(model,'standalone_affine'):
        model.standalone_affine.scale=8. if enabled else 0.
        model.standalone_affine.bias_scale=1. if enabled else 0.
    else:
        for module in model.modules():
            if isinstance(module,BaseTunerLayer):module.enable_adapters(enabled)
def build(spec):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_seed(spec['seed']);cfg=read(HERE/'models.json')[spec['model']]
    model=AutoModelForCausalLM.from_pretrained(cfg['path'],dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    model.requires_grad_(False);model.config.use_cache=False
    hooks=[];counts={'input':0,'output':0};arm=spec['arm']
    if arm=='hidden_r8':
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,
            lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'],bias='none')).get_base_model()
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
    elif arm!='base':
        assert arm in ARMS
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec['seed']+1000003)
            aff=LowRankAffineMap(model.config.hidden_size,16,128,0,arm=='input')
        model.add_module('standalone_affine',aff.float())
        if arm=='input':
            def post(module,args,output):counts['input']+=1;return aff(output)
            hooks.append(model.get_input_embeddings().register_forward_hook(post))
        else:
            def pre(module,args):counts['output']+=1;return (aff(args[0]).to(module.weight.dtype),)+args[1:]
            hooks.append(model.lm_head.register_forward_pre_hook(pre))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert all(is_adapter_name(n) for n in trainable)
    assert all(('.lora_A.' in n or '.lora_B.' in n) if arm=='hidden_r8' else n.startswith(PREFIX) for n in trainable)
    assert any('lora_' in n for n,p in model.named_parameters())==(arm=='hidden_r8')
    count=sum(p.numel() for p in trainable.values());d=model.config.hidden_size
    if arm=='hidden_r8':
        expected=8*sum(module.in_features+module.out_features for module in model.modules() if isinstance(module,BaseTunerLayer))
    else:expected=0 if arm=='base' else 32*d+(d if arm=='input' else 0)
    assert count==expected,(count,expected)
    assert all(p.dtype==torch.float32 for p in trainable.values())
    audit={'arm':arm,'trainable_parameters':count,'trainable_names':list(trainable),'base_parameter_count':sum(p.numel() for n,p in model.named_parameters() if not is_adapter_name(n)),
        'hidden_lora_installed':arm=='hidden_r8','adapter_init_sha256':digest(adapter_state(model)),
        'linear_init_sha256':digest({n:p for n,p in adapter_state(model).items() if n.endswith('weight')}),'tie_word_embeddings':model.config.tie_word_embeddings}
    model._boundary_handles=hooks;model._boundary_counts=counts
    return model,audit
def batch(items,meta,train=False,left=False):
    seq=[x['input_ids']+([meta['label_ids'][x['gold']]] if train else []) for x in items]
    lengths=[len(s) for s in seq];width=max(lengths)
    ids=[([meta['pad_id']]*(width-len(s))+s if left else s+[meta['pad_id']]*(width-len(s))) for s in seq]
    masks=[([0]*(width-len(s))+[1]*len(s) if left else [1]*len(s)+[0]*(width-len(s))) for s in seq]
    return torch.tensor(ids,device='cuda'),torch.tensor(masks,device='cuda'),torch.tensor(lengths,device='cuda')
def forward_selected(model,ids,mask,lengths,train=False):
    h=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    if train:
        pos=torch.stack([lengths-2,lengths-1],dim=1);selected=h[torch.arange(len(lengths),device=ids.device)[:,None],pos]
    else:selected=h[torch.arange(len(lengths),device=ids.device),lengths-1]
    return model.lm_head(selected).float()
def load_checkpoint(cp):
    spec=read(cp/'spec.json');model,audit=build(spec)
    if spec['arm']!='base':
        state=load_file(str(cp/'adapter.safetensors'));expected=adapter_state(model);assert state.keys()==expected.keys()
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.requires_grad:assert p.shape==state[n].shape;p.copy_(state[n])
        assert digest(adapter_state(model))==digest(state)
    return model.cuda().eval().float(),spec,audit
