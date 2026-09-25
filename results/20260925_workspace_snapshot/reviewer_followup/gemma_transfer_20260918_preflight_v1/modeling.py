import sys,hashlib
import torch
from transformers import AutoModelForCausalLM,set_seed
from peft import LoraConfig,get_peft_model,TaskType
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file
from common import *
sys.path.insert(0,str(HERE/'source'))
from affine_adapter import LowRankAffineMap
from budget import add_hidden_budget,hidden_digest

def is_adapter(n):return n.startswith('boundary_') or '.lora_A.' in n or '.lora_B.' in n
def adapter_state(model):return {n:p.detach().float().cpu().contiguous() for n,p in model.named_parameters() if is_adapter(n)}
def digest(state):
    h=hashlib.sha256()
    for n,p in sorted(state.items()):
        q=p.detach().cpu().contiguous();h.update(n.encode());h.update(str(q.dtype).encode());h.update(str(tuple(q.shape)).encode());h.update(q.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def frozen_digest(model):
    return digest({n.replace('.base_layer.','.'):p for n,p in model.named_parameters() if not is_adapter(n)})
def base_precision(model,dtype):
    for n,p in model.named_parameters():
        if not is_adapter(n):p.data=p.data.to(dtype)
def enabled(model,on):
    for m in model.modules():
        if isinstance(m,BaseTunerLayer):m.enable_adapters(on)
    for name in ['boundary_input','boundary_output']:
        if hasattr(model,name):
            m=getattr(model,name);m.scale=8. if on else 0.;m.bias_scale=1. if on else 0.

def install_boundaries(model,arm,seed):
    counts={'input':0,'output':0};hooks=[]
    for where,present in [('input',arm in ['input','both','hidden_input','hidden_both']),('output',arm in ['output','both','hidden_both','hidden_output'])]:
        if not present:continue
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+1000003)
            aff=LowRankAffineMap(model.config.hidden_size,16,128,0,where=='input')
        model.add_module('boundary_'+where,aff.float())
        if where=='input':
            def post(module,args,output,aff=aff):
                counts['input']+=1
                raw=torch.nn.functional.embedding(args[0],module.weight,module.padding_idx,module.max_norm,module.norm_type,module.scale_grad_by_freq,module.sparse)
                return aff(raw)*module.embed_scale.to(module.weight.dtype)
            hooks.append(model.get_input_embeddings().register_forward_hook(post))
        else:
            def pre(module,args,aff=aff):counts['output']+=1;return (aff(args[0]).to(module.weight.dtype),)+args[1:]
            hooks.append(model.lm_head.register_forward_pre_hook(pre))
    model._boundary_handles=hooks;model._boundary_counts=counts
    return counts,hooks

def build(spec):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_seed(spec['seed']);cfg=read(HERE/'models.json')[spec['model']]
    model=AutoModelForCausalLM.from_pretrained(cfg['path'],dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    assert model.config.bos_token_id==TOKEN_BOS_ID and model.config.eos_token_id==TOKEN_EOS_ID
    assert model.config.model_type=='gemma2' and model.config._attn_implementation=='eager'
    assert model.config.attn_logit_softcapping==50. and model.config.final_logit_softcapping==30.
    model.requires_grad_(False);model.config.use_cache=False;arm=spec['arm'];budget={}
    assert arm in ['base']+ARMS
    if arm.startswith('hidden'):
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.05,
            target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'],bias='none')).get_base_model()
        if arm=='hidden_budget':
            budget=add_hidden_budget(model,8,spec['seed'])
            assert budget['actual_extra_parameters']==budget['target_extra_parameters']==65*model.config.hidden_size
    counts,hooks=install_boundaries(model,arm,spec['seed'])
    for n,p in model.named_parameters():
        assert p.requires_grad==is_adapter(n),(arm,n)
        if p.requires_grad:p.data=p.data.float()
    state=adapter_state(model);trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    audit={'arm':arm,'trainable_names':list(trainable),'trainable_parameters':sum(p.numel() for p in trainable.values()),
        'parameter_groups':{k:sum(p.numel() for n,p in trainable.items() if (n.startswith('boundary_') if k=='boundary' else '.lora_' in n)) for k in ['boundary','hidden']},
        'initialization_sha256':digest(state),'shared_hidden_initialization_sha256':hidden_digest(model,8) if arm.startswith('hidden') else None,
        'budget_control':budget,'tied_weights':model.get_input_embeddings().weight.data_ptr()==model.lm_head.weight.data_ptr(),
        'attention_implementation':model.config._attn_implementation,'attention_softcap':model.config.attn_logit_softcapping,
        'final_softcap':model.config.final_logit_softcapping,'input_boundary_location':'raw word embedding, before native sqrt(hidden_size) scaling',
        'native_embedding_scale':model.get_input_embeddings().scalar_embed_scale,
        'native_embedding_scale_buffer':{'value':float(model.get_input_embeddings().embed_scale),'dtype':str(model.get_input_embeddings().embed_scale.dtype)},
        'base_precision':'official FP32 checkpoint rounded to BF16 for training; same rounded base cast to FP32 for evaluation in every arm, including Base'}
    assert audit['tied_weights']==cfg['tie_word_embeddings']
    assert audit['trainable_parameters']==read(HERE/'BUDGET_PLAN.json')[spec['model']][arm]
    audit['boundary_initialization_sha256']={side:digest({n:p for n,p in state.items() if n.startswith('boundary_'+side)}) for side in ['input','output'] if hasattr(model,'boundary_'+side)}
    d=model.config.hidden_size;expect=(33*d if arm in ['input','both','hidden_input','hidden_both'] else 0)+(32*d if arm in ['output','both','hidden_both','hidden_output'] else 0)
    assert audit['parameter_groups']['boundary']==expect
    assert bool(audit['parameter_groups']['hidden'])==arm.startswith('hidden')
    model._boundary_handles=hooks;model._boundary_counts=counts
    return model.cuda(),audit

def batch(items,training=True):
    sequences=[r['prompt_ids']+(r['target_ids'] if training else []) for r in items];width=max(map(len,sequences));ids=[];mask=[];labels=[]
    for r,s in zip(items,sequences):
        pad=width-len(s)
        ids.append(s+[TOKEN_PAD_ID]*pad if training else [TOKEN_PAD_ID]*pad+s)
        mask.append([1]*len(s)+[0]*pad if training else [0]*pad+[1]*len(s))
        if training:labels.append([-100]*len(r['prompt_ids'])+r['target_ids']+[-100]*pad)
    return (torch.tensor(ids,device='cuda'),torch.tensor(mask,device='cuda'),torch.tensor(labels,device='cuda') if training else None)

def projected_logits(model,hidden):
    logits=model.lm_head(hidden)
    cap=model.config.final_logit_softcapping
    return (torch.tanh(logits/cap)*cap).float() if cap is not None else logits.float()

def loss(model,ids,mask,labels):
    hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    keep=labels[:,1:]!=-100;selected=hidden[:,:-1][keep];gold=labels[:,1:][keep]
    logits=projected_logits(model,selected)
    ce=torch.nn.functional.cross_entropy(logits,gold,reduction='none')
    return ce.mean(),ce,gold

def load_adapter(model,path):
    state=load_file(str(path));assert state.keys()==adapter_state(model).keys()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if is_adapter(n):p.copy_(state[n])
    assert digest(adapter_state(model))==digest(state)
