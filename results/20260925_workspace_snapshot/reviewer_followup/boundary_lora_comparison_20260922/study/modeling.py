import sys,hashlib
import torch
from transformers import AutoModelForCausalLM,set_seed
from peft import LoraConfig,get_peft_model,TaskType
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file
from common import *
sys.path.insert(0,str(HERE/'source'))
from affine_adapter import LowRankAffineMap
from shared_boundary import SharedTransposeHead
from vocab_boundary import VocabLoRA,VocabHead
PAD_ID=151643
from budget import add_hidden_budget,hidden_digest

def is_adapter(n):return n.startswith('boundary_') or '.lora_A.' in n or '.lora_B.' in n
def adapter_state(model):return {n:p.detach().float().cpu().contiguous() for n,p in model.named_parameters() if is_adapter(n)}
def digest(state):
    h=hashlib.sha256()
    for n,p in sorted(state.items()):
        q=p.detach().cpu().contiguous();h.update(n.encode());h.update(str(q.dtype).encode());h.update(str(tuple(q.shape)).encode());h.update(q.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def frozen_digest(model):
    return digest({n.replace('.base_layer.','.').replace('lm_head.base_head.','lm_head.'):p for n,p in model.named_parameters() if not is_adapter(n)})
def base_precision(model,dtype):
    for n,p in model.named_parameters():
        if not is_adapter(n):p.data=p.data.to(dtype)
def enabled(model,on):
    for m in model.modules():
        if isinstance(m,BaseTunerLayer):m.enable_adapters(on)
    for name in ['boundary_input','boundary_output']:
        if hasattr(model,name):
            m=getattr(model,name);m.scale=m.active_scale if on else 0.;m.bias_scale=1. if on else 0.

def build(spec):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_seed(spec['seed']);cfg=read(HERE/'models.json')[spec['model']]
    model=AutoModelForCausalLM.from_pretrained(cfg['path'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    global PAD_ID
    PAD_ID=model.config.eos_token_id
    assert PAD_ID==(128001 if spec['model'].startswith('llama') else 151643)
    model.requires_grad_(False);model.config.use_cache=False;arm=spec['arm'];budget={}
    assert arm in ['base','hidden']
    if arm.startswith('hidden'):
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.05,
            target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'],bias='none')).get_base_model()
        if arm=='hidden_budget':budget=add_hidden_budget(model,8,spec['seed'])
    counts={'input':0,'output':0};hooks=[]
    rank=spec.get('rank',16);method=spec.get('method','none');placement=spec.get('placement','none');shared=placement=='shared'
    if shared:assert cfg['tie_word_embeddings'], 'Never share adapters on untied bases'
    for where in ['input','output']:
        present=placement in (["e","eu","shared"] if where=='input' else ["u","eu"])
        if not present:continue
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec['seed']+1000003)
            aff=(LowRankAffineMap(model.config.hidden_size,rank,8*rank,0,spec.get('bias',True) and where=='input') if method=='affine' else VocabLoRA(model.config.vocab_size,model.config.hidden_size,rank,where))
        aff.active_scale=aff.scale
        model.add_module('boundary_'+where,aff.float())
        if where=='input':
            def post(module,args,output,aff=aff):
                counts['input']+=1
                return aff(output) if method=='affine' else output+aff.embedding(args[0]).to(output.dtype)
            hooks.append(model.get_input_embeddings().register_forward_hook(post))
        elif method=='affine':
            def pre(module,args,aff=aff):counts['output']+=1;return (aff(args[0]).to(module.weight.dtype),)+args[1:]
            hooks.append(model.lm_head.register_forward_pre_hook(pre))
        else:model.lm_head=VocabHead(model.lm_head,aff,counts)
    if shared:
        model.lm_head=(SharedTransposeHead(model.lm_head,model.boundary_input,counts) if method=='affine' else VocabHead(model.lm_head,model.boundary_input,counts))
    for n,p in model.named_parameters():
        assert p.requires_grad==is_adapter(n),(arm,n)
        if p.requires_grad:p.data=p.data.float()
    state=adapter_state(model);trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    audit={'arm':arm,'trainable_names':list(trainable),'trainable_parameters':sum(p.numel() for p in trainable.values()),
        'parameter_groups':{k:sum(p.numel() for n,p in trainable.items() if (n.startswith('boundary_') if k=='boundary' else '.lora_' in n)) for k in ['boundary','hidden']},
        'initialization_sha256':digest(state),'shared_hidden_initialization_sha256':hidden_digest(model,8) if arm.startswith('hidden') else None,
        'budget_control':budget,'tied_weights':model.get_input_embeddings().weight.data_ptr()==model.lm_head.weight.data_ptr()}
    assert audit['tied_weights']==cfg['tie_word_embeddings']
    audit['boundary_initialization_sha256']={side:digest({n:p for n,p in state.items() if n.startswith('boundary_'+side)}) for side in ['input','output'] if hasattr(model,'boundary_'+side)}
    d=model.config.hidden_size;v=model.config.vocab_size
    if method=='none':expect=0
    elif method=='vocab':expect=rank*(v+d)*(2 if placement=='eu' else 1)
    else:expect=(2*rank*d*(2 if placement=='eu' else 1))+(d if spec.get('bias',True) and placement in ['e','eu','shared'] else 0)
    audit['boundary_sharing']={'shared':shared,'rank':rank,'output_transpose':shared,'unique_parameter_ids':len({id(p) for p in trainable.values()}),'active_sides':['input','output'] if shared else list(audit['boundary_initialization_sha256'])}
    audit['method']=method;audit['placement']=placement;audit['expected_boundary_parameters']=expect
    assert audit['boundary_sharing']['unique_parameter_ids']==len(trainable)
    assert audit['parameter_groups']['boundary']==expect
    assert bool(audit['parameter_groups']['hidden'])==arm.startswith('hidden')
    if placement=='eu':
        left={p.data_ptr() for p in model.boundary_input.parameters()};right={p.data_ptr() for p in model.boundary_output.parameters()}
        assert left.isdisjoint(right);audit['independent_boundary_storage_verified']=True
    model._boundary_handles=hooks;model._boundary_counts=counts
    return model.cuda(),audit

def batch(items,training=True):
    sequences=[r['prompt_ids']+(r['target_ids'] if training else []) for r in items];width=max(map(len,sequences));ids=[];mask=[];labels=[]
    for r,s in zip(items,sequences):
        pad=width-len(s)
        ids.append(s+[PAD_ID]*pad if training else [PAD_ID]*pad+s)
        mask.append([1]*len(s)+[0]*pad if training else [0]*pad+[1]*len(s))
        if training:labels.append([-100]*len(r['prompt_ids'])+r['target_ids']+[-100]*pad)
    return (torch.tensor(ids,device='cuda'),torch.tensor(mask,device='cuda'),torch.tensor(labels,device='cuda') if training else None)

def loss(model,ids,mask,labels):
    hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    keep=labels[:,1:]!=-100;selected=hidden[:,:-1][keep];gold=labels[:,1:][keep]
    logits=model.lm_head(selected).float()
    ce=torch.nn.functional.cross_entropy(logits,gold,reduction='none')
    return ce.mean(),ce,gold

def load_adapter(model,path):
    state=load_file(str(path));assert state.keys()==adapter_state(model).keys()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if is_adapter(n):p.copy_(state[n])
    assert digest(adapter_state(model))==digest(state)
