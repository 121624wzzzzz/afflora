import sys,hashlib
import torch
from transformers import AutoConfig,Qwen3_5ForCausalLM,set_seed
from architecture import TARGETS,architecture_plan
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
# Keep native unfused RMSNormGated on CPU/GPU; fused recurrence is selected by dtype.
native.FusedRMSNormGated=None
from peft import LoraConfig,get_peft_model,TaskType
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file
from safetensors import safe_open
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
def kernel_mode(model,fast):
    for m in model.modules():
        if isinstance(m,native.Qwen3_5GatedDeltaNet):
            m.causal_conv1d_fn=native.causal_conv1d_fn if fast else None
            m.causal_conv1d_update=native.causal_conv1d_update if fast else native.torch_causal_conv1d_update
            m.chunk_gated_delta_rule=native.chunk_gated_delta_rule if fast else native.torch_chunk_gated_delta_rule
            m.recurrent_gated_delta_rule=native.fused_recurrent_gated_delta_rule if fast else native.torch_recurrent_gated_delta_rule

def base_precision(model,dtype):
    kernel_mode(model,dtype==torch.bfloat16)
    for n,p in model.named_parameters():
        if not is_adapter(n):p.data=p.data.to(torch.float32 if n.replace('.base_layer.','.') in model._native_fp32_names else dtype)
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
                return aff(output).to(module.weight.dtype)
            hooks.append(model.get_input_embeddings().register_forward_hook(post))
        else:
            def pre(module,args,aff=aff):counts['output']+=1;return (aff(args[0]).to(module.weight.dtype),)+args[1:]
            hooks.append(model.lm_head.register_forward_pre_hook(pre))
    model._boundary_handles=hooks;model._boundary_counts=counts
    return counts,hooks

def build(spec):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_seed(spec['seed']);cfg=read(HERE/'models.json')[spec['model']]
    text_config=AutoConfig.from_pretrained(cfg['path'],local_files_only=True).text_config
    model,loading=Qwen3_5ForCausalLM.from_pretrained(cfg['path'],config=text_config,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,output_loading_info=True)
    loading={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}
    assert not loading.get('missing_keys') and not loading.get('mismatched_keys') and not loading.get('error_msgs'),loading
    assert all(n.startswith(('model.visual.','mtp.')) for n in loading.get('unexpected_keys',[])),loading
    # Preserve the original FP32 decay and gated-norm tensors; from_pretrained(dtype=BF16)
    # otherwise rounds them because this native class has no keep-in-FP32 declaration.
    weight_map=read(Path(cfg['path'])/'model.safetensors.index.json')['weight_map'];native_fp32=set()
    parameters=dict(model.named_parameters())
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(Path(cfg['path'])/shard),framework='pt',device='cpu') as f:
            for key in f.keys():
                if not key.startswith('model.language_model.'):continue
                if f.get_slice(key).get_dtype()!='F32':continue
                name=key.replace('model.language_model.','model.',1)
                assert name.endswith(('.linear_attn.A_log','.linear_attn.norm.weight')),name
                parameters[name].data=f.get_tensor(key).clone();native_fp32.add(name)
    assert len(native_fp32)==48
    model._native_fp32_names=native_fp32
    assert model.config.bos_token_id==TOKEN_BOS_ID and model.config.eos_token_id==TOKEN_EOS_ID
    assert model.config.model_type=='qwen3_5_text' and model.config._attn_implementation=='eager'
    from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
    assert is_fast_path_available
    plan=architecture_plan(model.config)
    actual={n:[m.in_features,m.out_features] for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n!='lm_head'}
    assert actual==plan['modules'],(set(actual)^set(plan['modules']))
    model.requires_grad_(False);model.config.use_cache=False;arm=spec['arm'];budget={}
    assert arm in ['base']+ARMS
    if arm.startswith('hidden'):
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.05,
            target_modules=TARGETS,bias='none')).get_base_model()
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
        'attention_implementation':model.config._attn_implementation,
        'input_boundary_location':'native raw word embedding; Qwen3.5 has no embedding scale',
        'native_kernel':'Native FLA/causal-conv training; native PyTorch FP32 evaluation; native unfused gated RMSNorm',
        'text_projection_modules':actual,'loading_info':loading,'native_fp32_tensor_names':sorted(native_fp32),
        'gradient_checkpointing':'non-reentrant, preserve_rng_state=True',
        'base_precision':'Official mixed checkpoint retained exactly: BF16 matrices and 48 original FP32 decay/norm tensors; FP32 evaluation in every arm including Base'}
    assert audit['tied_weights']==cfg['tie_word_embeddings']
    assert audit['trainable_parameters']==read(HERE/'BUDGET_PLAN.json')[spec['model']][arm]
    audit['boundary_initialization_sha256']={side:digest({n:p for n,p in state.items() if n.startswith('boundary_'+side)}) for side in ['input','output'] if hasattr(model,'boundary_'+side)}
    d=model.config.hidden_size;expect=(33*d if arm in ['input','both','hidden_input','hidden_both'] else 0)+(32*d if arm in ['output','both','hidden_both','hidden_output'] else 0)
    assert audit['parameter_groups']['boundary']==expect
    assert bool(audit['parameter_groups']['hidden'])==arm.startswith('hidden')
    model._boundary_handles=hooks;model._boundary_counts=counts
    if arm!='base':model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    kernel_mode(model,True)
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
    return logits.float()

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
