"""Prepare new, verified copies; never mutate the sealed 4B study."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
ARCHITECTURE='''"""Architecture-derived exact budget, fixed without observing task outcomes."""
import itertools
TARGETS=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj',
         'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']
def architecture_plan(cfg):
 d=cfg.hidden_size; f=cfg.intermediate_size;q=cfg.num_attention_heads*cfg.head_dim;k=cfg.num_key_value_heads*cfg.head_dim
 lk=cfg.linear_num_key_heads*cfg.linear_key_head_dim;lv=cfg.linear_num_value_heads*cfg.linear_value_head_dim
 modules={};groups={}
 for i,kind in enumerate(cfg.layer_types):
  stem=f'model.layers.{i}'
  for n,ins,outs in [('gate_proj',d,f),('up_proj',d,f),('down_proj',f,d)]:modules[f'{stem}.mlp.{n}']=[ins,outs]
  if kind=='full_attention':
   block='self_attn';sizes={'q_proj':(d,2*q),'k_proj':(d,k),'v_proj':(d,k),'o_proj':(q,d)}
   grouped={'query':'q_proj','key':'k_proj','output':'o_proj'}
  else:
   assert kind=='linear_attention';block='linear_attn'
   sizes={'in_proj_qkv':(d,2*lk+lv),'in_proj_z':(d,lv),'in_proj_b':(d,cfg.linear_num_value_heads),'in_proj_a':(d,cfg.linear_num_value_heads),'out_proj':(lv,d)}
   grouped={'query':'in_proj_qkv','output':'out_proj'}
  for n,shape in sizes.items():modules[f'{stem}.{block}.{n}']=list(shape)
  for group,n in grouped.items():
   name=f'{stem}.{block}.{n}';groups.setdefault((group,sum(modules[name])),[]).append(name)
 keys=sorted(groups,key=lambda x:(['query','key','output'].index(x[0]),x[1]));extra=65*d;choices=[]
 for counts in itertools.product(*(range(len(groups[k])+1) for k in keys)):
  if sum(n*k[1] for n,k in zip(counts,keys))!=extra:continue
  nq=sum(n for n,k in zip(counts,keys) if k[0]=='query');nk=sum(n for n,k in zip(counts,keys) if k[0]=='key');no=sum(n for n,k in zip(counts,keys) if k[0]=='output')
  choices.append(((-nq,-nk,no,tuple(-n for n in counts)),counts))
 assert choices,('No exact active-rank budget',extra,keys)
 counts=min(choices)[1];chosen=[]
 for k,count in zip(keys,counts):
  names=groups[k];chosen += [names[i*len(names)//count] for i in range(count)]
 assert len(set(chosen))==len(chosen) and sum(sum(modules[n]) for n in chosen)==extra
 return {'hidden_size':d,'hidden_rank':8,'boundary_rank':16,'input_bias':True,'output_bias':False,'extra':extra,
  'hidden_parameters':8*sum(sum(v) for v in modules.values()),'strict_match':True,'modules':modules,
  'rank_pattern':{n:9 for n in chosen},'group_costs':{f'{g}:{c}':c for g,c in keys},
  'extra_rank_counts':{f'{g}:{c}':n for (g,c),n in zip(keys,counts)},
  'rule':'Exact budget, at most one added rank/module. Maximize query count, then key count, minimize output count; ties maximize counts in query/key/output then ascending-cost order. Even spacing by layer within each semantic/cost group.',
  'scope':'All 12 named dense projections in text decoder; no vision, lm_head, convolution, norm or recurrent scalar adaptation.'}
'''

def edit(p,old,new):
 s=p.read_text();assert old in s,(p,old);p.write_text(s.replace(old,new))

def main():
 for key,size in [('08b','0.8B'),('2b','2B'),('9b','9B')]:
  p=ROOT/f'qwen35_{key}_20260919';assert not (p/'CODE_FROZEN.json').exists()
  edit(p/'common.py',"MODEL='qwen35_4b_base'",f"MODEL='qwen35_{key}_base'")
  (p/'architecture.py').write_text(ARCHITECTURE)
  edit(p/'prepare.py','Qwen3.5-4B-Base',f'Qwen3.5-{size}-Base')
  edit(p/'prepare.py','qwen35_4b_official',f'qwen35_{key}_official')
  edit(p/'prepare.py',' and cfg.tie_word_embeddings'," and cfg.tie_word_embeddings==read(HERE/'models.json')[MODEL]['tie_word_embeddings']")
  edit(p/'modeling.py','assert len(native_fp32)==48',"assert len(native_fp32)==2*text_config.layer_types.count('linear_attention')")
  edit(p/'modeling.py','BF16 matrices and 48 original FP32 decay/norm tensors','BF16 matrices and all original FP32 decay/norm tensors')
  edit(p/'audit_one.py',"assert len(init['native_fp32_tensor_names'])==48", "assert len(init['native_fp32_tensor_names'])==2*read(HERE/'MODEL_IDENTITY_AUDIT.json')['linear_attention_layers']")
  edit(p/'analyze.py','bonferroni4','bonferroni12')
  edit(p/'analyze.py','1-.05/8','1-.05/24')
  edit(p/'analyze.py','[.625,99.375]','[100*.05/24,100*(1-.05/24)]')
  edit(p/'analyze.py',"'family':4","'family':12")
  edit(p/'analyze.py','Family-4','Family-12')
  edit(p/'analyze.py','Qwen3.5-4B',f'Qwen3.5-{size}')
  edit(p/'analyze.py','tied frozen weights with independent effective E/U updates','native tied/untied frozen weights with independent effective E/U updates')
  # Store a dynamic identity count in the downloader, before writing its audits.
  d=ROOT/f'download_qwen35_{key}_verified_20260919.py'
  edit(d,"'hidden_size':cfg['hidden_size']", "'linear_attention_layers':cfg['layer_types'].count('linear_attention'),'hidden_size':cfg['hidden_size']")
  technical={
   'status':'documented_before_fits','source_study':'qwen35_transfer_20260919',
   'changes':['Generalized exact rank budget to unequal hybrid projection widths.',
    'Check native tied/untied status and original FP32 tensor count per actual architecture.',
    'Three new sizes form one family of 12 primary contrasts; 4B remains a historical family-4 result.',
    'Add a repeated initial forward/backward probe before formal admission; freeze final numerical policy only after these technical probes.'],
   'preserved':['2048 training examples, 64 steps, microbatch 2, optimizer and native training/evaluation precision',
    'Same verified prompts/data/metrics, LR grid, 2 dev and 5 confirmation seeds',
    'All native text projections and boundary parameterization'],
   'known_limitation':'Sealed 4B initial BF16 first-batch losses were not universally identical across jobs despite verified paired initialization. No claim that this explains all effects or is resolved.'}
  (p/'TECHNICAL_CORRECTIONS.json').write_text(json.dumps(technical,indent=2)+'\n')
  protocol=f'''# Qwen3.5-{size}-Base: prespecified multiscale extension

This is one of three new sizes (0.8B, 2B, 9B) requested after the sealed 4B
results were known. All three sizes are retained, irrespective of outcomes.
The 4B historical study is not a new independent replication and is not pooled
into the new inferential family. No trained weights/predictions are reused.

Official pretrained-only Qwen/Qwen3.5-{size}-Base is downloaded from ModelScope.
MODEL_IDENTITY_AUDIT.json pins the official HF revision and independently verified
git/LFS identities. Use native text decoder, tokenizer and actual tied/untied
status; no vision adaptation. Inspect every loaded base tensor against shards.
All original FP32 recurrent decay/norm weights are retained; other matrices BF16.
The isolated Torch 2.6/Transformers 5.6.2 runtime is unchanged. Native FLA and
causal convolution training, native PyTorch FP32 evaluation, unfused gated norm,
eager full attention, TF32 off, non-reentrant RNG-preserving checkpointing.

Verified copies of the sealed 4B source/data/scorers are documented in SOURCE_REUSE.
WikiSQL train/dev/confirm=2048/1024/2048, TREC50=2048/256/500. Retokenize all inputs,
verify groups, labels and reference scoring. Same prompts and no truncation.
These public sets are previously evaluated project benchmarks, not globally
unseen data and not evidence against pretraining exposure.
WikiSQL primary=official execution accuracy, greedy generation cap 256/native EOS.
TREC50 primary=joint two-token code likelihood among 50 candidates, no EOS,
no length normalization, five complete native prefixes, no scoring cache.
Native 50-candidate checks on shortest/longest examples of every evaluation,
tolerance 2e-4, identical argmax. Save and audit all responses/scores.

H: rank 8, alpha 16, dropout .05 on all dense decoder projections. HEU: H plus
rank-16 alpha-128 dropout-0 independent E/U maps, E biased/U unbiased. Budget H
adds exactly 65*d active trainable parameters by the rule in architecture.py:
at most one added rank per query/key/output projection, exact sum, maximize query
count then key count, minimize output count; tie by semantic group then ascending
per-rank cost; evenly spaced layers within each semantic/cost group. Preserve
shared rank-8 initialization and alpha/r=2. Verify every extra rank updates.
This is a configured-method comparison, not pure placement causality.

Every fit: 2048 examples once, 64 optimizer steps, effective batch 32/microbatch 2,
AdamW betas(.9,.999), eps1e-8, weight decay0, 2 warmup steps then cosine,
joint global clip1. True token-mean loss including EOS, masked prompts. Final
checkpoint only. Save initialization/order, gradients and update norms. Assert
frozen base, optimizer whitelist, finite FP32 adapters/moments and exact reload.

Same LR candidate and tie order in every arm/task/size:
2e-4, 1e-4, 4e-4, 5e-5, 3e-4, 8e-4. Boundary LR equals H LR.
Development seeds 7600/7601: 72 fits per size. Freeze choices only after all dev
audits pass. Then confirmation seeds 7700..7704: 30 fits, plus 2 frozen Base
evaluations. Ten technical smokes cover all arms and low/high boundary LRs.
114 jobs per size, 342 new jobs total. Equal tuning counts do not imply equal FLOPs.

Twelve primary contrasts across the THREE new sizes: HEU-H and HEU-budget H on
each of two tasks. Report all seed differences, mean, sample SD, marginal 95%
and family-12 Bonferroni t intervals with df4. 20,000 paired cluster bootstraps
(WikiSQL table/TREC question; random seeds20300920/21), conditional on the fixed
five fitted models, with marginal and family-12 percentiles. This does not
combine seed/data/tuning uncertainty. Four-B results remain historical family4.
Small samples, fixed data/configuration and t assumptions limit inference.
No score-based retries, selective reporting or universal positivity claims.

Before freezing: formula, actual checkpoint, hybrid kernel, memory and repeated
initial forward/backward technical probes. Known 4B BF16 first-loss variation
must be reported; passing a local repeat does not prove bitwise reproducibility
across kernels/devices. Numerical failures halt admission for diagnosis.
One global scheduler admits at most one study worker per eligible GPU with at
least 68GiB free. Never terminate other workloads. Failed artifacts retained;
technical changes require a recorded fresh freeze before any affected fit.
After completion: independent output and paired-initialization audits, report,
seal all artifacts and independently verify hashes and recomputed contrasts.
'''
  (p/'PROTOCOL.md').write_text(protocol)
  print('configured',key,flush=True)

if __name__=='__main__':main()
