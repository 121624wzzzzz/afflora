"""CPU-only sampled matrix identity audit using real frozen weights and learned branches.

This is a post-hoc implementation diagnostic, not an end-to-end merged-model evaluation.
Only fixed CLUENER seed 6100 checkpoints that already passed scoring audit are examined.
"""
from reporting_common import *
import torch
import torch.nn.functional as F
from safetensors import safe_open

def matrix_rows(path,key,indices,tied=False):
 index=path/'model.safetensors.index.json'
 if index.exists():
  mapping=read(index)['weight_map']
  if key not in mapping:
   assert tied and key=='lm_head.weight';key='model.embed_tokens.weight'
  shard=path/mapping[key]
 else:shard=path/'model.safetensors'
 with safe_open(str(shard),framework='pt',device='cpu') as f:
  if key not in f.keys():
   assert tied and key=='lm_head.weight';key='model.embed_tokens.weight'
  # Slice first, so no full vocabulary matrix is materialized.
  s=f.get_slice(key);return torch.cat([s[i:i+1] for i in indices],dim=0).double()

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 models=read(ROOT/'core/models.json');data=collect('core');records=[];missing=[]
 arms=['input','output','both','hidden_input','hidden_output','hidden_both']
 for m in MODELS:
  cfg=models[m];path=Path(cfg['path']);mc=read(path/'config.json');d=mc['hidden_size'];vocab=mc['vocab_size']
  ids=sorted(set(torch.linspace(0,vocab-1,127).long().tolist()+[151643]));tied=cfg['tie_word_embeddings']
  we=matrix_rows(path,'model.embed_tokens.weight',ids,tied);wu=matrix_rows(path,'lm_head.weight',ids,tied)
  assert (torch.equal(we,wu) if tied else True)
  gen=torch.Generator().manual_seed(20260917);h=torch.randn(8,d,generator=gen,dtype=torch.float64)
  for arm in arms:
   key=('cluener',m,arm,6100)
   if key not in data:missing.append({'model':m,'arm':arm});continue
   row=data[key]
   if row['reused']:
    anchors=read(ROOT/'core/ANCHOR_INDEX.json');a=next(a for a in anchors if a['task']=='cluener' and a['model']==m and a['arm']==arm and a['seed']==6100)
    adapter=Path(a['source_checkpoint'])/'adapter.safetensors'
   else:adapter=ROOT/'core/checkpoints'/row['name']/'adapter.safetensors'
   with safe_open(str(adapter),framework='pt',device='cpu') as f:
    states={n:f.get_tensor(n).double() for n in f.keys() if n.startswith('boundary_')}
   result={'model':m,'arm':arm,'seed':6100,'original_tied':tied,'sampled_vocab_rows':len(ids),
           'hidden_vectors':len(h),'adapter_sha256':sha(adapter),'branch_checks':{}}
   merged={'input':we,'output':wu}
   for side,w in [('input',we),('output',wu)]:
    prefix='boundary_'+side+'.'
    if prefix+'down.weight' not in states:continue
    a=states[prefix+'down.weight'];b=states[prefix+'up.weight'];bias=states.get(prefix+'bias')
    assert a.shape==(16,d) and b.shape==(d,16)
    transform=torch.eye(d,dtype=torch.float64)+8*(b@a)
    if side=='input':
     runtime=w+8*F.linear(F.linear(w,a),b)+bias
     merged[side]=w@transform.T+bias;reference=merged[side]
     rf=w.float()+8*F.linear(F.linear(w.float(),a.float()),b.float())+bias.float()
     mf=w.float()@transform.float().T+bias.float()
    else:
     assert bias is None
     runtime=F.linear(h+8*F.linear(F.linear(h,a),b),w)
     merged[side]=w@transform;reference=F.linear(h,merged[side])
     rf=F.linear(h.float()+8*F.linear(F.linear(h.float(),a.float()),b.float()),w.float())
     mf=F.linear(h.float(),w.float()@transform.float())
    error=float((runtime-reference).abs().max());assert error<1e-10,(m,arm,side,error)
    result['branch_checks'][side]={'float64_max_abs_error':error,'float32_max_abs_error':float((rf-mf).abs().max()),
                                  'sampled_effective_weight_delta_frobenius':float((merged[side]-w).norm())}
   assert result['branch_checks']
   if tied:
    gap=merged['input']-merged['output'];centered=gap-gap.mean(dim=0,keepdim=True)
    result['independent_merged_sides_sampled_relative_gap']=float(gap.norm()/we.norm())
    # A common vocabulary-row shift only adds a common scalar to logits.
    # Remove that softmax-invariant component before interpreting incompatibility.
    result['sampled_relative_gap_after_removing_common_logit_shift']=float(centered.norm()/we.norm())
    result['sampled_relative_logit_max_abs_gap']=float(F.linear(h,centered).abs().max())
   records.append(result)
   print(m,arm,'matrix identity passed',flush=True)
 write(ROOT/'BOUNDARY_ALGEBRA_AUDIT.json',{'at':now(),'status':'passed','scope':'post-hoc sampled real-weight CPU matrix identities; fixed CLUENER seed 6100; not full-vocabulary or end-to-end merged-model evaluation',
       'passed_runs':len(records),'missing_fixed_cells':missing,'records':records})
 print('Audited',len(records),'fixed available cells; missing',len(missing))

if __name__=='__main__':main()
