"""Controlled two-kernel intervention, preserving all weights and benchmark settings."""
import sys,importlib,itertools,time,json
from pathlib import Path
import torch
from probe_qwen35_kernel_configs_20260920 import config,unwrap,caches,PARENT

def main():
 root=PARENT/'qwen35_08b_20260919';out=PARENT/'qwen35_numerics_20260920/kernel_pair';out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(root))
 from common import read,write,sha,MODEL,now
 from modeling import build,batch,loss,adapter_state,digest,frozen_digest
 from run import preflight
 from initial_repeat import initial_repeat
 source=root/'checkpoints/search_wikisql_hidden_budget_c1_s7600';ref=root/'checkpoints/search_wikisql_hidden_budget_c0_s7600'
 spec=read(source/'spec.json');init=read(source/'INITIALIZATION.json');data=read(root/f'tokens/wikisql_{MODEL}_train.json')
 model,audit=build(spec);assert digest(adapter_state(model))==init['initialization_sha256'];before=frozen_digest(model);assert before==init['frozen_before']
 pf=preflight(model,spec,data);local=initial_repeat(model,data,spec['seed'],2);assert local['status']=='passed';baseline=caches()
 ko=unwrap(importlib.import_module('fla.ops.common.chunk_o').chunk_fwd_kernel_o)
 kt=unwrap(importlib.import_module('fla.ops.utils.solve_tril').merge_16x16_to_64x64_inverse_kernel)
 os=list(ko.configs);ts=[c for c in kt.configs if c.num_stages==3 and c.kwargs=={'DOT_PRECISION':'ieee'}];assert len(os)==len(ts)==3
 order=torch.randperm(len(data),generator=torch.Generator().manual_seed(spec['seed'])).tolist()[:32];denom=sum(len(data[i]['target_ids']) for i in order)
 target=read(source/'INITIAL_REPEAT.json')['microbatch_losses'][0];reference=read(ref/'INITIAL_REPEAT.json')['microbatch_losses'][0]
 measurements=[]
 for co,ct in itertools.product(os,ts):
  ko.configs=[co];ko.cache.clear();kt.configs=[ct];kt.cache.clear();values=[];started=time.monotonic()
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   model.train();model.zero_grad(set_to_none=True)
   for begin in range(0,32,2):
    ids,mask,labels=batch([data[i] for i in order[begin:begin+2]])
    with torch.autocast('cuda',dtype=torch.bfloat16):_,ce,_=loss(model,ids,mask,labels);value=ce.sum()/denom
    assert torch.isfinite(value);values.append(float(value.detach()))
   del value,ce
  row=dict(o_config=config(co),triangular_config=config(ct),loss=sum(values),microbatch_losses=values,
   matches_recorded_exception_exactly=values==target,matches_reference_exactly=values==reference,seconds=time.monotonic()-started)
  measurements.append(row);write(out/'PROGRESS.json',dict(measurements=measurements));print(json.dumps(row),flush=True)
 assert digest(adapter_state(model))==init['initialization_sha256'] and frozen_digest(model)==before
 result=dict(at=now(),status='passed',scope='Zero updates; only two process-local official Triton configurations vary.',
  source_files={str(p):sha(p) for p in [source/'INITIAL_REPEAT.json',ref/'INITIAL_REPEAT.json',source/'INITIALIZATION.json',source/'spec.json']},
  baseline_kernel_caches=baseline,baseline_local_repeat=local,measurements=measurements,preflight=pf,base_and_adapters_unchanged=True)
 write(out/'RESULTS.json',result);print('pair probe complete',flush=True)
if __name__=='__main__':main()
