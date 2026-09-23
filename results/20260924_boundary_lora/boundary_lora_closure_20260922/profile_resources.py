"""Sequential same-device resource microbenchmark, not an accuracy experiment."""
import time,gc,statistics,subprocess
import torch
from common import *
from modeling import build,batch,loss,load_adapter,base_precision
from plan import spec,variant

def synced():torch.cuda.synchronize();return time.perf_counter()
def prefill(model,ids,mask):return model.lm_head(model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state[:,-1]).float()
def main():
 old=Path(read(HERE/'SOURCE.json')['study']);records=read(old/'RESULTS.json')['records'];reused=read(old/'REUSE_AUDIT.json')['runs'];out=[]
 for modelname in ['qwen3_06b_base','qwen25_15b_base']:
  for repeat in range(3):
   variants=[('none','none',0),('affine','shared',16),('vocab','shared',1)];variants=variants[repeat:]+variants[:repeat]
   for method,place,rank in variants:
    s=spec(modelname,'cluener',variant(method,place,rank),6100,'resource');match=next(r for r in records if all(r['spec'][k]==s[k] for k in ['model','task','method','placement','rank','seed']) and r['spec']['stage']=='common' and r['spec']['bias']==s['bias'])
    cp=(Path(next(r['checkpoint'] for r in reused if r['spec']['name']==match['spec']['name'])) if match['reused'] else old/'checkpoints'/match['spec']['name'])
    model,init=build(s);load_adapter(model,cp/'adapter.safetensors');data=read(HERE/f'tokens/cluener_{modelname}_train.json');ids,mask,labels=batch(data[:2]);params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=2e-4,weight_decay=0);times=[]
    for i in range(12):
     if i==2:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
     model.train();start=synced();opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(model,ids,mask,labels)[0]
     value.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();duration=synced()-start
     if i>=2:times.append(duration)
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();del opt
    model.eval();base_precision(model,torch.float32);pi,pm,_=batch(data[:2],False)
    def generate():
     with torch.inference_mode():return model.generate(input_ids=pi,attention_mask=pm,do_sample=False,min_new_tokens=32,max_new_tokens=32,eos_token_id=None,pad_token_id=model.config.eos_token_id,use_cache=True)
    generate();unmerged=[]
    for i in range(3):start=synced();generate();unmerged.append(synced()-start)
    with torch.inference_mode():before=prefill(model,pi,pm)
    merge_error=0.
    if method!='none':
     a=model.boundary_input;w=model.get_input_embeddings().weight
     with torch.no_grad():
      if method=='affine':delta=(w@a.down.weight.T)@a.up.weight.T*a.scale;a_bias=a.bias
      else:delta=a.left@a.right*a.scale;a_bias=None
      w.add_(delta)
      if a_bias is not None:w.add_(a.bias*a.bias_scale)
     for handle in model._boundary_handles:handle.remove()
     model.lm_head=model.lm_head.base_head
     assert model.lm_head.weight.data_ptr()==model.get_input_embeddings().weight.data_ptr()
     with torch.inference_mode():after=prefill(model,pi,pm)
     merge_error=float((before-after).abs().max());torch.testing.assert_close(before,after,rtol=1e-4,atol=5e-4)
     del delta,after
    generate();merged=[]
    for i in range(3):start=synced();generate();merged.append(synced()-start)
    row=dict(model=modelname,method=method,repeat=repeat,trainable_parameters=init['trainable_parameters'],boundary_parameters=init['parameter_groups']['boundary'],adapter_file_bytes=(cp/'adapter.safetensors').stat().st_size,training_step_seconds=times,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,generation_32_tokens_unmerged_seconds=unmerged,generation_32_tokens_boundary_merged_seconds=merged,merge_logits_max_abs_error=merge_error,device=torch.cuda.get_device_name(),microbatch=2,train_sequence_length=ids.shape[1],source_checkpoint=str(cp))
    out.append(row);write(HERE/'RESOURCE_BENCHMARK.json',dict(at=now(),status='running',records=out,scope='Two fixed CLUENER training examples; 2 warmup + 10 optimizer steps; 3 timed 32-token generations; boundary merge only, H remains identical architecture; no task scores. Shared hardware can affect timing.'))
    print(modelname,method,repeat,'train step',statistics.median(times),'peak GB',peak/2**30,'merge error',merge_error,flush=True)
    del model,params,ids,mask,labels,pi,pm,before,value
    if method!='none':del a,w
    gc.collect();torch.cuda.empty_cache()
 result=read(HERE/'RESOURCE_BENCHMARK.json');result['status']='passed';write(HERE/'RESOURCE_BENCHMARK.json',result)
if __name__=='__main__':main()
