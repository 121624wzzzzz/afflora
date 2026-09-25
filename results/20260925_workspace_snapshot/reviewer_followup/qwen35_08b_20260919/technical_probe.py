"""Actual checkpoint identity and text execution checks before any benchmark fit."""
import inspect,time,torch
from safetensors import safe_open
from transformers import AutoConfig,Qwen3_5ForConditionalGeneration
from common import *
from modeling import build,base_precision,is_adapter,loss,batch,adapter_state,digest

def main():
 assert not (HERE/'CODE_FROZEN.json').exists();cfg=read(HERE/'models.json')[MODEL]
 model,audit=build({'seed':7597,'model':MODEL,'arm':'hidden_both'});model.eval();torch.cuda.synchronize()
 # Independently inspect every real loaded language tensor against pinned shards.
 index=read(Path(cfg['path'])/'model.safetensors.index.json')['weight_map'];checked=0
 by_shard={}
 for name,p in model.named_parameters():
  if is_adapter(name):continue
  raw=name.replace('.base_layer.','.');source=raw.replace('model.','model.language_model.',1)
  assert source in index,(name,source)
  by_shard.setdefault(index[source],[]).append((source,p))
 for shard,items in by_shard.items():
  with safe_open(str(Path(cfg['path'])/shard),framework='pt',device='cpu') as f:
   for name,p in items:
    expected=f.get_tensor(name);assert expected.dtype in [torch.bfloat16,torch.float32],(name,expected.dtype)
    assert expected.dtype==p.dtype,(name,expected.dtype,p.dtype)
    assert torch.equal(expected,p.detach().cpu()),name;checked+=1
 # Full multimodal wrapper reuses these exact language modules; vision is not invoked.
 fullcfg=AutoConfig.from_pretrained(cfg['path'],local_files_only=True);fullcfg._attn_implementation='eager'
 with torch.device('meta'):full=Qwen3_5ForConditionalGeneration(fullcfg)
 full.model.language_model=model.model;full.lm_head=model.lm_head;full.eval()
 base_precision(model,torch.float32)
 ids=torch.tensor([[100,200,300,400,500,600],[TOKEN_PAD_ID,TOKEN_PAD_ID,100,200,300,400]],device='cuda');mask=torch.tensor([[1]*6,[0,0,1,1,1,1]],device='cuda')
 pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,1)
 with torch.inference_mode():
  a=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
  b=full(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
  err=float((a[mask.bool()]-b[mask.bool()]).abs().max());assert err==0,err
  initial=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=True)
  token=torch.tensor([[700],[500]],device='cuda');extended=torch.cat([ids,token],1);em=torch.cat([mask,torch.ones_like(token)],1);ep=em.cumsum(-1)-1;ep.masked_fill_(em==0,1)
  cached=model(input_ids=token,attention_mask=em,position_ids=ep[:,-1:],past_key_values=initial.past_key_values,use_cache=True,logits_to_keep=1).logits.float().log_softmax(-1)
  direct=model(input_ids=extended,attention_mask=em,position_ids=ep,use_cache=False,logits_to_keep=1).logits.float().log_softmax(-1)
  cache_error=float((cached-direct).abs().max());assert cache_error<1e-3,cache_error
 del a,b,initial,cached,direct,full
 base_precision(model,torch.bfloat16);model.train();model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
 # Length-only synthetic workload. No development/confirmation answer is scored.
 ids=(torch.arange(671,device='cuda')[None,:].repeat(2,1)%1000)+100;mask=torch.ones_like(ids);labels=torch.full_like(ids,-100);labels[:,-3:]=ids[:,-3:]
 torch.cuda.synchronize();start=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(model,ids,mask,labels)[0]
 value.backward();torch.cuda.synchronize();seconds=time.monotonic()-start
 assert torch.isfinite(value) and all(p.grad is None for n,p in model.named_parameters() if not is_adapter(n))
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if is_adapter(n))
 result={'at':now(),'status':'passed','source_text_tensors_checked':checked,'vlm_text_logits_max_error':err,'cache_full_logprob_max_error':cache_error,
   'synthetic_microbatch':2,'synthetic_sequence_length':671,'forward_backward_seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
   'model_audit':audit,'scope':'Checkpoint tensors, neutral token parity and synthetic timing only; no benchmark metric.'}
 write(HERE/'CHECKPOINT_RUNTIME_AUDIT.json',result);print(canonical(result),flush=True)
if __name__=='__main__':main()
