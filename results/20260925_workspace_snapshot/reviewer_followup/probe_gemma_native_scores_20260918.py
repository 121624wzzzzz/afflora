"""Zero-update diagnosis of the failed high-LR technical smoke; no task-score selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'gemma_transfer_20260918'
sys.path.insert(0,str(ROOT))
import torch
from common import *
from modeling import build,load_adapter,base_precision,adapter_state,digest,batch
from classification import candidate_scores,positions
def main():
 name='smoke_trec50_hidden_both_c5_s7399';root=ROOT/'checkpoints'/name
 spec=read(root/'spec.json');model,init=build(spec);load_adapter(model,root/'adapter.safetensors')
 before=digest(adapter_state(model));base_precision(model,torch.float32);model.eval()
 tokens=read(ROOT/f'tokens/trec50_{MODEL}_dev.json')[:4]
 items=[min(tokens,key=lambda x:len(x['prompt_ids'])),max(tokens,key=lambda x:len(x['prompt_ids']))];codes=read(ROOT/f'tokens/{MODEL}_codes.json')
 with torch.inference_mode():
  cached=candidate_scores(model,items,codes);native=torch.empty_like(cached);native_loss=torch.empty_like(cached)
  for first in sorted({x[0] for x in codes}):
   ids,mask,_=batch([dict(r,prompt_ids=r['prompt_ids']+[first]) for r in items],False)
   out=model(input_ids=ids,attention_mask=mask,position_ids=positions(mask),use_cache=False,logits_to_keep=2)
   probs=out.logits.float().log_softmax(-1)
   for j,(a,b) in enumerate(codes):
    if a!=first:continue
    native[:,j]=probs[:,0,a]+probs[:,1,b]
    for bi in range(len(items)):
     native_loss[bi,j]=-2*model.loss_function(out.logits[bi:bi+1].contiguous(),labels=None,vocab_size=model.config.vocab_size,
         shift_labels=torch.tensor([[a,b]],device=out.logits.device))
  diffs=(cached-native).abs();vals=native.topk(2,dim=-1).values
 result={'at':now(),'status':'diagnosed','scope':'Same saved high-LR smoke adapter and two existing smoke examples; zero training updates; no development configuration selection.',
  'adapter_sha256':sha(root/'adapter.safetensors'),'parameter_tensors_unchanged':before==digest(adapter_state(model)),
  'cached_native_max_abs_logprob_error':float(diffs.max()),'cached_native_argmax_equal':torch.equal(cached.argmax(-1),native.argmax(-1)),
  'native_class_margin':(vals[:,0]-vals[:,1]).tolist(),'native_loss_joint_max_abs_error':float((native-native_loss).abs().max()),
  'all_logprob_errors':diffs.tolist(),'native_joint_vs_loss_argmax_equal':torch.equal(native.argmax(-1),native_loss.argmax(-1)),
  'revision_policy':'Use native complete-prefix scoring for every TREC evaluation, avoiding KV-reuse rounding differences. Keep the existing 2e-4 audit tolerance and all candidates/seeds/metrics unchanged.'}
 assert result['parameter_tensors_unchanged'] and result['native_loss_joint_max_abs_error']<2e-4
 write(ROOT/'NATIVE_SCORE_DIAGNOSIS.json',result);print(canonical({k:v for k,v in result.items() if k!='all_logprob_errors'}),flush=True)
if __name__=='__main__':main()
