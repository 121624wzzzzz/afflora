"""Native complete-prefix joint likelihood; no KV-reuse scoring optimization."""
import torch
from modeling import batch

def positions(mask):
    p=mask.long().cumsum(-1)-1;p.masked_fill_(mask==0,1);return p

@torch.inference_mode()
def candidate_scores(model,items,codes):
    assert len(codes)==50 and all(len(c)==2 for c in codes)
    prefixes=sorted({c[0] for c in codes});assert len(prefixes)==5
    result=torch.empty((len(items),len(codes)),device=model.get_input_embeddings().weight.device)
    for first in prefixes:
        extended=[dict(r,prompt_ids=r['prompt_ids']+[first]) for r in items];ids,mask,_=batch(extended,False)
        out=model(input_ids=ids,attention_mask=mask,position_ids=positions(mask),use_cache=False,logits_to_keep=2)
        probs=out.logits.float().log_softmax(-1)
        for j,(a,b) in enumerate(codes):
            if a==first:result[:,j]=probs[:,0,a]+probs[:,1,b]
    assert torch.isfinite(result).all()
    return result

@torch.inference_mode()
def verify_candidates(model,items,codes):
    got=candidate_scores(model,items,codes);ref=torch.empty_like(got)
    # Independent complete forwards plus native causal-LM cross entropy.
    # Explicit shifted targets supervise exactly the two code tokens, without EOS.
    for first in sorted({x[0] for x in codes}):
        extended=[dict(r,prompt_ids=r['prompt_ids']+[first]) for r in items];ids,mask,_=batch(extended,False)
        out=model(input_ids=ids,attention_mask=mask,position_ids=positions(mask),use_cache=False,logits_to_keep=2)
        for j,(a,b) in enumerate(codes):
            if a!=first:continue
            target=torch.tensor([[a,b]],device=out.logits.device)
            for bi in range(len(items)):
                ref[bi,j]=-2*model.loss_function(out.logits[bi:bi+1].contiguous(),labels=None,
                    vocab_size=model.config.vocab_size,shift_labels=target)
    err=float((got-ref).abs().max());assert err<2e-4,err
    assert torch.equal(got.argmax(-1),ref.argmax(-1))
    return {'max_abs_joint_logprob_error':err,'argmax_equal':True,'examples':len(items),'candidate_count':len(codes),
        'scoring_path':'five native complete-prefix forwards; no cache',
        'reference':'Independent complete-prefix forwards with native ForCausalLMLoss on two explicit shifted target tokens'}
