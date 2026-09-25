"""Exact joint two-token candidate likelihood, with an independent full-forward check."""
import torch
from modeling import batch

def positions(mask):
    p=mask.long().cumsum(-1)-1;p.masked_fill_(mask==0,1);return p

@torch.inference_mode()
def candidate_scores(model,items,codes):
    ids,mask,_=batch(items,False);codes=torch.tensor(codes,device=ids.device)
    prefixes=sorted(set(codes[:,0].tolist()));assert len(prefixes)==5
    firstidx=torch.tensor([prefixes.index(int(c)) for c in codes[:,0]],device=ids.device)
    out=model.model(input_ids=ids,attention_mask=mask,position_ids=positions(mask),use_cache=True)
    first=model.lm_head(out.last_hidden_state[:,-1:]).float()[:,0,:].log_softmax(-1)
    cache=out.past_key_values;cache.batch_repeat_interleave(len(prefixes))
    prefix_ids=torch.tensor(prefixes,device=ids.device).repeat(len(items))[:,None]
    expanded_mask=torch.cat([mask.repeat_interleave(len(prefixes),0),torch.ones((len(items)*len(prefixes),1),device=ids.device,dtype=mask.dtype)],1)
    out=model.model(input_ids=prefix_ids,attention_mask=expanded_mask,position_ids=positions(expanded_mask)[:,-1:],
        past_key_values=cache,cache_position=torch.tensor([ids.shape[1]],device=ids.device),use_cache=True)
    second=model.lm_head(out.last_hidden_state).float()[:,0,:].log_softmax(-1).reshape(len(items),len(prefixes),-1)
    return first[:,codes[:,0]]+second[:,firstidx,codes[:,1]]

@torch.inference_mode()
def verify_candidates(model,items,codes):
    got=candidate_scores(model,items,codes);ref=torch.empty_like(got)
    # Separate complete forwards per prefix, no reused cache.
    for first in sorted({x[0] for x in codes}):
        extended=[dict(r,prompt_ids=r['prompt_ids']+[first]) for r in items];ids,mask,_=batch(extended,False)
        out=model.model(input_ids=ids,attention_mask=mask,position_ids=positions(mask),use_cache=False)
        probs=model.lm_head(out.last_hidden_state[:,-2:]).float().log_softmax(-1)
        for j,(a,b) in enumerate(codes):
            if a==first:ref[:,j]=probs[:,0,a]+probs[:,1,b]
    err=float((got-ref).abs().max());assert err<2e-4,err
    assert torch.equal(got.argmax(-1),ref.argmax(-1))
    return {'max_abs_joint_logprob_error':err,'argmax_equal':True,'examples':len(items),'candidate_count':len(codes),'reference':'5 independent complete prefix forwards; no cache'}
