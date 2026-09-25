import argparse,math,time
import torch
from common import *
from modeling import build,load_adapter,adapter_state,digest,enabled,frozen_digest

def tensors(items,eos):
 width=max(map(len,items));ids=torch.tensor([[eos]*(width-len(x))+x for x in items],device='cuda');mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in items],device='cuda');return ids,mask

def logits_last(model,ids,mask):return model.lm_head(model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state[:,-1,:]).float()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args();job=next(j for j in read(HERE/'JOBS.json') if j['name']==args.name)
 for rel,h in read(HERE/'CODE_FROZEN.json')['files'].items():assert sha(HERE/rel)==h,rel
 audit=read(HERE/'INPUT_AUDIT.json')
 if job['checkpoint']:
  cp=Path(job['checkpoint']);assert sha(cp/'adapter.safetensors')==job['checkpoint_sha256']
  for f,h in audit['source_checkpoint_files'].items():
   if Path(f).parent==cp:assert sha(f)==h
 model,initial=build(job['spec']);model.eval()
 if job['checkpoint']:
  load_adapter(model,Path(job['checkpoint'])/'adapter.safetensors');assert digest(adapter_state(model))==job['tensor_sha256']
 model.float();model.requires_grad_(False)
 base_before=frozen_digest(model);adapter_before=digest(adapter_state(model));eos=model.config.eos_token_id
 root=HERE/'results'/job['name'];root.mkdir(exist_ok=False);write(root/'JOB.json',job)
 summaries={};preflight={};started=time.monotonic()
 with torch.inference_mode():
  for task in ['sciq','anli_r1']:
   path=HERE/f"tokens/{job['model']}_{task}.json";assert sha(path)==audit['files'][str(path)];data=read(path);labels=data['label_ids'];rows_=data['rows'];n=len(labels)
   pilot=[r['variants'][0]['input_ids'] for r in rows_[:2]];ids,mask=tensors(pilot,eos)
   selected=logits_last(model,ids,mask);full=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,-1,:].float()
   error=float((selected-full).abs().max());assert error<2e-4,error
   enabled(model,False);disabled=logits_last(model,ids,mask)[:,labels].cpu().tolist();enabled(model,True);model.requires_grad_(False)
   if job['method']!='base':
    ref=read(HERE/'results'/('base__'+job['model'])/'COMPLETE.json')['preflight'][task]['disabled_candidate_logits']
    base_error=max(abs(x-y) for a,b in zip(disabled,ref) for x,y in zip(a,b));assert base_error<2e-4,base_error
   else:base_error=0.
   preflight[task]=dict(full_forward_last_logits_error=error,disabled_vs_base_error=base_error,disabled_candidate_logits=disabled)
   del selected,full,ids,mask
   flat=[(i,k,v) for i,r in enumerate(rows_) for k,v in enumerate(r['variants'])];accum=[[] for r in rows_];task_start=time.monotonic()
   for start in range(0,len(flat),16):
    chunk=flat[start:start+16];ids,mask=tensors([v['input_ids'] for i,k,v in chunk],eos);logits=logits_last(model,ids,mask)
    logp=(logits[:,labels]-torch.logsumexp(logits,dim=-1,keepdim=True)).cpu().tolist()
    for (i,k,v),scores in zip(chunk,logp):
     assert all(math.isfinite(x) for x in scores);accum[i].append(dict(rotation=k,order=v['order'],candidate_logprobs=scores))
    if start%256==0:write(root/'PROGRESS.json',dict(at=now(),task=task,variants_done=start+len(chunk),variants_total=len(flat),seconds=time.monotonic()-task_start))
   records=[]
   for r,variants in zip(rows_,accum):
    assert len(variants)==n;semantic=[0.]*n;rot_correct=[];mass=[]
    for v in variants:
     mx=max(v['candidate_logprobs']);probs=[math.exp(x-mx) for x in v['candidate_logprobs']];total=sum(probs);probs=[x/total for x in probs]
     for k,original in enumerate(v['order']):semantic[original]+=probs[k]/n
     pred=v['order'][max(range(n),key=lambda k:probs[k])];rot_correct.append(pred==r['gold']);mass.append(sum(math.exp(x) for x in v['candidate_logprobs']))
    prediction=max(range(n),key=lambda k:semantic[k]);records.append(dict(id=r['id'],gold=r['gold'],prediction=prediction,correct=prediction==r['gold'],semantic_probabilities=semantic,rotation_correct=rot_correct,candidate_mass=sum(mass)/n,variants=variants))
   write_rows(root/f'{task}.jsonl',records)
   summaries[task]=dict(n=len(records),accuracy=100*sum(r['correct'] for r in records)/len(records),rotation_accuracy=[100*sum(r['rotation_correct'][k] for r in records)/len(records) for k in range(n)],correct_probability=sum(r['semantic_probabilities'][r['gold']] for r in records)/len(records),candidate_mass=sum(r['candidate_mass'] for r in records)/len(records),seconds=time.monotonic()-task_start,responses_sha256=sha(root/f'{task}.jsonl'))
   write(root/'SUMMARY.json',summaries)
 assert frozen_digest(model)==base_before and digest(adapter_state(model))==adapter_before
 assert all(not p.requires_grad and p.grad is None for p in model.parameters())
 write(root/'COMPLETE.json',dict(at=now(),status='passed',seconds=time.monotonic()-started,preflight=preflight,base_frozen=True,adapter_frozen=True,summary_sha256=sha(root/'SUMMARY.json')))
if __name__=='__main__':main()
