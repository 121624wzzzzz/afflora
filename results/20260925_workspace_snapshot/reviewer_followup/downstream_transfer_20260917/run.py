import argparse,math,time,os
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_file
from common import *
from modeling import *
from scoring import score,aggregate

def evaluate(model,spec,tag,limit=None):
    split=spec['eval_split']
    data=rows(HERE/f"data/{spec['task']}_{split}.jsonl")
    tokens=read(HERE/f"tokens/{spec['task']}_{spec['model']}_{split}.json")
    if limit is not None:data=data[:limit];tokens=tokens[:limit]
    assert [r['id'] for r in data]==[r['id'] for r in tokens]
    tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[spec['model']]['path'],local_files_only=True)
    out=HERE/'evaluations'/spec['name']/tag;out.mkdir(parents=True,exist_ok=False)
    base_precision(model,torch.float32);model.eval();records=[];started=time.monotonic()
    label_ids=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABC']
    for start in range(0,len(data),16):
        chunk=tokens[start:start+16];ids,mask,_=batch(chunk,False)
        if spec['task']=='anli_r1':
            with torch.inference_mode():
                positions=mask.long().cumsum(-1)-1;positions.masked_fill_(mask==0,1)
                hidden=model.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state[:,-1:]
                logits=model.lm_head(hidden).float()[:,0,:]
                selected=logits[:,label_ids];chosen=selected.argmax(-1);unrestricted=logits.argmax(-1)
                mass=(selected.logsumexp(-1)-logits.logsumexp(-1)).exp()
            for i,r in enumerate(data[start:start+16]):
                pred='ABC'[int(chosen[i])];raw=int(unrestricted[i]);gold_id=label_ids['ABC'.index(r['target'])]
                records.append({'id':r['id'],'predicted_label':pred,'label_logits':selected[i].tolist(),
                    'label_probability_mass':float(mass[i]),'unrestricted_next_token_id':raw,
                    'content_correct':pred==r['target'],'unrestricted_correct':raw==gold_id,'unrestricted_valid':raw in label_ids})
        else:
            with torch.inference_mode():
                generated=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=256,
                    eos_token_id=151643,pad_token_id=151643,use_cache=True)
            for r,seq in zip(data[start:start+16],generated[:,ids.shape[1]:].tolist()):
                ended=151643 in seq;seq=seq[:seq.index(151643)] if ended else seq
                text=tok.decode(seq,skip_special_tokens=False);s=score(text,r)
                records.append(dict(s,id=r['id'],text=text,token_ids=seq,native_eos=ended,capped=not ended and len(seq)>=256,length=len(seq)))
        write(out/'PROGRESS.json',{'done':len(records),'total':len(data),'seconds':time.monotonic()-started})
        write_rows(out/'responses.jsonl',records)
        if start%128==0:print(json.dumps({'evaluation':tag,'done':len(records),'total':len(data)}),flush=True)
    summary=aggregate(records,spec['task']);summary.update(seconds=time.monotonic()-started,spec=spec,tag=tag,
        input_ids_sha256=htext(canonical([r['prompt_ids'] for r in tokens])),responses_sha256=sha(out/'responses.jsonl'))
    if spec['task']=='wikisql':summary.update(native_eos_pct=100*sum(x['native_eos'] for x in records)/len(records),
        capped_pct=100*sum(x['capped'] for x in records)/len(records),mean_generated_tokens=sum(x['length'] for x in records)/len(records))
    else:summary['mean_label_probability_mass']=sum(x['label_probability_mass'] for x in records)/len(records)
    write(out/'SUMMARY.json',summary);base_precision(model,torch.bfloat16);return summary

def preflight(model,spec,data):
    model.eval();base_precision(model,torch.float32)
    items=sorted(data,key=lambda r:len(r['prompt_ids'])+len(r['target_ids']))[:2];ids,mask,labels=batch(items)
    with torch.inference_mode():
        adapted=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        enabled(model,False);original=model(input_ids=ids,attention_mask=mask,use_cache=False).logits;enabled(model,True)
        zero_error=float((adapted-original).abs().max());assert zero_error==0,zero_error
        hf=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False)
        ref=torch.nn.functional.cross_entropy(hf.logits[:,:-1].float().reshape(-1,hf.logits.shape[-1]),labels[:,1:].reshape(-1))
        hf_error=float(abs(ref-hf.loss));assert hf_error<2e-5,hf_error
        optimized,_,_=loss(model,ids,mask,labels)
        selected_error=float(abs(optimized-ref));assert selected_error<2e-4,selected_error
        valid=labels[:,1:]!=-100
        exact=torch.nn.functional.cross_entropy(hf.logits[:,:-1][valid].double(),labels[:,1:][valid])
        masked=torch.nn.functional.cross_entropy(hf.logits[:,:-1].double().reshape(-1,hf.logits.shape[-1]),labels[:,1:].reshape(-1))
        assert abs(float(exact-masked))<1e-12
    base_precision(model,torch.bfloat16)
    return {'zero_residual_max_abs_error':zero_error,'hf_shifted_loss_error':hf_error,'selected_loss_error':selected_error,'exact_mask_error':float(abs(exact-masked))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();spec=read(a.spec)
    frozen=read(HERE/'CODE_FROZEN.json')
    for rel,h in frozen['files'].items():assert sha(HERE/rel)==h,rel
    root=HERE/'checkpoints'/spec['name'];root.mkdir(parents=True,exist_ok=False);write(root/'spec.json',spec)
    model,audit=build(spec);data=read(HERE/f"tokens/{spec['task']}_{spec['model']}_{spec.get('train_split','train')}.json")
    audit['preflight']=preflight(model,spec,data);audit['frozen_before']=frozen_digest(model);write(root/'INITIALIZATION.json',audit)
    if spec['arm']=='base':
        
        for split in ['dev','test']:evaluate(model,dict(spec,eval_split=split),split)
        write(root/'COMPLETE.json',{'at':now(),'status':'passed'});return
    state=adapter_state(model);save_file(state,str(root/'initial_adapter.safetensors'))
    generator=torch.Generator().manual_seed(spec['seed']);order=torch.randperm(len(data),generator=generator).tolist()
    if spec.get('smoke'):order=order[:64]
    write(root/'TRAIN_ORDER.json',{'indices':order,'ids':[data[i]['id'] for i in order]})
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    opt=torch.optim.AdamW([p for _,p in named],lr=spec['lr'],weight_decay=0,betas=(.9,.999),eps=1e-8)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for n,p in named}
    steps=math.ceil(len(order)/32);warmup=math.ceil(.03*steps);started=time.monotonic();history=[];grad_checks=0
    for step in range(steps):
        model.train();opt.zero_grad(set_to_none=True);selected=order[step*32:(step+1)*32]
        factor=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
        for g in opt.param_groups:g['lr']=spec['lr']*factor
        # True token-mean loss across microbatches, including the native EOS.
        denom=sum(len(data[i]['target_ids']) for i in selected);total=0.
        for begin in range(0,len(selected),4):
            items=[data[i] for i in selected[begin:begin+4]];ids,mask,labels=batch(items)
            with torch.autocast('cuda',dtype=torch.bfloat16):_,ce,gold=loss(model,ids,mask,labels);value=ce.sum()/denom
            assert torch.isfinite(value);value.backward();total+=float(value.detach());grad_checks+=1
            assert all(p.grad is None for n,p in model.named_parameters() if not is_adapter(n))
        norm=torch.nn.utils.clip_grad_norm_([p for n,p in named],1.,error_if_nonfinite=True);opt.step()
        record={'step':step+1,'steps':steps,'loss':total,'grad_norm':float(norm),'elapsed_s':time.monotonic()-started}
        history.append(record);write(root/'PROGRESS.json',record)
        if step%8==0 or step==steps-1:print(json.dumps(record),flush=True)
        if step+1 in spec.get('curve_steps',[]):
            save_file(adapter_state(model),str(root/f'adapter_step{step+1}.safetensors'))
            evaluate(model,spec,f'step{step+1}')
    after=frozen_digest(model);assert after==audit['frozen_before']
    assert all(torch.isfinite(p).all() for n,p in named)
    assert all(v['exp_avg'].dtype==torch.float32 and v['exp_avg_sq'].dtype==torch.float32 for v in opt.state.values())
    final=adapter_state(model);save_file(final,str(root/'adapter.safetensors'))
    assert digest(final)!=digest(state)
    # Reload the saved state into the same architecture after destructive zeroing.
    model.eval();base_precision(model,torch.float32);items=data[:2];ids,mask,labels=batch(items)
    with torch.inference_mode():before=float(loss(model,ids,mask,labels)[0])
    with torch.no_grad():
        for n,p in named:p.zero_()
    load_adapter(model,root/'adapter.safetensors')
    with torch.inference_mode():reloaded=float(loss(model,ids,mask,labels)[0])
    assert before==reloaded,(before,reloaded)
    base_precision(model,torch.bfloat16)
    write(root/'TRAINING.json',{'at':now(),'steps':steps,'examples':len(order),'history':history,'frozen_before':audit['frozen_before'],'frozen_after':after,
        'base_gradient_checks':grad_checks,'optimizer_fp32':True,'optimizer_whitelist_verified':True,'adapter_sha256':sha(root/'adapter.safetensors'),
        'adapter_tensor_sha256':digest(final),'reload_loss_error':abs(before-reloaded),'seconds':time.monotonic()-started})
    if not spec.get('smoke'):
        for split in ['dev','test']:evaluate(model,dict(spec,eval_split=split),split)
    write(root/'COMPLETE.json',{'at':now(),'status':'passed'})
if __name__=='__main__':main()
