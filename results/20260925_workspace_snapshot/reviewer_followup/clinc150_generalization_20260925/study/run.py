import argparse,math,time,os
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_file
from common import *
from modeling import *
from scoring import score,aggregate

def evaluate(model,spec,tag,limit=None):
    split=spec['eval_split'];task=spec['task'];cfg=TASK_SETTINGS[task]
    data=rows(HERE/f"data/{task}_{split}.jsonl");tokens=read(HERE/f"tokens/{task}_{spec['model']}_{split}.json")
    if limit is not None:data=data[:limit];tokens=tokens[:limit]
    assert [r['id'] for r in data]==[r['id'] for r in tokens]
    tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[spec['model']]['path'],local_files_only=True)
    out=HERE/'evaluations'/spec['name']/tag;out.mkdir(parents=True,exist_ok=False)
    base_precision(model,torch.float32);model.eval();records=[];started=time.monotonic()
    bs=cfg['test_batch'] if spec.get('smoke') else cfg[split+'_batch'];cap=cfg['max_new_tokens']
    for start in range(0,len(data),bs):
        chunk=tokens[start:start+bs];ids,mask,_=batch(chunk,False)
        with torch.inference_mode():
            generated=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=cap,
                eos_token_id=model.config.eos_token_id,pad_token_id=model.config.eos_token_id,use_cache=True)
        for r,seq in zip(data[start:start+bs],generated[:,ids.shape[1]:].tolist()):
            ended=model.config.eos_token_id in seq;seq=seq[:seq.index(model.config.eos_token_id)] if ended else seq
            text=tok.decode(seq,skip_special_tokens=False);s=score(text,r)
            records.append(dict(s,id=r['id'],text=text,token_ids=seq,native_eos=ended,capped=not ended and len(seq)>=cap,length=len(seq)))
        write(out/'PROGRESS.json',{'done':len(records),'total':len(data),'seconds':time.monotonic()-started})
        write_rows(out/'responses.jsonl',records)
        if start%128==0:print(json.dumps({'evaluation':tag,'done':len(records),'total':len(data)}),flush=True)
    summary=aggregate(records,task);summary.update(seconds=time.monotonic()-started,spec=spec,tag=tag,batch_size=bs,
        native_eos_pct=100*sum(x['native_eos'] for x in records)/len(records),capped_pct=100*sum(x['capped'] for x in records)/len(records),
        mean_generated_tokens=sum(x['length'] for x in records)/len(records),
        input_ids_sha256=htext(canonical([r['prompt_ids'] for r in tokens])),responses_sha256=sha(out/'responses.jsonl'))
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
    inputs=read(HERE/'INPUT_AUDIT.json')['files']
    for split in ['train']+spec.get('eval_splits',['dev','test']):
        for path in [HERE/f"data/{spec['task']}_{split}.jsonl",HERE/f"tokens/{spec['task']}_{spec['model']}_{split}.json"]:assert sha(path)==inputs[str(path)],str(path)
    root=HERE/'checkpoints'/spec['name'];root.mkdir(parents=True,exist_ok=False);write(root/'spec.json',spec)
    model,audit=build(spec);data=read(HERE/f"tokens/{spec['task']}_{spec['model']}_{spec.get('train_split','train')}.json")
    if spec.get('subset_indices'):data=[data[i] for i in spec['subset_indices']]
    matches=[r for r in read(HERE/'REUSE_AUDIT.json')['runs'] if r['spec']['model']==spec['model'] and r['spec']['seed']==spec['seed']]
    for r in matches:assert audit['shared_hidden_initialization_sha256']==r['hidden_initialization_sha256']
    audit['historical_hidden_matches']=len(matches)
    audit['preflight']=preflight(model,spec,data)
    if spec.get('smoke'):
        torch.cuda.reset_peak_memory_stats()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model.train();model.zero_grad(set_to_none=True)
            worst=sorted(data,key=lambda r:len(r['prompt_ids'])+len(r['target_ids']),reverse=True)[:spec['microbatch']]
            ids,mask,labels=batch(worst)
            with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(model,ids,mask,labels)[0]
            value.backward();assert all(p.grad is None for n,p in model.named_parameters() if not is_adapter(n))
            model.zero_grad(set_to_none=True);model.eval()
        audit['worst_length_memory_probe']={'batch_size':len(worst),'sequence_length':ids.shape[1],'max_allocated_bytes':torch.cuda.max_memory_allocated(),'status':'passed'}
        del ids,mask,labels,value
    audit['frozen_before']=frozen_digest(model);write(root/'INITIALIZATION.json',audit)
    if spec['arm']=='base':
        
        for split in spec.get('eval_splits',['dev','test']):evaluate(model,dict(spec,eval_split=split),split)
        write(root/'COMPLETE.json',{'at':now(),'status':'passed'});return
    state=adapter_state(model);save_file(state,str(root/'initial_adapter.safetensors'))
    generator=torch.Generator().manual_seed(spec['seed']);order=torch.randperm(len(data),generator=generator).tolist()
    if spec.get('smoke'):order=order[:64]
    write(root/'TRAIN_ORDER.json',{'indices':order,'ids':[data[i]['id'] for i in order]})
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    groups=[{'params':[p for n,p in named if not n.startswith('boundary_')],'initial_lr':spec['lr']},{'params':[p for n,p in named if n.startswith('boundary_')],'initial_lr':spec['lr']*spec.get('boundary_lr_ratio',1.)}]
    opt=torch.optim.AdamW(groups,lr=spec['lr'],weight_decay=0,betas=(.9,.999),eps=1e-8)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for n,p in named}
    steps=math.ceil(len(order)/32);warmup=math.ceil(.03*steps);started=time.monotonic();history=[];grad_checks=0
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();train_started=time.monotonic()
    for step in range(steps):
        model.train();opt.zero_grad(set_to_none=True);selected=order[step*32:(step+1)*32]
        factor=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
        for g in opt.param_groups:g['lr']=g['initial_lr']*factor
        # True token-mean loss across microbatches, including the native EOS.
        denom=sum(len(data[i]['target_ids']) for i in selected);total=0.
        for begin in range(0,len(selected),spec['microbatch']):
            items=[data[i] for i in selected[begin:begin+spec['microbatch']]];ids,mask,labels=batch(items)
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
    torch.cuda.synchronize()
    resource=dict(training_wall_seconds=time.monotonic()-train_started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name(),note='End-to-end training loop with logging; shared host/GPU contention may affect wall time.')
    write(root/'RESOURCES.json',resource)
    if spec.get('placement')=='shared':
        assert model._boundary_counts['input']>0 and model._boundary_counts['output']>0
        write(root/'SHARED_RUNTIME.json',{'calls':model._boundary_counts,'stored_once':True,'output_transpose':True})
    after=frozen_digest(model);assert after==audit['frozen_before']
    assert all(torch.isfinite(p).all() for n,p in named)
    assert all(v['exp_avg'].dtype==torch.float32 and v['exp_avg_sq'].dtype==torch.float32 for v in opt.state.values())
    final=adapter_state(model);save_file(final,str(root/'adapter.safetensors'))
    assert digest(final)!=digest(state)
    changed={group:any(not torch.equal(state[n],p) for n,p in final.items() if (n.startswith('boundary_'+group) if group in ['input','output'] else '.lora_' in n)) for group in ['hidden','input','output']}
    present={group:any(n.startswith('boundary_'+group) if group in ['input','output'] else '.lora_' in n for n in final) for group in changed}
    assert changed==present,(changed,present)
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
        'base_gradient_checks':grad_checks,'optimizer_fp32':True,'optimizer_initial_lrs':[g['initial_lr'] for g in opt.param_groups],'optimizer_whitelist_verified':True,'adapter_sha256':sha(root/'adapter.safetensors'),
        'adapter_tensor_sha256':digest(final),'reload_loss_error':abs(before-reloaded),'changed_groups':changed,'present_groups':present,'seconds':time.monotonic()-started})
    if spec.get('smoke'):
        evaluate(model,dict(spec,eval_split='dev'),'smoke_dev',limit=TASK_SETTINGS[spec['task']]['test_batch'])
    else:
        for split in spec.get('eval_splits',['dev','test']):evaluate(model,dict(spec,eval_split=split),split)
    write(root/'COMPLETE.json',{'at':now(),'status':'passed'})
if __name__=='__main__':main()
