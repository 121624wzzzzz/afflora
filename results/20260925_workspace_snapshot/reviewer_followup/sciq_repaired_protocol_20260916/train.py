import argparse,math,time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from settings import *
from modeling import build,adapter_state,digest,frozen_digest,batch,forward_selected,set_adapter_enabled

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();spec=read(a.spec)
    cp=Path(spec['checkpoint']);cp.mkdir(parents=True,exist_ok=False);write(cp/'spec.json',spec)
    model,audit=build(spec);model.cuda();frozen_before=frozen_digest(model)
    data=read(HERE/'tokens'/f"{spec['model']}_train.json");meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
    # Zero-start equality against the identical frozen model with its map disabled.
    model.eval();ids,mask,lengths=batch(data[:3],meta,train=True)
    with torch.inference_mode():
        active=forward_selected(model,ids,mask,lengths,train=True)
        set_adapter_enabled(model,False);inactive=forward_selected(model,ids,mask,lengths,train=True)
        set_adapter_enabled(model,True)
        zero_error=float((active-inactive).abs().max());assert zero_error==0,zero_error
    del active,inactive
    audit.update(frozen_parameters_before_sha256=frozen_before,zero_start_max_abs_logit_error=zero_error)
    write(cp/'INITIALIZATION.json',audit)
    model.train();torch.cuda.reset_peak_memory_stats()
    generator=torch.Generator().manual_seed(spec['seed']);order=torch.randperm(len(data),generator=generator).tolist()
    if spec['phase']=='smoke':order=order[:64]
    write(cp/'TRAIN_ORDER.json',{'indices':order})
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert set(n for n,p in named)==set(audit['trainable_names'])
    opt=torch.optim.AdamW([p for n,p in named],lr=spec['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert {id(p) for group in opt.param_groups for p in group['params']}=={id(p) for n,p in named}
    steps=math.ceil(len(order)/32);warmup=math.ceil(.03*steps);losses=[];grads=[];started=time.monotonic();backwards=0
    for step in range(steps):
        selected=order[step*32:(step+1)*32];opt.zero_grad(set_to_none=True)
        scale=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
        for group in opt.param_groups:group['lr']=spec['lr']*scale
        total=0.
        for begin in range(0,len(selected),8):
            items=[data[i] for i in selected[begin:begin+8]];ids,mask,lengths=batch(items,meta,train=True)
            gold=torch.tensor([[meta['label_ids'][x['gold']],meta['eos_id']] for x in items],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=forward_selected(model,ids,mask,lengths,train=True)
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),gold.reshape(-1))
            assert torch.isfinite(loss),step
            (loss*len(items)/len(selected)).backward();backwards+=1
            assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
            total+=float(loss.detach())*len(items)/len(selected)
        norm=torch.nn.utils.clip_grad_norm_([p for n,p in named],1.,error_if_nonfinite=True)
        opt.step();losses.append(total);grads.append(float(norm))
        if step%20==0 or step==steps-1:
            record={'step':step+1,'steps':steps,'loss':total,'grad_norm':float(norm),'lr':opt.param_groups[0]['lr'],'elapsed_s':time.monotonic()-started}
            print(json.dumps(record),flush=True);write(cp/'PROGRESS.json',record)
    frozen_after=frozen_digest(model);assert frozen_before==frozen_after
    state=adapter_state(model);assert all(torch.isfinite(v).all() for v in state.values())
    if spec['arm'] in ['hidden_r8','small_q']:
        assert all(v.norm()>0 for k,v in state.items() if '.lora_B.' in k)
        assert model._boundary_counts=={'input':0,'output':0}
    else:
        assert model.standalone_affine.up.weight.norm()>0
        assert model._boundary_counts[spec['arm']]>0 and model._boundary_counts[{'input':'output','output':'input'}[spec['arm']]]==0
    save_file(state,str(cp/'adapter.safetensors'))
    write(cp/'TRAINING.json',{'steps':steps,'examples':len(order),'losses':losses,'gradient_norms':grads,'seconds':time.monotonic()-started,
        'adapter_sha256':sha(cp/'adapter.safetensors'),'tensor_sha256':digest(state),'hook_counts':model._boundary_counts,
        'frozen_parameters_before_sha256':frozen_before,'frozen_parameters_after_sha256':frozen_after,
        'frozen_parameters_bitwise_unchanged':True,'base_gradient_checks':backwards,'optimizer_whitelist_verified':True,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'finished_at':now()})
    if spec['phase']=='smoke':
        model.eval().float();items=data[:3];ids,mask,lengths=batch(items,meta,train=True)
        with torch.inference_mode():
            got=forward_selected(model,ids,mask,lengths,train=True)
            all_logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
            positions=torch.stack([lengths-2,lengths-1],1);reference=all_logits[torch.arange(3,device='cuda')[:,None],positions]
            error=float((got-reference).abs().max());assert error<2e-4,error
            gold=torch.tensor([[meta['label_ids'][x['gold']],meta['eos_id']] for x in items],device='cuda')
            optimized=F.cross_entropy(got.reshape(-1,got.shape[-1]),gold.reshape(-1))
            targets=torch.full(ids.shape,-100,device='cuda',dtype=torch.long);targets[torch.arange(3,device='cuda')[:,None],positions]=gold
            full=F.cross_entropy(all_logits.reshape(-1,got.shape[-1]),targets.reshape(-1));loss_error=float(abs(optimized-full))
            # Separate masking identity from shape-dependent FP32 GEMM noise.
            selected_reference=F.cross_entropy(reference.double().reshape(-1,got.shape[-1]),gold.reshape(-1))
            full_reference=F.cross_entropy(all_logits.double().reshape(-1,got.shape[-1]),targets.reshape(-1))
            exact_mask_error=float(abs(selected_reference-full_reference));assert exact_mask_error<1e-12
            assert loss_error<=2*error+2e-6,(loss_error,error)
        torch.save({'ids':ids.cpu(),'mask':mask.cpu(),'lengths':lengths.cpu(),'logits':got.cpu(),'full_logit_error':error,
            'masked_loss_error':loss_error,'exact_mask_identity_error':exact_mask_error,
            'tensor_sha256':digest(adapter_state(model))},cp/'SMOKE_FORWARD.pt')
    print(json.dumps({'status':'trained','name':spec['name']}),flush=True)

if __name__=='__main__':main()
