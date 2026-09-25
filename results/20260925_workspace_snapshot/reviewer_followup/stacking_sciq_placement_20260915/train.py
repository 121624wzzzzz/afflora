import argparse,math,time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from common import *
from modeling import build,adapter_state,digest,batch,forward_selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);args=p.parse_args()
    spec=read(args.spec);cp=Path(spec['checkpoint']);cp.mkdir(parents=True,exist_ok=False)
    write(cp/'spec.json',spec)
    model,audit=build(spec);write(cp/'INITIALIZATION.json',audit)
    model.cuda();model.train();torch.cuda.reset_peak_memory_stats()
    data=read(HERE/'tokens'/f"{spec['model']}_train.json")
    meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
    generator=torch.Generator().manual_seed(spec['seed'])
    order=torch.randperm(len(data),generator=generator).tolist()
    if spec['phase']=='smoke':order=order[:64]
    write(cp/'TRAIN_ORDER.json',{'indices':order,'sha256':__import__('hashlib').sha256(json.dumps(order).encode()).hexdigest()})
    steps=math.ceil(len(order)/32)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec['lr'],
        betas=(.9,.999),eps=1e-8,weight_decay=0)
    warmup=math.ceil(.03*steps)
    losses=[];grads=[];started=time.monotonic()
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
            (loss*len(items)/len(selected)).backward();total+=float(loss.detach())*len(items)/len(selected)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step();losses.append(total);grads.append(float(norm))
        if step%10==0 or step==steps-1:
            record={'step':step+1,'steps':steps,'loss':total,'grad_norm':float(norm),'lr':opt.param_groups[0]['lr'],'elapsed_s':time.monotonic()-started}
            print(json.dumps(record),flush=True);write(cp/'PROGRESS.json',record)
    state=adapter_state(model)
    assert all(torch.isfinite(v).all() for v in state.values())
    affine_up={n:float(v.norm()) for n,v in state.items() if 'placement_maps.' in n and '.up.' in n}
    if spec['arm'] in ['both','interior']:
        assert len(affine_up)==2 and all(x>0 for x in affine_up.values())
        assert all(x>0 for x in model._placement_counts.values())
    save_file(state,str(cp/'adapter.safetensors'))
    metrics={'steps':steps,'examples':len(order),'losses':losses,'gradient_norms':grads,'seconds':time.monotonic()-started,
        'adapter_sha256':sha(cp/'adapter.safetensors'),'tensor_sha256':digest(state),
        'affine_up_norms':affine_up,'hook_counts':model._placement_counts,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'finished_at':now()}
    write(cp/'TRAINING.json',metrics)
    if spec['phase']=='smoke':
        model.eval().float();items=data[:3];ids,mask,lengths=batch(items,meta,train=True)
        with torch.inference_mode():
            got=forward_selected(model,ids,mask,lengths,train=True)
            all_logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
            positions=torch.stack([lengths-2,lengths-1],1)
            reference=all_logits[torch.arange(3,device='cuda')[:,None],positions]
            error=float((got-reference).abs().max());assert error<2e-4,error
            gold=torch.tensor([[meta['label_ids'][x['gold']],meta['eos_id']] for x in items],device='cuda')
            optimized_loss=F.cross_entropy(got.reshape(-1,got.shape[-1]),gold.reshape(-1))
            mask_labels=torch.full(ids.shape,-100,device='cuda',dtype=torch.long)
            mask_labels[torch.arange(3,device='cuda')[:,None],positions]=gold
            full_loss=F.cross_entropy(all_logits.reshape(-1,got.shape[-1]),mask_labels.reshape(-1))
            loss_error=float(abs(optimized_loss-full_loss));assert loss_error<1e-5,loss_error
        torch.save({'ids':ids.cpu(),'mask':mask.cpu(),'lengths':lengths.cpu(),'logits':got.cpu(),
                    'full_logit_error':error,'masked_loss_error':loss_error,'tensor_sha256':digest(state)},cp/'SMOKE_FORWARD.pt')
    print(json.dumps({'status':'trained','name':spec['name'],'steps':steps,'seconds':metrics['seconds']}),flush=True)

if __name__=='__main__':main()
