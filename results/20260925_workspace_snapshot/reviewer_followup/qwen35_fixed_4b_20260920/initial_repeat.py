"""Paired repeat of the first optimizer batch, without updates or advancing RNG."""
import torch
from common import canonical,htext
from modeling import batch,loss,is_adapter,digest

def initial_repeat(model,data,seed,microbatch):
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed)).tolist()[:32]
    denom=sum(len(data[i]['target_ids']) for i in order)
    snapshots=[];totals=[];contributions=[]
    for repeat in range(2):
        # Each pass begins with exactly the RNG state used by training step one.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model.train();model.zero_grad(set_to_none=True);values=[]
            for begin in range(0,len(order),microbatch):
                ids,mask,labels=batch([data[i] for i in order[begin:begin+microbatch]])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    _,ce,_=loss(model,ids,mask,labels);value=ce.sum()/denom
                assert torch.isfinite(value);value.backward();values.append(float(value.detach()))
            assert all(p.grad is None for n,p in model.named_parameters() if not is_adapter(n))
            grads={n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if is_adapter(n)}
            assert all(torch.isfinite(g).all() for g in grads.values())
            snapshots.append(grads);totals.append(sum(values));contributions.append(values)
    squared_diff=squared_ref=0.;max_abs=0.;exact=True
    for name,g in snapshots[0].items():
        ref=snapshots[1][name];delta=g-ref
        exact=exact and torch.equal(g,ref);max_abs=max(max_abs,float(delta.abs().max()))
        squared_diff+=float(delta.double().square().sum());squared_ref+=float(ref.double().square().sum())
    rms=(squared_diff/max(squared_ref,1e-30))**.5
    diff=abs(totals[0]-totals[1]);model.zero_grad(set_to_none=True)
    result={'losses':totals,'microbatch_losses':contributions,'loss_absolute_difference':diff,
        'gradient_relative_rms':rms,'gradient_max_absolute_difference':max_abs,'gradients_bitwise_equal':exact,'gradient_sha256':[digest(g) for g in snapshots],
        'loss_tolerance':0,'gradient_relative_rms_tolerance':0,
        'order_sha256':htext(canonical(order)),
        'scope':'Same model/device, full first batch, same RNG, no optimizer update. This is not a cross-device or whole-trajectory determinism proof.'}
    result['status']='passed' if diff==0 and rms==0 and exact and contributions[0]==contributions[1] else 'failed'
    return result
