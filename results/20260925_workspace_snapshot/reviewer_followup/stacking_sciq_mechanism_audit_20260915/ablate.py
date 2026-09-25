"""Conditional input/output knockouts of already trained bilateral adapters."""
import argparse,json,time
import numpy as np
import torch
from support import *
from modeling import load_checkpoint,batch,adapter_state,digest

MODES=['full','input_only','output_only','both_off']

def summarize(scores,unrestricted,logmass,data):
    clean=np.array([not x['ambiguous_gold'] for x in data]);gold=np.array([x['gold'] for x in data]);rotation=np.array([x['rotation'] for x in data])
    pred=scores.argmax(-1);correct=pred==gold
    logz=np.logaddexp.reduce(scores.astype(np.float64),axis=-1);nll=logz-scores[np.arange(len(scores)),gold]
    label_ids=np.array([32,33,34,35]);valid=np.isin(unrestricted,label_ids)
    out={}
    for name,mask in [('canonical',clean&(rotation==0)),('rotations',clean)]:
        out[name]={'n':int(mask.sum()),'accuracy':float(100*correct[mask].mean()),'candidate_nll':float(nll[mask].mean()),
            'label_mass_penalty':float(-logmass[mask].mean()),'full_vocab_label_nll':float((nll-logmass)[mask].mean()),
            'valid_first_label_percent':float(100*valid[mask].mean()),
            'unrestricted_first_token_accuracy':float(100*(unrestricted[mask]==label_ids[gold[mask]]).mean())}
    semantic=(pred+rotation)%4;out['semantic_consistency_percent']=float(100*np.all(semantic[clean].reshape(998,4)==semantic[clean].reshape(998,4)[:,0,None],axis=1).mean())
    out['all_four_correct_percent']=float(100*np.all(correct[clean].reshape(998,4),axis=1).mean())
    return out

def head_geometry(model):
    base=model.get_base_model();w=base.lm_head.weight[[32,33,34,35]].double().cpu();r=w[:3]-w[3]
    singular=torch.linalg.svdvals(r);rank=int(torch.linalg.matrix_rank(r))
    inv=torch.linalg.pinv(r)
    gen=torch.Generator().manual_seed(20260915);target=torch.randn((3,r.shape[1]),generator=gen,dtype=torch.float64)
    reconstructed=r@inv@target
    out={'relative_label_matrix_shape':list(r.shape),'rank':rank,'singular_values':singular.tolist(),
        'condition_number':float(singular[0]/singular[-1]),'right_inverse_max_error':float((r@inv-torch.eye(3)).abs().max()),
        'arbitrary_rank3_relative_correction_max_error':float((reconstructed-target).abs().max())}
    a=base.placement_maps['output'];u=a.up.weight.detach().double().cpu();v=a.down.weight.detach().double().cpu()
    delta=a.scale*r@u@v;out['learned_relative_update_singular_values']=torch.linalg.svdvals(delta).tolist()
    out['learned_relative_update_frobenius_ratio']=float(delta.norm()/r.norm())
    for key,mod in base.placement_maps.items():
        up=mod.up.weight.detach().double().cpu();down=mod.down.weight.detach().double().cpu()
        ru=torch.linalg.qr(up,mode='reduced').R;rd=torch.linalg.qr(down.T,mode='reduced').R
        ss=torch.linalg.svdvals(mod.scale*ru@rd.T)
        out[key+'_map']={'linear_residual_singular_values':ss.tolist(),'linear_residual_frobenius':float(ss.norm()),
            'linear_residual_spectral':float(ss[0]),'bias_norm':float(mod.bias.norm()) if mod.bias is not None else 0.}
    assert rank==3 and out['arbitrary_rank3_relative_correction_max_error']<1e-10
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args();cp=Path(a.checkpoint)
    for fn in ['spec.json','adapter.safetensors','INITIALIZATION.json','test_predictions.jsonl']:checked(cp/fn)
    spec=read(cp/'spec.json');assert spec['arm']=='both'
    output=HERE/'ablations'/spec['name'];output.mkdir(parents=True,exist_ok=False)
    meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
    data=read(HERE/'tokens'/f"{spec['model']}_test_rotations.json");original=load_predictions(spec)
    assert [(x['id'],x['rotation']) for x in data]==[(x['id'],x['rotation']) for x in original]
    original_scores=np.array([x['label_logits'] for x in original],dtype=np.float32)
    model,_,_=load_checkpoint(cp);before=digest(adapter_state(model));base=model.get_base_model();maps=base.placement_maps
    geometry=head_geometry(model)
    modes=MODES+(['unadapted_base'] if spec['seed']==2002 else [])
    scores={k:np.empty((len(data),4),dtype=np.float32) for k in modes}
    unrestricted={k:np.empty(len(data),dtype=np.int64) for k in modes};logmass={k:np.empty(len(data),dtype=np.float64) for k in modes}
    scales={k:(m.scale,m.bias_scale) for k,m in maps.items()};started=time.monotonic();max_error=0.;flips=0
    label_ids=torch.tensor(meta['label_ids'],device='cuda')
    def set_on(which,on):
        maps[which].scale=scales[which][0]*on;maps[which].bias_scale=scales[which][1]*on
    def store_result(mode,head,begin,end):
        logits=head.float();ll=logits[:,label_ids]
        scores[mode][begin:end]=ll.cpu().numpy();unrestricted[mode][begin:end]=logits.argmax(-1).cpu().numpy()
        logmass[mode][begin:end]=(ll.double().logsumexp(-1)-logits.double().logsumexp(-1)).cpu().numpy()
    with torch.inference_mode():
        for start in range(0,len(data),16):
            items=data[start:start+16];end=start+len(items);ids,mask,lengths=batch(items,meta)
            for inp in [1,0]:
                set_on('input',inp)
                h=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                h=h[torch.arange(len(items),device='cuda'),lengths-1]
                for out in [1,0]:
                    set_on('output',out);mode={(1,1):'full',(1,0):'input_only',(0,1):'output_only',(0,0):'both_off'}[inp,out]
                    store_result(mode,base.lm_head(h),start,end)
            if 'unadapted_base' in modes:
                set_on('input',0);set_on('output',0)
                with model.disable_adapter():
                    h=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                    h=h[torch.arange(len(items),device='cuda'),lengths-1]
                    store_result('unadapted_base',base.lm_head(h),start,end)
            error=float(np.abs(scores['full'][start:end]-original_scores[start:end]).max());max_error=max(max_error,error)
            flips+=int(np.sum(scores['full'][start:end].argmax(-1)!=original_scores[start:end].argmax(-1)))
            assert max_error<=1e-5,(start,max_error)
            if start%512==0:print(json.dumps({'name':spec['name'],'rows':end,'total':len(data),'elapsed':time.monotonic()-started,'replay_error':max_error}),flush=True)
    for k in maps:set_on(k,1)
    assert before==digest(adapter_state(model))
    result={'spec':spec,'at':now(),'seconds':time.monotonic()-started,'status':'complete','scope':'post hoc conditional inference knockouts, no retraining',
        'full_replay_max_abs_logit_error':max_error,'full_replay_prediction_flips':flips,'adapter_tensors_unchanged':True,
        'checkpoint_sha256':sha(cp/'adapter.safetensors'),'geometry':geometry,'metrics':{k:summarize(scores[k],unrestricted[k],logmass[k],data) for k in modes}}
    np.savez_compressed(output/'scores.npz',**{k:scores[k] for k in modes},**{k+'_unrestricted':unrestricted[k] for k in modes},**{k+'_log_label_mass':logmass[k] for k in modes})
    result['scores_sha256']=sha(output/'scores.npz');write(output/'RESULTS.json',result)
    print(json.dumps({'status':'complete','name':spec['name'],'canonical_accuracy':{k:v['canonical']['accuracy'] for k,v in result['metrics'].items()}}),flush=True)

if __name__=='__main__':main()
