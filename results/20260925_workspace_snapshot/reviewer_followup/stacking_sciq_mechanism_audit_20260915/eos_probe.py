import argparse, time
import numpy as np
import torch
from support import *
from modeling import load_checkpoint, batch, adapter_state, digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args();cp=Path(a.checkpoint)
    for fn in ['spec.json','adapter.safetensors','INITIALIZATION.json']:checked(cp/fn)
    spec=read(cp/'spec.json');output=HERE/'eos'/spec['name'];output.mkdir(parents=True,exist_ok=False)
    data=read(HERE/'tokens'/f"{spec['model']}_test_rotations.json")
    meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
    original=load_predictions(spec)
    model,_,_=load_checkpoint(cp);base=model.get_base_model();maps=base.placement_maps
    before=digest(adapter_state(model));scales={k:(m.scale,m.bias_scale) for k,m in maps.items()}
    modes=['full','input_only','output_only','both_off'];scores={m:np.empty(len(data),dtype=np.float64) for m in modes}
    def on(k,enabled):maps[k].scale=scales[k][0]*enabled;maps[k].bias_scale=scales[k][1]*enabled
    started=time.monotonic();replay_error=None
    with torch.inference_mode():
        for start in range(0,len(data),16):
            items=data[start:start+16];end=start+len(items);ids,mask,lengths=batch(items,meta,train=True)
            for inp in [1,0]:
                on('input',inp)
                h=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                if start==0 and inp==1:
                    on('output',1)
                    ll=base.lm_head(h[torch.arange(len(items),device='cuda'),lengths-2])[:,meta['label_ids']].float().cpu().numpy()
                    replay_error=float(np.max(np.abs(ll-np.array([r['label_logits'] for r in original[:len(items)]]))))
                    assert replay_error<1e-4,replay_error
                h=h[torch.arange(len(items),device='cuda'),lengths-1]
                for out in [1,0]:
                    on('output',out);mode={(1,1):'full',(1,0):'input_only',(0,1):'output_only',(0,0):'both_off'}[inp,out]
                    logits=base.lm_head(h).double()
                    scores[mode][start:end]=(logits.logsumexp(-1)-logits[:,meta['eos_id']]).cpu().numpy()
            if start%1024==0:print(json.dumps({'name':spec['name'],'rows':end,'elapsed':time.monotonic()-started}),flush=True)
    for k in maps:on(k,1)
    assert before==digest(adapter_state(model))
    clean=np.array([not x['ambiguous_gold'] for x in data]);canonical=clean&np.array([x['rotation']==0 for x in data])
    result={'spec':spec,'at':now(),'seconds':time.monotonic()-started,'status':'complete','adapter_tensors_unchanged':True,
        'prefix_label_replay_max_abs_error':replay_error,
        'eos_nll':{m:{split:float(scores[m][mask].mean()) for split,mask in [('canonical',canonical),('rotations',clean)]} for m in modes}}
    np.savez_compressed(output/'scores.npz',**scores);result['scores_sha256']=sha(output/'scores.npz')
    write(output/'RESULTS.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
