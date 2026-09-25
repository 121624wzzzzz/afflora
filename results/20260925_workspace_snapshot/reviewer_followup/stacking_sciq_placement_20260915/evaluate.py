import argparse,json
import numpy as np
import torch
from common import *
from modeling import load_checkpoint,adapter_state,digest,batch,forward_selected

def metrics(rr):
    return {'n':len(rr),'accuracy':100*float(np.mean([x['correct'] for x in rr])),
        'candidate_nll':float(np.mean([x['nll'] for x in rr])),
        'brier':float(np.mean([x['brier'] for x in rr])),
        'margin':float(np.mean([x['margin'] for x in rr])),
        'unrestricted_first_token_accuracy':100*float(np.mean([x['unrestricted_correct'] for x in rr])),
        'valid_label_rate':100*float(np.mean([x['valid_label'] for x in rr]))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['validation','test','smoke'],required=True);args=p.parse_args()
    cp=Path(args.checkpoint);model,spec,audit=load_checkpoint(cp)
    meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
    if args.split=='smoke':
        saved=torch.load(cp/'SMOKE_FORWARD.pt',weights_only=True,map_location='cpu')
        with torch.inference_mode():
            got=forward_selected(model,saved['ids'].cuda(),saved['mask'].cuda(),saved['lengths'].cuda(),train=True).cpu()
        error=float((got-saved['logits']).abs().max());assert error<=1e-5,error
        assert digest(adapter_state(model))==saved['tensor_sha256']
        write(cp/'RELOAD_AUDIT.json',{'status':'passed','fp32_reload_max_abs_logit_error':error,
            'masked_loss_error':saved['masked_loss_error'],'full_logit_error':saved['full_logit_error'],
            'tensor_sha256':saved['tensor_sha256']})
        print(json.dumps({'status':'smoke_passed','name':spec['name'],'error':error}));return
    if args.split=='test':assert (HERE/'SELECTION.json').exists() and spec['phase']=='confirmation'
    data=read(HERE/'tokens'/f"{spec['model']}_{'test_rotations' if args.split=='test' else 'validation'}.json")
    report=[];label_ids=torch.tensor(meta['label_ids'],device='cuda')
    with torch.inference_mode():
        for start in range(0,len(data),16):
            items=data[start:start+16];ids,mask,lengths=batch(items,meta)
            logits=forward_selected(model,ids,mask,lengths)
            scores=logits[:,label_ids];lp=scores.double().log_softmax(-1);prob=lp.exp()
            predicted=scores.argmax(-1);unrestricted=logits.argmax(-1)
            for j,item in enumerate(items):
                gold=item['gold'];pp=prob[j].cpu().numpy();pp[gold]-=1
                wrong=scores[j].clone();wrong[gold]=-torch.inf
                report.append({'id':item['id'],'rotation':item['rotation'],'gold':gold,'ambiguous_gold':item['ambiguous_gold'],'prediction':int(predicted[j]),
                    'correct':bool(predicted[j]==gold),'nll':float(-lp[j,gold]),'brier':float((pp**2).sum()),
                    'margin':float(scores[j,gold]-wrong.max()),'label_logits':scores[j].cpu().tolist(),
                    'unrestricted_token':int(unrestricted[j]),'unrestricted_correct':bool(unrestricted[j]==label_ids[gold]),
                    'valid_label':int(unrestricted[j]) in meta['label_ids']})
            if start%512==0:print(json.dumps({'eval':args.split,'rows':min(start+16,len(data)),'total':len(data)}),flush=True)
        # Numerical checks compare fixed-size replay and individually shaped
        # queries on validation only. Test uses the frozen inference protocol.
        if args.split=='validation':
            items=data[:8];ids,mask,lengths=batch(items,meta)
            grouped=forward_selected(model,ids,mask,lengths)[:,label_ids]
            replay=forward_selected(model,ids,mask,lengths)[:,label_ids]
            assert torch.equal(grouped,replay)
            errors=[]
            for j,item in enumerate(items):
                ids,mask,lengths=batch([item],meta)
                one=forward_selected(model,ids,mask,lengths)[0,label_ids]
                errors.append(float((one-grouped[j]).abs().max()))
            assert max(errors)<5e-4,errors
            write(cp/'NUMERICAL_AUDIT.json',{'status':'passed','exact_replay':True,'batch_shape_max_abs_logit_error':max(errors)})
    output=cp/f'{args.split}_predictions.jsonl'
    output.write_text(''.join(json.dumps(x)+'\n' for x in report))
    canonical=[x for x in report if x['rotation']==0 and not x['ambiguous_gold']]
    result={'spec':spec,'finished_at':now(),'primary':metrics(canonical),'prediction_sha256':sha(output),
            'checkpoint_sha256':sha(cp/'adapter.safetensors')}
    if args.split=='test':
        clean=[x for x in report if not x['ambiguous_gold']]
        result['full_1000_nominal_labels']=metrics([x for x in report if x['rotation']==0])
        result['rotation_average']=metrics(clean)
        groups={}
        for x in clean:groups.setdefault(x['id'],[]).append(x)
        assert len(groups)==998 and all(len(v)==4 for v in groups.values())
        result['all_four_correct']=100*float(np.mean([all(x['correct'] for x in v) for v in groups.values()]))
        result['per_rotation']={str(i):metrics([x for x in clean if x['rotation']==i]) for i in range(4)}
    write(cp/f'{args.split}_metrics.json',result)
    print(json.dumps({'status':'evaluated','name':spec['name'],'split':args.split,'primary':result['primary']}),flush=True)

if __name__=='__main__':main()
