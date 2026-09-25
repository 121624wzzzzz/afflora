import argparse
import numpy as np
import torch
from settings import *
from modeling import load_checkpoint,adapter_state,digest,batch,forward_selected
from generation import generate_rows,summarize_generation,smoke_generation

def metrics(rr):
    return {'n':len(rr),'accuracy':100*float(np.mean([x['correct'] for x in rr])),
        'candidate_nll':float(np.mean([x['nll'] for x in rr])),'brier':float(np.mean([x['brier'] for x in rr])),
        'unrestricted_first_token_accuracy':100*float(np.mean([x['unrestricted_correct'] for x in rr])),
        'valid_label_rate':100*float(np.mean([x['valid_label'] for x in rr]))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['validation','test','smoke'],required=True);a=p.parse_args()
    cp=Path(a.checkpoint);model,spec,audit=load_checkpoint(cp);before=digest(adapter_state(model))
    meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
    if a.split=='smoke':
        saved=torch.load(cp/'SMOKE_FORWARD.pt',weights_only=True,map_location='cpu')
        with torch.inference_mode():got=forward_selected(model,saved['ids'].cuda(),saved['mask'].cuda(),saved['lengths'].cuda(),train=True).cpu()
        error=float((got-saved['logits']).abs().max());assert error<=1e-5,error
        assert before==saved['tensor_sha256']
        items=read(HERE/'tokens'/f"{spec['model']}_validation.json")[:8]
        check=smoke_generation(model,spec,items,meta)
        assert before==digest(adapter_state(model))
        write(cp/'RELOAD_AUDIT.json',{'status':'passed','fp32_reload_max_abs_logit_error':error,'full_logit_error':saved['full_logit_error'],
            'masked_loss_error':saved['masked_loss_error'],'generation':check})
        print(json.dumps({'status':'smoke_passed','name':spec['name']}),flush=True);return
    if a.split=='test':assert (HERE/'SELECTION.json').exists() and spec['phase'] in ['confirmation','reference']
    data=read(HERE/'tokens'/f"{spec['model']}_{'test_rotations' if a.split=='test' else 'validation'}.json")
    report=[];checks=[];label_ids=torch.tensor(meta['label_ids'],device='cuda')
    with torch.inference_mode():
        for start in range(0,len(data),16):
            items=data[start:start+16];ids,mask,lengths=batch(items,meta);logits=forward_selected(model,ids,mask,lengths)
            scores=logits[:,label_ids];lp=scores.double().log_softmax(-1);prob=lp.exp();pred=scores.argmax(-1);unrestricted=logits.argmax(-1)
            for j,item in enumerate(items):
                gold=item['gold'];pp=prob[j].cpu().numpy();pp[gold]-=1
                report.append({k:item[k] for k in ['id','rotation','gold','ambiguous_gold']}|{
                    'prediction':int(pred[j]),'correct':bool(pred[j]==gold),'nll':float(-lp[j,gold]),'brier':float((pp**2).sum()),
                    'label_logits':scores[j].cpu().tolist(),'unrestricted_token':int(unrestricted[j]),
                    'unrestricted_correct':bool(unrestricted[j]==label_ids[gold]),'valid_label':int(unrestricted[j]) in meta['label_ids']})
                top=scores[j].topk(2).values
                if float(top[0]-top[1])<=.001:
                    ii,mm,ll=batch([item],meta);one=forward_selected(model,ii,mm,ll)[0,label_ids]
                    diff=float((one-scores[j]).abs().max());assert diff<5e-4,diff
                    checks.append({'id':item['id'],'rotation':item['rotation'],'max_abs_error':diff,'original_prediction':int(pred[j]),
                        'single_prediction':int(one.argmax()),'prediction_flip':bool(pred[j]!=one.argmax())})
            if start%512==0:print(json.dumps({'eval':a.split,'rows':min(start+16,len(data)),'total':len(data)}),flush=True)
        items=data[:8];ids,mask,lengths=batch(items,meta)
        one=forward_selected(model,ids,mask,lengths)[:,label_ids];two=forward_selected(model,ids,mask,lengths)[:,label_ids]
        assert torch.equal(one,two)
    predpath=cp/f'{a.split}_predictions.jsonl';predpath.write_text(''.join(json.dumps(x)+'\n' for x in report))
    canonical=[x for x in report if x['rotation']==0 and not x['ambiguous_gold']]
    result={'spec':spec,'at':now(),'primary':metrics(canonical),'prediction_sha256':sha(predpath),
        'checkpoint_sha256':sha(cp/'adapter.safetensors') if spec['arm']!='base' else None,
        'near_tie_rechecks':checks,'numerical_replay_exact':True}
    if a.split=='test':
        clean=[x for x in report if not x['ambiguous_gold']];result['rotation_average']=metrics(clean)
        result['per_rotation']={str(i):metrics([x for x in clean if x['rotation']==i]) for i in range(4)}
        gen=generate_rows(model,spec,[x for x in data if x['rotation']==0],meta)
        gp=cp/'test_generation.jsonl';gp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gen))
        result['generation']=summarize_generation([x for x in gen if not x['ambiguous_gold']]);result['generation_sha256']=sha(gp)
        lookup={x['id']:x['unrestricted_token'] for x in report if x['rotation']==0}
        result['generation_first_token_disagreements_with_candidate_pass']=[x['id'] for x in gen if x['generated_ids'][0]!=lookup[x['id']]]
    assert before==digest(adapter_state(model));result['adapter_tensors_unchanged']=True
    write(cp/f'{a.split}_metrics.json',result)
    print(json.dumps({'status':'evaluated','name':spec['name'],'split':a.split,'primary':result['primary'],'generation':result.get('generation')}),flush=True)

if __name__=='__main__':main()
