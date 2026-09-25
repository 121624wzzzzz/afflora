import argparse
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from settings import *
from modeling import load_checkpoint, adapter_state, digest, frozen_digest, batch, forward_selected
from scoring import decode_output


def summarize(records, mapping):
    clean = [r for r in records if not r['ambiguous_gold']]
    return {'n': len(clean)} | {name: float(100 * np.mean([r[field] for r in clean])) for name, field in mapping.items()}


def main():
    assert not (HERE/'ARTIFACT_MANIFEST.json').exists(),'Sealed run: create a new output directory.'
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',choices=['validation','test','smoke'],required=True)
    args=parser.parse_args();cp=Path(args.checkpoint);model,spec,audit=load_checkpoint(cp)
    # All evaluation artifacts live in the repair directory, including reused runs.
    out=HERE/'evaluations'/spec['name'];out.mkdir(parents=True,exist_ok=True)
    modelcfg=read(HERE/'models.json')[spec['model']]
    tok=AutoTokenizer.from_pretrained(modelcfg['path'],local_files_only=True)
    meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
    before=digest(adapter_state(model));frozen_before=frozen_digest(model)
    if args.split=='smoke':
        saved=torch.load(cp/'SMOKE_FORWARD.pt',weights_only=True,map_location='cpu')
        with torch.inference_mode():
            got=forward_selected(model,saved['ids'].cuda(),saved['mask'].cuda(),saved['lengths'].cuda(),train=True).cpu()
            reload_error=float((got-saved['logits']).abs().max());assert reload_error<=1e-5
            items=read(HERE/'tokens'/f"{spec['model']}_validation.json")[:8]
            seq=[r['input_ids']+[meta['label_ids'][r['gold']],meta['eos_id']] for r in items]
            width=max(map(len,seq));ids=torch.tensor([s+[meta['pad_id']]*(width-len(s)) for s in seq],device='cuda')
            mask=torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seq],device='cuda')
            labels=torch.full_like(ids,-100)
            for i,s in enumerate(seq):labels[i,len(s)-2:len(s)]=ids[i,len(s)-2:len(s)]
            full=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False)
            ii,mm,ll=batch(items,meta,train=True);selected=forward_selected(model,ii,mm,ll,train=True)
            gold=torch.tensor([[meta['label_ids'][r['gold']],meta['eos_id']] for r in items],device='cuda')
            manual=F.cross_entropy(selected.reshape(-1,selected.shape[-1]),gold.reshape(-1))
            target_positions=torch.tensor([[len(s)-3,len(s)-2] for s in seq],device='cuda')
            reference=full.logits[torch.arange(len(seq),device='cuda')[:,None],target_positions].float()
            target_logit_error=float((selected-reference).abs().max());assert target_logit_error<5e-4
            reference_loss=F.cross_entropy(reference.reshape(-1,reference.shape[-1]),gold.reshape(-1))
            standard_mask_error=float(abs(reference_loss-full.loss));assert standard_mask_error<1e-5,standard_mask_error
            loss_error=float(abs(manual-full.loss));assert loss_error<=2*target_logit_error+2e-6,(loss_error,target_logit_error)
            ii,mm,ll=batch(items,meta,left=True);pos=(mm.cumsum(-1)-1).masked_fill(mm==0,1)
            standard=model(input_ids=ii,attention_mask=mm,position_ids=pos,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            ii,mm,ll=batch(items,meta);manual_logits=forward_selected(model,ii,mm,ll)
            forward_error=float((standard-manual_logits).abs().max());assert forward_error<5e-4,forward_error
        assert before==digest(adapter_state(model)) and frozen_before==frozen_digest(model)
        write(out/'smoke.json',{'status':'passed','reload_max_logit_error':reload_error,'standard_shifted_loss_error':loss_error,
                              'independent_full_forward_error':forward_error,'trainable_parameters':audit['trainable_parameters'],
                              'standard_mask_error':standard_mask_error,'target_logit_error':target_logit_error,
                              'exact_training_mask_identity_error':saved['exact_mask_identity_error']})
        return
    data=read(HERE/'tokens'/f"{spec['model']}_{args.split}.json")
    raw=rows(HERE/'data'/f'{args.split}.jsonl');assert [r['id'] for r in raw]==[r['id'] for r in data]
    predictions=[];label_ids=torch.tensor(meta['label_ids'],device='cuda')
    with torch.inference_mode():
        for start in range(0,len(data),16):
            items=data[start:start+16];ids,mask,lengths=batch(items,meta);logits=forward_selected(model,ids,mask,lengths)
            scores=logits[:,label_ids];lp=scores.double().log_softmax(-1);gold=torch.tensor([r['gold'] for r in items],device='cuda')
            # Decompose full-vocabulary response-letter and EOS teacher-forced loss.
            ii,mm,ll=batch(items,meta,train=True);two=forward_selected(model,ii,mm,ll,train=True)
            targets=torch.stack([label_ids[gold],torch.full_like(gold,meta['eos_id'])],1)
            loss=F.cross_entropy(two.reshape(-1,two.shape[-1]),targets.reshape(-1),reduction='none').reshape(-1,2)
            for j,item in enumerate(items):
                pred=int(scores[j].argmax());token=int(logits[j].argmax())
                predictions.append({k:item[k] for k in ['id','gold','ambiguous_gold']}|{'prediction':pred,'correct':pred==item['gold'],
                    'label_logits':scores[j].cpu().tolist(),'candidate_nll':float(-lp[j,item['gold']]),
                    'unrestricted_token':token,'unrestricted_correct':token==meta['label_ids'][item['gold']],
                    'valid_label':token in meta['label_ids'],'answer_ce':float(loss[j,0]),'eos_ce':float(loss[j,1])})
    pp=out/f'{args.split}_predictions.jsonl';pp.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    metrics=summarize(predictions,{'candidate_accuracy':'correct','unrestricted_first_token_accuracy':'unrestricted_correct','valid_label_rate':'valid_label'})
    clean=[r for r in predictions if not r['ambiguous_gold']]
    metrics.update({k:float(np.mean([r[k] for r in clean])) for k in ['candidate_nll','answer_ce','eos_ce']})
    result={'spec':spec,'split':args.split,'primary':metrics,'parameters':audit['trainable_parameters'],
            'prediction_sha256':sha(pp),'input_tokens_sha256':sha(HERE/'tokens'/f"{spec['model']}_{args.split}.json")}
    previous=cp/f'{args.split}_predictions.jsonl'
    if previous.exists():
        old=[r for r in rows(previous) if r.get('rotation',0)==0]
        assert [r['id'] for r in old]==[r['id'] for r in predictions]
        error=max(abs(a-b) for p,q in zip(predictions,old) for a,b in zip(p['label_logits'],q['label_logits']))
        flips=sum(p['prediction']!=q['prediction'] for p,q in zip(predictions,old))
        assert error<5e-4 and flips==0,(error,flips)
        result['original_candidate_replay']={'max_logit_error':error,'prediction_flips':flips,'source_sha256':sha(previous)}
    if args.split=='test':
        native=model.generation_config.eos_token_id
        stops=sorted(set((native if isinstance(native,list) else [native])+[meta['eos_id']]))
        stops=[x for x in stops if x is not None]
        genargs=dict(do_sample=False,num_beams=1,max_new_tokens=MAX_NEW_TOKENS,use_cache=True,repetition_penalty=1.,
                     eos_token_id=stops,pad_token_id=meta['pad_id'],temperature=None,top_p=None,top_k=None)
        generated=[];lookup={r['id']:r for r in predictions};first_token_disagreements=[]
        with torch.inference_mode():
            for start in range(0,len(data),16):
                items=data[start:start+16];ids,mask,lengths=batch(items,meta,left=True)
                output=model.generate(input_ids=ids,attention_mask=mask,**genargs)[:,ids.shape[1]:].cpu().tolist()
                for item,source,seq in zip(items,raw[start:start+16],output):
                    record={k:item[k] for k in ['id','gold','ambiguous_gold']}|decode_output(seq,tok,stops,source['choices'],item['gold'],MAX_NEW_TOKENS)
                    if seq[0]!=lookup[item['id']]['unrestricted_token']:first_token_disagreements.append(item['id'])
                    generated.append(record)
                if start%256==0:print(json.dumps({'name':spec['name'],'generated':min(start+16,len(data)),'total':len(data)}),flush=True)
        assert not first_token_disagreements,first_token_disagreements
        gp=out/'test_generation.jsonl';gp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in generated))
        result['generation']=summarize(generated,{'answer_accuracy':'correct','valid_answer_rate':'valid_answer',
            'exact_letter_format_rate':'exact_letter_format','strict_accuracy':'strict_correct','terminated_rate':'terminated',
            'length_cap_rate':'hit_length_cap','completed_answer_accuracy':'completed_correct'})
        clean_gen=[r for r in generated if not r['ambiguous_gold']]
        result['generation'].update(mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in clean_gen])),
                                    extraction_routes=dict(Counter(r['route'] for r in clean_gen)))
        result.update(generation_sha256=sha(gp),generation_settings=genargs,first_token_disagreements=first_token_disagreements)
    assert before==digest(adapter_state(model));assert frozen_before==frozen_digest(model)
    result.update(adapter_tensors_unchanged=True,original_weights_unchanged_during_eval=True)
    write(out/f'{args.split}_metrics.json',result)
    print(json.dumps({'name':spec['name'],'primary':metrics,'generation':result.get('generation')}),flush=True)


if __name__=='__main__':main()
