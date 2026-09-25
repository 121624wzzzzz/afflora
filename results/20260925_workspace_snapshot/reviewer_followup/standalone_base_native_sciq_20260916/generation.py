import numpy as np
import torch
from transformers import AutoTokenizer
from settings import *
from modeling import batch

def setup(model,spec,meta):
    tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[spec['model']]['path'],local_files_only=True)
    eos=model.generation_config.eos_token_id
    stops=sorted(set((eos if isinstance(eos,list) else [eos])+[meta['eos_id']]))
    stops=[s for s in stops if s is not None]
    args=dict(max_new_tokens=MAX_NEW_TOKENS,do_sample=False,num_beams=1,use_cache=True,repetition_penalty=1.,
        eos_token_id=stops,pad_token_id=meta['pad_id'],temperature=None,top_p=None,top_k=None)
    return tok,stops,args

def decode_result(ids,tok,stops,gold):
    stop_index=next((i for i,t in enumerate(ids) if t in stops),None)
    content=ids if stop_index is None else ids[:stop_index]
    text=tok.decode(content,skip_special_tokens=False)
    normalized=text.strip();valid=normalized in ['A','B','C','D']
    return {'generated_ids':ids,'content_ids':content,'text':text,'terminated':stop_index is not None,
        'hit_length_cap':stop_index is None and len(ids)>=MAX_NEW_TOKENS,'valid_answer':valid,
        'strict_correct':valid and normalized=='ABCD'[gold]}

@torch.inference_mode()
def generate_rows(model,spec,data,meta):
    tok,stops,args=setup(model,spec,meta);result=[]
    for start in range(0,len(data),16):
        items=data[start:start+16];ids,mask,lengths=batch(items,meta,left=True)
        generated=model.generate(input_ids=ids,attention_mask=mask,**args)[:,ids.shape[1]:].cpu().tolist()
        for item,seq in zip(items,generated):
            # Remove padding after the first native stopping token, preserving EOS itself.
            stop=next((i for i,t in enumerate(seq) if t in stops),None)
            if stop is not None:seq=seq[:stop+1]
            result.append({k:item[k] for k in ['id','rotation','gold','ambiguous_gold']}|decode_result(seq,tok,stops,item['gold']))
        if start%256==0:print(json.dumps({'generation_rows':min(start+16,len(data)),'total':len(data)}),flush=True)
    return result

def summarize_generation(rr):
    return {'n':len(rr),'strict_accuracy':100*float(np.mean([r['strict_correct'] for r in rr])),
        'valid_answer_rate':100*float(np.mean([r['valid_answer'] for r in rr])),
        'terminated_rate':100*float(np.mean([r['terminated'] for r in rr])),
        'length_cap_rate':100*float(np.mean([r['hit_length_cap'] for r in rr]))}

@torch.inference_mode()
def smoke_generation(model,spec,items,meta):
    tok,stops,args=setup(model,spec,meta);ids,mask,lengths=batch(items,meta,left=True)
    pos=mask.long().cumsum(-1)-1;pos.masked_fill_(mask==0,1)
    manual=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits[:,-1,:].float()
    generated=model.generate(input_ids=ids,attention_mask=mask,return_dict_in_generate=True,output_scores=True,**args)
    score=generated.scores[0].float();error=float((manual-score).abs().max());assert error<5e-4,error
    assert torch.equal(score.argmax(-1),generated.sequences[:,ids.shape[1]])
    # Explicitly verify exact-answer parsing, including rejected explanations.
    assert decode_result([32,meta['eos_id']],tok,stops,0)['strict_correct']
    assert not decode_result(tok.encode('A because',add_special_tokens=False),tok,stops,0)['strict_correct']
    return {'first_step_logit_max_abs_error':error,'first_step_argmax_matches':True,'parser_exact_answer_checks':True}
