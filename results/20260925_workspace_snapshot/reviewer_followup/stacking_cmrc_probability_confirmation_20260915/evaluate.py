import argparse
import json
import math
import os
from pathlib import Path
import sys
from common import HERE,CHAT,read,rows,sha,write,now,checkpoint_hashes
from experiment import jobs,variant
sys.path[:0]=[str(HERE/'source/corrected_sft_experiment'),str(CHAT/'ifeval_deps')]
os.environ['NLTK_DATA']=str(CHAT/'nltk_data')
import evaluate_corrected_sft as legacy
import cmrc_official_py3 as official
import torch
import torch.nn.functional as F
from transformers import set_seed

def configure():
    torch.set_num_threads(4);set_seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def load(cp):
    model,tokenizer,_,_=legacy.load_model(argparse.Namespace(run_dir=str(cp),model_path=None,affine_ablation='none',device='cuda'))
    model.float();model.get_base_model().set_attn_implementation('eager');model.eval()
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point())
    assert model.get_base_model().config._attn_implementation=='eager'
    return model,tokenizer

@torch.inference_mode()
def score(model,tokenizer,row,answer,copies=1,padding=0,full=False,fp64=False):
    tokens=row['prompt_ids']+answer['answer_ids'];start=len(row['prompt_ids']);end=len(tokens)
    ids=torch.full((copies,end+padding),tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
    ids[:,:end]=torch.tensor(tokens,device='cuda');mask[:,:end]=1
    positions=torch.arange(start-1,end,device='cuda')
    output=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=0 if full else positions).logits
    if full:output=output[:,positions]
    targets=torch.tensor(answer['answer_ids']+[row['eos']],device='cuda')
    result=[]
    for logits in output:
        selected=logits.float().contiguous();loss=F.cross_entropy(selected,targets,reduction='none')
        error=0.
        if fp64:
            error=(loss.double()-F.cross_entropy(selected.double(),targets,reduction='none')).abs().max().item();assert error<5e-6,error
        hit=selected.argmax(-1).eq(targets)
        content=loss[:-1];nll=content.double().sum().item();eos=loss[-1].item()
        r={'id':row['id'],'answer':answer['answer'],'reference_indices':answer['reference_indices'],'tokens':len(content),
           'content_nll':nll,'eos_nll':eos,'probability':math.exp(-nll-eos),'content_top1':int(hit[:-1].sum()),'eos_top1':bool(hit[-1]),
           'first_nll':content[0].item(),'first_top1':bool(hit[0]),'rest_nll':content[1:].double().sum().item(),
           'rest_top1':int(hit[1:-1].sum()),'fp64_max_error':error,'token_losses':loss.tolist()}
        result.append(r)
    return result

def numerical_audit(model,tokenizer,data):
    checks=[]
    for index,row in enumerate(data[:8]):
        answer=row['unique_answers'][0]
        single=score(model,tokenizer,row,answer,fp64=True)[0]
        repeat=score(model,tokenizer,row,answer,fp64=True)[0]
        full=score(model,tokenizer,row,answer,full=True)[0] if index==0 else None
        alternatives=score(model,tokenizer,row,answer,copies=2,padding=17)
        max_loss=max(max(abs(a-b) for a,b in zip(single['token_losses'],other['token_losses'])) for other in alternatives)
        max_prob=max(abs(single['probability']-other['probability']) for other in alternatives)
        checks.append({'id':row['id'],'canonical_replay_exact':single==repeat,
            'token_loss_max_diff':max_loss,'probability_max_diff':max_prob,
            'fp64_max_diff':single['fp64_max_error'],'full_logits_compared':index==0,
            'full_logits_loss_max_diff':max(abs(a-b) for a,b in zip(single['token_losses'],full['token_losses'])) if full else 0.,
            'full_logits_probability_diff':abs(single['probability']-full['probability']) if full else 0.,
            'top1_count_differences':[other['content_top1']-single['content_top1'] for other in alternatives]})
    passed=all(r['canonical_replay_exact'] and r['token_loss_max_diff']<=5e-4 and r['probability_max_diff']<=1e-4
        and r['full_logits_loss_max_diff']<=2e-4 and r['full_logits_probability_diff']<=1e-5 for r in checks)
    return {'status':'passed' if passed else 'failed','checked_at':now(),'checks':checks,
        'shape_tolerance':{'token_nll':5e-4,'probability':1e-4},'canonical_repeat':'bitwise equal',
        'precision':'FP32 eager, TF32 disabled; canonical repeat, full-logits and duplicated padded-batch comparisons'}

def probabilities(model,tokenizer,data,path):
    result=[]
    with path.open('x') as f:
        for i,row in enumerate(data):
            # Exactly one evaluation for each unique prompt/answer pair.
            values=[score(model,tokenizer,row,answer)[0] for answer in row['unique_answers']]
            for value in values:value.pop('token_losses')
            logp=torch.tensor([-r['content_nll']-r['eos_nll'] for r in values],dtype=torch.float64)
            set_nll=-torch.logsumexp(logp,dim=0).item();mass=math.exp(-set_nll)
            assert 0<mass<=1+1e-6,(row['id'],mass)
            references=[None]*len(row['references'])
            for value in values:
                for j in value['reference_indices']:assert references[j] is None;references[j]=value
            assert all(r is not None for r in references)
            total_tokens=sum(r['tokens'] for r in references);total_nll=sum(r['content_nll'] for r in references)
            item={'id':row['id'],'cluster':row['cluster'],'answer_set_probability':mass,'answer_set_nll':set_nll,
                'content_tokens':total_tokens,'content_nll':total_nll,'content_macro_ce':sum(r['content_nll']/r['tokens'] for r in references)/len(references),
                'reference_count':len(references),'eos_nll':sum(r['eos_nll'] for r in references),'content_top1':sum(r['content_top1'] for r in references),
                'first_nll':sum(r['first_nll'] for r in references),'first_top1':sum(r['first_top1'] for r in references),
                'rest_nll':sum(r['rest_nll'] for r in references),'rest_top1':sum(r['rest_top1'] for r in references),'unique_answers':values}
            result.append(item);f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
            if (i+1)%128==0 or i+1==len(data):print(path.stem,i+1,'/',len(data),flush=True)
    return result

def batch_prompts(data):
    current=[];width=0
    for row in data:
        n=len(row['prompt_ids'])
        if current and (len(current)>=16 or max(width,n)*(len(current)+1)>8192):yield current;current=[];width=0
        current.append(row);width=max(width,n)
    if current:yield current

@torch.inference_mode()
def generation(model,tokenizer,data,path):
    eos=model.generation_config.eos_token_id
    eos=sorted(set((eos if isinstance(eos,list) else [eos])+[data[0]['eos']]))
    result=[]
    with path.open('x') as f:
        for batch in batch_prompts(data):
            width=max(len(r['prompt_ids']) for r in batch)
            ids=torch.full((len(batch),width),tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
            for i,r in enumerate(batch):ids[i,-len(r['prompt_ids']):]=torch.tensor(r['prompt_ids'],device='cuda');mask[i,-len(r['prompt_ids']):]=1
            generated=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=256,repetition_penalty=1.,
                pad_token_id=tokenizer.pad_token_id,eos_token_id=eos,use_cache=True)[:,width:]
            for row,tokens in zip(batch,generated.tolist()):
                stop=next((i for i,t in enumerate(tokens) if t in eos),None);n=stop+1 if stop is not None else len(tokens)
                text=tokenizer.decode(tokens[:n],skip_special_tokens=True)
                em=official.calc_em_score(row['references'],text);f1=official.calc_f1_score(row['references'],text)
                r={'id':row['id'],'cluster':row['cluster'],'response':text,'token_ids':tokens[:n],'generated_tokens':n,
                    'terminal_token':tokens[stop] if stop is not None else None,'hit_token_cap':stop is None,'em':em,'f1':f1,'avg':(em+f1)/2}
                result.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush();print('generation',len(result),'/',len(data),flush=True)
    return result

def summary(prob,gen):
    n=len(prob);tokens=sum(r['content_tokens'] for r in prob);refs=sum(r['reference_count'] for r in prob)
    metrics={k:sum(r[k] for r in prob)/n for k in ['answer_set_probability','answer_set_nll','content_macro_ce']}
    metrics.update(content_micro_ce=sum(r['content_nll'] for r in prob)/tokens,
        total_micro_ce=sum(r['content_nll']+r['eos_nll'] for r in prob)/(tokens+refs),eos_ce=sum(r['eos_nll'] for r in prob)/refs,
        content_top1=sum(r['content_top1'] for r in prob)/tokens,first_ce=sum(r['first_nll'] for r in prob)/refs,
        first_top1=sum(r['first_top1'] for r in prob)/refs,rest_ce=sum(r['rest_nll'] for r in prob)/(tokens-refs),
        rest_top1=sum(r['rest_top1'] for r in prob)/(tokens-refs))
    if gen:
        assert [r['id'] for r in prob]==[r['id'] for r in gen]
        metrics.update({k:sum(r[k] for r in gen)/n for k in ['em','f1','avg','generated_tokens','hit_token_cap']})
    return {'n':n,'metrics':metrics}

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    configure();j=next(j for j in jobs(a.smoke) if j['name']==a.name);cp=Path(j['checkpoint'])
    hashes=checkpoint_hashes(cp);assert hashes==read(cp/'TRAIN_COMPLETE.json')['checkpoint_hashes']
    if not a.smoke:
        for path,h in read(HERE/'manifest.json')['local_sha256'].items():assert sha(path)==h,path
    out=HERE/('smoke_outputs' if a.smoke else 'outputs')/j['name'];out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists()
    model,tokenizer=load(cp)
    internal=rows(HERE/'token_cache'/j['model']/'internal_dev.jsonl');public=rows(HERE/'token_cache'/j['model']/'public_dev.jsonl')
    numeric=numerical_audit(model,tokenizer,internal);write(out/'NUMERICAL_AUDIT.json',numeric);assert numeric['status']=='passed',numeric
    if a.smoke:internal=internal[:8];public=internal
    dev=probabilities(model,tokenizer,internal,out/'internal_probability.jsonl')
    prob=probabilities(model,tokenizer,public,out/'public_probability.jsonl')
    gen=generation(model,tokenizer,public,out/'generation.jsonl')
    write(out/'scores.json',{'job':j,'public':summary(prob,gen),'internal':summary(dev,None),'precision':'FP32 eager TF32 disabled'})
    assert checkpoint_hashes(cp)==hashes
    write(out/'COMPLETE.json',{'status':'passed','job':j,'completed_at':now(),'checkpoint_hashes':hashes,
        'manifest_sha256':sha(HERE/'manifest.json') if not a.smoke else None,
        'files':{str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}})
    print('COMPLETE',j['name'],summary(prob,gen),flush=True)

if __name__=='__main__':main()
