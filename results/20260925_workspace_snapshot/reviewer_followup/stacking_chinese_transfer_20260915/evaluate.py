"""Checkpoint replay, answer-only likelihood, and Chinese answer generation."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common import HERE, CHAT, checkpoint_hashes, now, prompt_ids, read, rows, sha, write
sys.path[:0]=[str(CHAT/'source/corrected_sft_experiment'),str(CHAT/'ifeval_deps')]
os.environ['NLTK_DATA']=str(CHAT/'nltk_data')
import torch
import torch.nn.functional as F
from transformers import set_seed
import evaluate_corrected_sft as legacy
import cmrc_official_py3 as official

@torch.inference_mode()
def score_batch(model,tokenizer,items,numerical_check=False):
    width=max(len(x['ids']) for x in items)
    ids=torch.full((len(items),width),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
    attention=torch.zeros_like(ids)
    for i,x in enumerate(items):
        ids[i,:len(x['ids'])]=torch.tensor(x['ids'],device='cuda');attention[i,:len(x['ids'])]=1
    logits=model(input_ids=ids,attention_mask=attention,use_cache=False).logits
    result=[]
    for i,x in enumerate(items):
        start=x['start'];end=len(x['ids'])
        targets=ids[i,start:end]
        selected=logits[i,start-1:end-1,:].float().contiguous()
        losses=F.cross_entropy(selected,targets,reduction='none')
        if numerical_check:
            reference=F.cross_entropy(selected.double(),targets,reduction='none')
            assert (losses.double()-reference).abs().max().item()<5e-6
            manual=torch.logsumexp(selected.double(),dim=-1)-selected.double().gather(1,targets[:,None]).squeeze(1)
            assert torch.allclose(reference,manual,atol=1e-10,rtol=0)
        result.append({'nll':losses.double().sum().item(),'tokens':len(targets),
            'top1_correct':int((selected.argmax(-1)==targets).sum().item())})
    return result

def batches(items,max_rows,token_budget):
    batch=[];width=0
    for item in items:
        n=len(item['ids'])
        if batch and (len(batch)>=max_rows or max(width,n)*(len(batch)+1)>token_budget):
            yield batch;batch=[];width=0
        batch.append(item);width=max(width,n)
    if batch:yield batch

def scoring_items(tokenizer,task,data):
    items=[]
    for row in data:
        prefix=prompt_ids(tokenizer,task,row)
        answers=row['answers'] if task=='cmrc' else row['choices']
        for index,answer in enumerate(answers):
            tokens=tokenizer.encode(answer,add_special_tokens=False)
            assert tokens and tokenizer.decode(tokens)==answer,(row['id'],index)
            assert len(prefix)+len(tokens)<=8192,(row['id'],'context_limit')
            items.append({'row_id':row['id'],'index':index,'ids':prefix+tokens,'start':len(prefix)})
    return items

def resume_lines(path):
    if not path.exists():return []
    raw=path.read_bytes()
    assert not raw or raw.endswith(b'\n'),f'Incomplete output line; preserve and repair explicitly: {path}'
    return rows(path)

def likelihood(model,tokenizer,task,data,out,smoke):
    path=out/f'{task}_likelihood.jsonl';saved=resume_lines(path)
    items=scoring_items(tokenizer,task,data)
    assert len(saved)<=len(items)
    for old,x in zip(saved,items):assert (old['row_id'],old['index'])==(x['row_id'],x['index'])
    with path.open('a') as f:
        for batch in batches(items[len(saved):],16,4096):
            got=score_batch(model,tokenizer,batch,numerical_check=smoke)
            for x,v in zip(batch,got):
                result={'row_id':x['row_id'],'index':x['index'],**v}
                saved.append(result);f.write(json.dumps(result)+'\n')
            f.flush()
            if smoke or len(saved)%128<len(batch):print(task,'likelihood',len(saved),'/',len(items),flush=True)
    assert len(saved)==len(items)
    return saved

@torch.inference_mode()
def generation(model,tokenizer,data,out,eos,smoke):
    path=out/'cmrc_generation.jsonl';saved=resume_lines(path)
    assert len(saved)<=len(data)
    for a,b in zip(saved,data):assert a['id']==b['id']
    items=[{'ids':prompt_ids(tokenizer,'cmrc',r),'row':r} for r in data[len(saved):]]
    assert all(len(x['ids'])+256<=8192 for x in items),'No silent context truncation'
    with path.open('a') as f:
        for batch in batches(items,16,8192):
            width=max(len(x['ids']) for x in batch)
            ids=torch.full((len(batch),width),tokenizer.pad_token_id,dtype=torch.long,device='cuda');attention=torch.zeros_like(ids)
            for i,x in enumerate(batch):
                ids[i,-len(x['ids']):]=torch.tensor(x['ids'],device='cuda');attention[i,-len(x['ids']):]=1
            generated=model.generate(input_ids=ids,attention_mask=attention,do_sample=False,
                max_new_tokens=256,repetition_penalty=1.,pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos,use_cache=True)[:,width:]
            for x,tokens in zip(batch,generated.tolist()):
                stop=next((i for i,t in enumerate(tokens) if t in eos),None)
                count=stop+1 if stop is not None else len(tokens)
                text=tokenizer.decode(tokens[:count],skip_special_tokens=True)
                row=x['row'];em=official.calc_em_score(row['answers'],text);f1=official.calc_f1_score(row['answers'],text)
                result={'id':row['id'],'cluster':row['cluster'],'response':text,'token_ids':tokens[:count],
                    'generated_tokens':count,'terminal_token':tokens[stop] if stop is not None else None,
                    'hit_token_cap':stop is None,'em':em,'f1':f1,'avg':(em+f1)/2}
                saved.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n')
            f.flush();print('cmrc generation',len(saved),'/',len(data),flush=True)
    return saved

def task_summary(task,data,likelihoods,generated=None):
    by={r['id']:[] for r in data}
    for x in likelihoods:by[x['row_id']].append(x)
    details=[]
    for row in data:
        values=by[row['id']];assert [x['index'] for x in values]==list(range(len(row['answers'] if task=='cmrc' else row['choices'])))
        d={'id':row['id'],'cluster':row['cluster']}
        if task=='c3':
            mean_scores=[-x['nll']/x['tokens'] for x in values];total_scores=[-x['nll'] for x in values]
            pred=max(range(len(values)),key=lambda i:mean_scores[i]);pred_total=max(range(len(values)),key=lambda i:total_scores[i])
            gold=row['gold'];wrong=[v for i,v in enumerate(mean_scores) if i not in row['gold_indices']]
            d.update(subset=row['subset'],accuracy=pred in row['gold_indices'],accuracy_sum=pred_total in row['gold_indices'],
                prediction=pred,gold=gold,margin=mean_scores[gold]-max(wrong),gold_ce=-mean_scores[gold],
                gold_top1=values[gold]['top1_correct']/values[gold]['tokens'])
        else:
            # Equal weighting of references within each question, then questions.
            d.update(answer_ce=sum(v['nll']/v['tokens'] for v in values)/len(values),
                answer_top1=sum(v['top1_correct']/v['tokens'] for v in values)/len(values))
        details.append(d)
    if task=='cmrc':
        assert [x['id'] for x in generated]==[x['id'] for x in details]
        for d,g in zip(details,generated):d.update({k:g[k] for k in ['em','f1','avg','hit_token_cap','generated_tokens']})
        metrics=['em','f1','avg','answer_ce','answer_top1','hit_token_cap','generated_tokens']
    else:metrics=['accuracy','accuracy_sum','margin','gold_ce','gold_top1']
    result={'n':len(details),'metrics':{k:sum(x[k] for x in details)/len(details) for k in metrics},'per_example':details}
    if task=='c3':result['subsets']={s:{k:sum(x[k] for x in details if x['subset']==s)/sum(x['subset']==s for x in details) for k in metrics} for s in ['d','m'] if any(x['subset']==s for x in details)}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);set_seed(0)
    audit=read(HERE/'REUSE_AUDIT.json');j=next(x for x in audit['endpoints'] if x['name']==a.name)
    assert checkpoint_hashes(j['checkpoint'])==j['checkpoint_hashes']
    if not a.smoke:
        manifest=read(HERE/'manifest.json')
        for path,h in manifest['local_sha256'].items():assert sha(path)==h,path
    out=HERE/('smoke' if a.smoke else 'outputs')/a.name;out.mkdir(parents=True,exist_ok=True)
    identity={'job':j,'smoke':a.smoke,'protocol_sha256':sha(HERE/'DESIGN.md'),
        'manifest_sha256':sha(HERE/'manifest.json') if not a.smoke else None,
        'implementation_sha256':{str(HERE/n):sha(HERE/n) for n in ['evaluate.py','common.py','cmrc_official_py3.py']}}
    meta=out/'identity.json'
    if meta.exists():assert read(meta)==identity
    else:write(meta,identity)
    model,tokenizer,model_path,run_args=legacy.load_model(argparse.Namespace(run_dir=j['checkpoint'],model_path=None,affine_ablation='none',device='cuda'))
    old=read(CHAT/f"reports/{a.name}.test.json")['per_example'][:8]
    replay=legacy.evaluate(model,tokenizer,rows(CHAT/'data/test.jsonl')[:8],1,1024,torch.device('cuda'))['per_example']
    assert [(x['record_id'],x['token_count']) for x in old]==[(x['record_id'],x['token_count']) for x in replay]
    error=max(abs(x['mean_ce']-y['mean_ce']) for x,y in zip(old,replay));assert error<=1e-8,error
    write(out/'REPLAY_AUDIT.json',{'status':'passed','examples':8,'max_abs_ce_difference':error,'checkpoint_hashes':checkpoint_hashes(j['checkpoint'])})
    print('Checkpoint CE replay passed',error,flush=True)
    assert official.calc_em_score(['北京'],'北京。')==1
    assert official.calc_f1_score(['北京'],'北')==2/3
    assert official.calc_em_score(['北京'],'上海')==0
    config=read(Path(model_path)/'generation_config.json');eos=config.get('eos_token_id',tokenizer.eos_token_id)
    eos=sorted(set((eos if isinstance(eos,list) else [eos])+tokenizer.encode('<|im_end|>',add_special_tokens=False)))
    for task in ['c3','cmrc']:
        data=rows(HERE/'data'/f"{task}_{'train' if a.smoke else 'eval'}.jsonl")
        if a.smoke:data=(data[:8] if task=='cmrc' else [r for r in data if r['subset']=='d'][:4]+[r for r in data if r['subset']=='m'][:4])
        got=likelihood(model,tokenizer,task,data,out,a.smoke)
        generated=generation(model,tokenizer,data,out,eos,a.smoke) if task=='cmrc' else None
        report=task_summary(task,data,got,generated)
        report['source_sha256']={str(out/f'{task}_likelihood.jsonl'):sha(out/f'{task}_likelihood.jsonl')}
        if generated:report['source_sha256'][str(out/'cmrc_generation.jsonl')]=sha(out/'cmrc_generation.jsonl')
        write(out/f'{task}_scores.json',report)
        print(task,report['metrics'],flush=True)
    assert checkpoint_hashes(j['checkpoint'])==j['checkpoint_hashes']
    write(out/'COMPLETE.json',{'status':'passed','completed_at':now(),'identity':identity,
        'files':{str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}})

if __name__=='__main__':main()
