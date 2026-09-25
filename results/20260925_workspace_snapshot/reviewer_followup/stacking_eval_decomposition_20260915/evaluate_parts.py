import argparse
import math
import sys
from pathlib import Path
from shared import D,B,read,rows,sha,write,now,jobs
sys.path.insert(0,str(B))
import evaluate as previous
import torch
import torch.nn.functional as F

def items_for(tokenizer,split):
    if split=='public_dev':
        data=rows(B/'data/cmrc_eval.jsonl')
        items=previous.scoring_items(tokenizer,'cmrc',data)
        for x in items:x['end']=len(x['ids'])
        return items
    items=[]
    for r in rows(B/'data/dev.jsonl'):
        x=previous.legacy.tokenize_conversation(r,tokenizer,2048)
        ix=[i for i,v in enumerate(x['labels']) if v!=-100]
        assert ix==list(range(ix[0],ix[-1]+1))
        items.append({'row_id':r['record_id'],'index':0,'ids':x['input_ids'],'start':ix[0],'end':ix[-1]})
    return items

@torch.inference_mode()
def score(model,tokenizer,items,split,check):
    width=max(len(x['ids']) for x in items)
    ids=torch.full((len(items),width),tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
    for i,x in enumerate(items):
        ids[i,:len(x['ids'])]=torch.tensor(x['ids'],device='cuda');mask[i,:len(x['ids'])]=1
    logits=model(input_ids=ids,attention_mask=mask,use_cache=(split=='internal_dev')).logits
    eos=tokenizer.encode('<|im_end|>',add_special_tokens=False)[0]
    result=[]
    for i,x in enumerate(items):
        start,end=x['start'],x['end'];targets=ids[i,start:end]
        selected=logits[i,start-1:end-1].float().contiguous()
        losses=F.cross_entropy(selected,targets,reduction='none')
        e=logits[i,end-1].float().contiguous().unsqueeze(0);et=torch.tensor([eos],device='cuda')
        eloss=F.cross_entropy(e,et,reduction='none')
        error=0.
        if check:
            error=max((losses.double()-F.cross_entropy(selected.double(),targets,reduction='none')).abs().max().item(),
                      (eloss.double()-F.cross_entropy(e.double(),et,reduction='none')).abs().item())
            assert error<5e-6,error
        gold=selected.gather(1,targets[:,None]).squeeze(1)
        pred=selected.argmax(-1);hit=pred.eq(targets)
        vocab=torch.arange(selected.shape[-1],device='cuda')
        rank=1+(selected>gold[:,None]).sum(-1)+((selected==gold[:,None])&(vocab[None,:]<targets[:,None])).sum(-1)
        hit5=rank<=5
        top2=selected.topk(2,dim=-1)
        rival=torch.where(top2.indices[:,0].eq(targets),top2.values[:,1],top2.values[:,0])
        margin=gold-rival
        ep=e.argmax(-1).eq(et).item()
        v={'row_id':x['row_id'],'index':x['index'],'tokens':len(targets),'nll':losses.double().sum().item(),
           'top1_correct':int(hit.sum()),'top5_correct':int(hit5.sum()),'margin_sum':margin.double().sum().item(),
           'first_nll':losses[0].item(),'first_top1':bool(hit[0]),'first_top5':bool(hit5[0]),'first_margin':margin[0].item(),
           'rest_nll':losses[1:].double().sum().item(),'rest_top1_correct':int(hit[1:].sum()),
           'eos_nll':eloss.item(),'eos_top1':bool(ep),'teacher_exact':bool(hit.all()) and bool(ep),'fp64_check_max':error,
           'argmax_vs_topk1_different':int(pred.ne(top2.indices[:,0]).sum()),'gold_tied_best':int(margin.eq(0).sum())}
        assert v['tokens']>0 and all(math.isfinite(v[k]) for k in ['nll','eos_nll','margin_sum'])
        result.append(v)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);previous.set_seed(0)
    j=next(j for j in jobs() if j['name']==a.name)
    cp=Path(j['checkpoint']);hashes=previous.checkpoint_hashes(cp)
    assert hashes==read(cp/'TRAIN_COMPLETE.json')['checkpoint_hashes']
    if not a.smoke:
        for path,h in read(D/'manifest.json')['code_sha256'].items():assert sha(path)==h,path
    out=D/('smoke_outputs' if a.smoke else 'outputs')/a.name;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'Completed outputs are immutable'
    model,tokenizer,_,_=previous.legacy.load_model(argparse.Namespace(run_dir=str(cp),model_path=None,affine_ablation='none',device='cuda'))
    replay={}
    for split in ['internal_dev','public_dev']:
        items=items_for(tokenizer,split)
        if a.smoke:items=items[:2] if split=='internal_dev' else next(previous.batches(items,16,4096))
        old=rows(B/'outputs'/a.name/'cmrc_likelihood.jsonl') if split=='public_dev' else read(B/'outputs'/a.name/'internal_dev_ce.json')['per_example']
        old=old[:len(items)];count=0;max_error=0.
        with (out/f'{split}.jsonl').open('x') as stream:
            for batch in previous.batches(items,16 if split=='public_dev' else 1,4096):
                got=score(model,tokenizer,batch,split,check=a.smoke or count==0)
                for x,v in zip(batch,got):
                    o=old[count]
                    if split=='public_dev':
                        assert (o['row_id'],o['index'],o['tokens'],o['top1_correct'])==(v['row_id'],v['index'],v['tokens'],v['top1_correct'])
                        error=abs(o['nll']-v['nll']);assert error<1e-8,(a.name,split,count,error)
                    else:
                        assert o['record_id']==v['row_id'] and o['token_count']==v['tokens']+1
                        error=abs(o['nll_sum']-v['nll']-v['eos_nll']);assert error<1e-4,(a.name,split,count,error)
                    max_error=max(max_error,error);v['old_nll_absolute_error']=error
                    stream.write(__import__('json').dumps(v)+'\n');count+=1
                stream.flush()
                if count%256<len(batch) or count==len(items):print(split,count,'/',len(items),flush=True)
        replay[split]={'items':count,'max_old_nll_error':max_error}
    assert previous.checkpoint_hashes(cp)==hashes
    write(out/'COMPLETE.json',{'status':'passed','job':j,'finished_at':now(),'checkpoint_hashes':hashes,'replay':replay,
        'files':{str(x):sha(x) for x in out.iterdir() if x.is_file()}})
    print('PASSED',a.name,replay,flush=True)

if __name__=='__main__':main()
