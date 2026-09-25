"""Read-only Base inference diagnosis; no changes to the sealed study."""
import argparse,sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'standalone_base_native_sciq_20260916'
sys.path.insert(0,str(SOURCE))
from modeling import build,batch,forward_selected
from common import prompt,read,rows,sha,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);a=parser.parse_args();m=a.model
    cfg=read(SOURCE/'models.json')[m]
    for path,h in cfg['files'].items():assert sha(path)==h
    tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
    data=rows(SOURCE/'data/test.jsonl');canonical=read(SOURCE/'tokens'/f'{m}_test.json')
    meta=read(SOURCE/'DATA_AUDIT.json')['tokenization'][m]
    reference=next((SOURCE/'checkpoints').glob(f'reference_{m}_*/test_predictions.jsonl'))
    previous=[r for r in rows(reference) if r['rotation']==0]
    assert [r['id'] for r in previous]==[r['id'] for r in data]
    rng=np.random.default_rng(20260916)
    while True:
        perm=rng.permutation(len(data))
        if np.all(perm!=np.arange(len(data))):break
    write(HERE/f'{m}_question_permutation.json',perm.tolist())
    model,audit=build({'model':m,'arm':'base','seed':5001});assert audit['trainable_parameters']==0
    model=model.cuda().float().eval();labels=torch.tensor(meta['label_ids'],device='cuda');report={}
    for condition in ['original','question_omitted','question_shuffled']:
        encoded=[]
        for i,row in enumerate(data):
            r=dict(row)
            if condition=='question_omitted':r['question']=''
            if condition=='question_shuffled':r['question']=data[int(perm[i])]['question']
            ids=tok.encode(prompt(r)+'\n\nAnswer:\n',add_special_tokens=False)
            if condition=='original':assert ids==canonical[i]['input_ids']
            encoded.append({'id':r['id'],'input_ids':ids,'gold':r['gold'],'rotation':0,'ambiguous_gold':r['ambiguous_gold']})
        pp=[];start=time.monotonic()
        with torch.inference_mode():
            for begin in range(0,len(encoded),16):
                items=encoded[begin:begin+16];ids,mask,lengths=batch(items,meta)
                logits=forward_selected(model,ids,mask,lengths);scores=logits[:,labels];pred=scores.argmax(-1);unrestricted=logits.argmax(-1)
                for j,item in enumerate(items):
                    pp.append({k:item[k] for k in ['id','gold','ambiguous_gold']}|{'prediction':int(pred[j]),'correct':bool(pred[j]==item['gold']),
                        'label_logits':scores[j].cpu().tolist(),'unrestricted_token':int(unrestricted[j]),'unrestricted_correct':bool(unrestricted[j]==labels[item['gold']]),
                        'valid_label':int(unrestricted[j]) in meta['label_ids']})
        clean=[r for r in pp if not r['ambiguous_gold']];path=HERE/f'{m}_{condition}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in pp))
        result={'n':len(clean),'candidate_accuracy':100*np.mean([r['correct'] for r in clean]),'unrestricted_first_token_accuracy':100*np.mean([r['unrestricted_correct'] for r in clean]),
            'valid_label_rate':100*np.mean([r['valid_label'] for r in clean]),'seconds':time.monotonic()-start,'prediction_sha256':sha(path)}
        if condition=='original':
            error=float(np.max(np.abs(np.array([r['label_logits'] for r in pp])-np.array([r['label_logits'] for r in previous]))))
            flips=sum(r['prediction']!=old['prediction'] for r,old in zip(pp,previous))
            assert error<5e-4 and flips==0
            result.update(independent_reference_max_abs_logit_error=error,independent_reference_prediction_flips=flips)
        report[condition]=result;print(m,condition,result,flush=True)
    write(HERE/f'{m}_RESULTS.json',{'model':m,'model_files_verified':cfg['files'],'original_weights_only':True,'conditions':report})

if __name__=='__main__':main()
