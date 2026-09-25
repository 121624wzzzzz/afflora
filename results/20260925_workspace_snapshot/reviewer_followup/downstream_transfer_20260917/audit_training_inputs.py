"""Independent final training-token/order audit; no model or artifact changes."""
import torch
from transformers import AutoTokenizer
from common import *

count=0;orders=0
for task in TASKS:
    data=rows(HERE/f'data/{task}_train.jsonl')
    for m in MODELS:
        tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[m]['path'],local_files_only=True)
        tokens=read(HERE/f'tokens/{task}_{m}_train.json')
        assert [r['id'] for r in tokens]==[r['id'] for r in data]
        for r,t in zip(data,tokens):
            assert t['prompt_ids']==tok.encode(prompt(r),add_special_tokens=False)
            text=r['target'] if task=='anli_r1' else canonical(r['target'])
            assert t['target_ids']==tok.encode(text,add_special_tokens=False)+[151643]
            count+=1
        for seed in range(7100,7105):
            order=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed)).tolist()
            expected={'indices':order,'ids':[data[i]['id'] for i in order]}
            for arm in ['hidden','hidden_budget','hidden_both']:
                path=HERE/'checkpoints'/f'{task}_{m}_{arm}_s{seed}'/'TRAIN_ORDER.json'
                assert read(path)==expected,path;orders+=1
write(HERE/'TRAINING_INPUT_AUDIT.json',{'at':now(),'status':'passed','training_token_records':count,'complete_orders_verified':orders})
print('Verified',count,'training encodings and',orders,'complete seed orders')
