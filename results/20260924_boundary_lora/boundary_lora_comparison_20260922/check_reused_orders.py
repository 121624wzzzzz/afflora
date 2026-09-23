from common import *
import torch
checked=[]
for r in read(HERE/'REUSE_AUDIT.json')['runs']:
 s=r['spec'];cp=Path(r['checkpoint']);tr=read(cp/'TRAINING.json');data=rows(HERE/f"data/{s['task']}_train.jsonl")
 order=torch.randperm(len(data),generator=torch.Generator().manual_seed(s['seed'])).tolist()
 assert read(cp/'TRAIN_ORDER.json')==dict(indices=order,ids=[data[i]['id'] for i in order]),str(cp)
 assert tr['steps']==64 and tr['examples']==2048,str(cp)
 assert tr['optimizer_whitelist_verified'] and tr['optimizer_fp32']
 checked.append(str(cp))
write(HERE/'REUSE_TRAIN_ORDER_AUDIT.json',dict(status='passed',at=now(),checked=checked))
print('reuse training order and optimizer checks passed',len(checked))
