import os
os.environ['DS_IGNORE_CUDA_DETECTION']='1'
from evaluate import *
configure()
j=next(x for x in jobs(True) if x['model']=='qwen25_15b_chat' and x['arm']=='both')
model,tok=load(Path(j['checkpoint']));data=rows(HERE/'token_cache'/j['model']/'internal_dev.jsonl')[:8]
report=[]
for row in data:
    answer=row['unique_answers'][0]
    single=score(model,tok,row,answer)[0]
    repeat=score(model,tok,row,answer)[0]
    full=score(model,tok,row,answer,full=True)[0]
    variants={}
    for copies,pad in [(1,17),(2,0),(2,17)]:
        r=score(model,tok,row,answer,copies=copies,padding=pad)[0]
        delta=[a-b for a,b in zip(single['token_losses'],r['token_losses'])]
        worst=max(range(len(delta)),key=lambda k:abs(delta[k]))
        variants[f'{copies}_{pad}']={'max_loss_diff':max(map(abs,delta)),'prob_diff':single['probability']-r['probability'],
            'worst_index':worst,'worst_loss':single['token_losses'][worst],'reference_loss':r['token_losses'][worst]}
    report.append({'id':row['id'],'canonical_replay_exact':single==repeat,'full_logits_max_diff':max(abs(a-b) for a,b in zip(single['token_losses'],full['token_losses'])),
        'variants':variants})
write(HERE/'NUMERICAL_INVESTIGATION.json',{'status':'investigated','checked_at':now(),'rows':report})
print(report,flush=True)
