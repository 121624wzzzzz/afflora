import argparse,gc,json
import torch
from common import *
from modeling import build,batch,forward_selected
from run import specs

def zero_check(model_key):
    report={};meta=read(HERE/'DATA_AUDIT.json')['tokenization'][model_key]
    items=read(HERE/'tokens'/f'{model_key}_validation.json')[:3]
    for arm in ['both','interior']:
        model,audit=build({'model':model_key,'arm':arm,'seed':2001});model.cuda().eval().float()
        ids,mask,lengths=batch(items,meta)
        with torch.inference_mode():
            active=forward_selected(model,ids,mask,lengths)
            active_counts=dict(model._placement_counts)
            for handle in model._placement_handles:handle.remove()
            inactive=forward_selected(model,ids,mask,lengths)
        error=float((active-inactive).abs().max());assert error==0,error
        assert all(v>0 for v in active_counts.values())
        report[arm]={'zero_start_max_logit_error':error,'hook_counts':active_counts,'positions':audit['positions']}
        del model,active,inactive;gc.collect();torch.cuda.empty_cache()
    write(HERE/f'ZERO_START_{model_key}.json',report);print(json.dumps(report))

def finalize():
    checks={}
    for model in read(HERE/'models.json'):
        assert (HERE/f'ZERO_START_{model}.json').exists()
        assert read(HERE/f'BOUNDARY_EQUIVALENCE_{model}.json')['status']=='passed'
        runs={s['arm']:Path(s['checkpoint']) for s in specs('smoke') if s['model']==model}
        init={a:read(cp/'INITIALIZATION.json') for a,cp in runs.items()}
        reloads={a:read(cp/'RELOAD_AUDIT.json') for a,cp in runs.items()}
        assert all(x['status']=='passed' for x in reloads.values())
        assert len({x['shared_hidden_init_sha256'] for x in init.values()})==1
        assert init['both']['affine_init_sha256']==init['interior']['affine_init_sha256']
        assert len({init[a]['total_parameters'] for a in ['both','interior','hidden_budget']})==1
        orders={sha(cp/'TRAIN_ORDER.json') for cp in runs.values()};assert len(orders)==1
        checks[model]={'initializations':init,'reloads':reloads,'zero_start':read(HERE/f'ZERO_START_{model}.json'),
            'verified_original_wrapper_equivalence':read(HERE/f'BOUNDARY_EQUIVALENCE_{model}.json'),
            'matched_data_order':True,'matched_shared_hidden_initialization':True,'matched_affine_initialization':True,'exact_budgets':True}
    write(HERE/'PREFLIGHT.json',{'status':'passed','at':now(),'models':checks})
    files=list(HERE.glob('*.py'))+list((HERE/'source').glob('*.py'))+[HERE/'DESIGN.md',HERE/'models.json',HERE/'DATA_AUDIT.json']
    files+=list((HERE/'tokens').glob('*.json'))+list((HERE/'data').glob('*.jsonl'))
    write(HERE/'FROZEN_PROTOCOL.json',{'at':now(),'sha256':{str(p):sha(p) for p in sorted(files)},
        'confirmation_seeds':list(range(2002,2007)),'primary_family_size':6,
        'rules':'DESIGN.md, analyze.py, run.py; no test outcome accessed'})
    print('Preflight passed; protocol frozen.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model');a=p.parse_args()
    if a.model:zero_check(a.model)
    else:finalize()
