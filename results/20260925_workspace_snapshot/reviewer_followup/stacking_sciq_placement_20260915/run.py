"""Persistent eight-GPU queue; each training is followed by independent reload."""
import argparse,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from common import *

def specs(phase):
    models=read(HERE/'models.json');out=[]
    for seed in ([2001] if phase!='confirmation' else range(2002,2007)):
        for model in models:
            for arm in ARMS:
                lrvalues=([5e-5] if phase=='smoke' else LRS if phase=='tuning' else [read(HERE/'SELECTION.json')['selected'][model][arm]['lr']])
                for lr in lrvalues:
                    name=f'{phase}_{model}_{arm}_sd{seed}_lr{lr:g}'
                    out.append({'name':name,'phase':phase,'model':model,'arm':arm,'seed':seed,'lr':lr,
                        'checkpoint':str(HERE/'checkpoints'/name)})
    return out

def worker(gpu,jobs):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0')
    for spec in jobs:
        name=spec['name'];sp=HERE/'specs'/f'{name}.json';cp=Path(spec['checkpoint'])
        write(sp,spec);log=HERE/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        assert not cp.exists(),str(cp)
        with log.open('w') as f:
            commands=[[PYTHON,'-u',str(HERE/'train.py'),'--spec',str(sp)],
                [PYTHON,'-u',str(HERE/'evaluate.py'),'--checkpoint',str(cp),'--split',
                 'smoke' if spec['phase']=='smoke' else 'validation' if spec['phase']=='tuning' else 'test']]
            for command in commands:
                print(json.dumps({'at':now(),'gpu':gpu,'name':name,'command':Path(command[2]).name}),flush=True)
                proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
                if proc.returncode:
                    write(cp/'FAILED.json',{'command':command,'code':proc.returncode,'at':now()})
                    raise RuntimeError(f'{name} failed: {log}')
        write(cp/'COMPLETE.json',{'at':now(),'gpu':gpu,'spec':spec})
        print(json.dumps({'at':now(),'status':'complete','gpu':gpu,'name':name}),flush=True)

def select():
    result={}
    for model in read(HERE/'models.json'):
        result[model]={}
        for arm in ARMS:
            candidates=[]
            for s in specs('tuning'):
                if s['model']==model and s['arm']==arm:
                    cp=Path(s['checkpoint']);assert (cp/'COMPLETE.json').exists()
                    r=read(cp/'validation_metrics.json')
                    candidates.append({'lr':s['lr'],'accuracy':r['primary']['accuracy'],
                        'candidate_nll':r['primary']['candidate_nll'],'source':str(cp),'metrics_sha256':sha(cp/'validation_metrics.json')})
            chosen=sorted(candidates,key=lambda x:(-x['accuracy'],x['candidate_nll'],x['lr']))[0]
            result[model][arm]={**chosen,'all_candidates':candidates}
    assert not (HERE/'SELECTION.json').exists()
    write(HERE/'SELECTION.json',{'frozen_at':now(),'selected':result,'rule':'validation accuracy, then candidate NLL, then lower LR',
        'test_metrics_accessed':False})
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','tuning','confirmation','select']);a=p.parse_args()
    if a.phase=='select':select();return
    if a.phase in ['tuning','confirmation']:assert (HERE/'PREFLIGHT.json').exists() and (HERE/'FROZEN_PROTOCOL.json').exists()
    if a.phase=='confirmation':assert (HERE/'SELECTION.json').exists()
    jobs=specs(a.phase);write(HERE/f'{a.phase.upper()}_JOBS.json',jobs)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,gpu,jobs[gpu::8]) for gpu in range(8)]
        for future in as_completed(futures):future.result()
    write(HERE/f'{a.phase.upper()}_COMPLETE.json',{'at':now(),'jobs':len(jobs)})

if __name__=='__main__':main()
