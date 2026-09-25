import argparse,os,subprocess,queue
from concurrent.futures import ThreadPoolExecutor,as_completed
from settings import *

def specs(phase):
    out=[]
    for seed in ([TUNING_SEED] if phase!='confirmation' else CONFIRMATION_SEEDS):
        for model in read(HERE/'models.json'):
            for arm in (['base'] if phase=='reference' else ARMS):
                lrs=([5e-5] if phase=='smoke' else LRS if phase=='tuning' else [0.] if phase=='reference' else [read(HERE/'SELECTION.json')['selected'][model][arm]['lr']])
                for lr in lrs:
                    name=f'{phase}_{model}_{arm}_sd{seed}_lr{lr:g}'
                    out.append({'phase':phase,'name':name,'model':model,'arm':arm,'seed':seed,'lr':lr,'checkpoint':str(HERE/'checkpoints'/name)})
    return out

def verify_inputs(frozen=False):
    reuse=read(HERE/'REUSE_AUDIT.json');old=Path(reuse['source_root'])
    assert sha(old/'ARTIFACT_MANIFEST.json')==reuse['original_manifest_sha256']
    for p,h in reuse['copied_source_sha256'].items():assert sha(HERE/p)==h,p
    for cfg in read(HERE/'models.json').values():
        for p,h in cfg['files'].items():assert sha(p)==h,p
    assert sha(HERE/'MODEL_PROVENANCE.json')==reuse['model_provenance_sha256']
    for p,h in reuse['generated_token_sha256'].items():assert sha(HERE/p)==h,p
    assert sha(HERE/'BASE_SPECIAL_TOKEN_DIAGNOSTIC.json')==reuse['special_token_diagnostic_sha256']
    if frozen:
        for p,h in read(HERE/'FROZEN_PROTOCOL.json')['sha256'].items():assert sha(p)==h,p

def run_phase(phase):
    if phase!='smoke':assert (HERE/'PREFLIGHT.json').exists();verify_inputs(True)
    else:verify_inputs()
    jobs=specs(phase);write(HERE/f'{phase.upper()}_JOBS.json',jobs);q=queue.Queue()
    for s in jobs:q.put(s)
    def worker(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0')
        while True:
            try:s=q.get_nowait()
            except queue.Empty:return
            cp=Path(s['checkpoint']);assert not cp.exists(),cp
            sp=HERE/'specs'/f"{s['name']}.json";write(sp,s);log=HERE/'logs'/f"{s['name']}.log";log.parent.mkdir(exist_ok=True)
            if phase=='reference':cp.mkdir(parents=True);write(cp/'spec.json',s)
            commands=[] if phase=='reference' else [[PYTHON,'-u',str(HERE/'train.py'),'--spec',str(sp)]]
            split='smoke' if phase=='smoke' else 'validation' if phase=='tuning' else 'test'
            commands.append([PYTHON,'-u',str(HERE/'evaluate.py'),'--checkpoint',str(cp),'--split',split])
            with log.open('w') as f:
                for cmd in commands:
                    print(json.dumps({'at':now(),'gpu':gpu,'name':s['name'],'stage':Path(cmd[2]).name}),flush=True)
                    proc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                    if proc.returncode:
                        write(cp/'FAILED.json',{'at':now(),'command':cmd,'code':proc.returncode});raise RuntimeError(str(log))
            write(cp/'COMPLETE.json',{'at':now(),'spec':s,'gpu':gpu});print(json.dumps({'at':now(),'name':s['name'],'status':'complete'}),flush=True)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for future in as_completed([pool.submit(worker,g) for g in GPUS]):future.result()
    write(HERE/f'{phase.upper()}_COMPLETE.json',{'at':now(),'jobs':len(jobs)})

def freeze():
    assert read(HERE/'SMOKE_COMPLETE.json')['jobs']==6;init=[]
    for s in specs('smoke'):
        cp=Path(s['checkpoint']);r=read(cp/'RELOAD_AUDIT.json');assert r['status']=='passed'
        t=read(cp/'TRAINING.json');assert t['frozen_parameters_bitwise_unchanged'] and t['optimizer_whitelist_verified']
        init.append((s,read(cp/'INITIALIZATION.json')))
    for model in read(HERE/'models.json'):
        records=[r for s,r in init if s['model']==model]
        assert len({r['linear_init_sha256'] for r in records if r['arm'] in ['input','output']})==1
        assert len({r['frozen_parameters_before_sha256'] for r in records})==1
    write(HERE/'PREFLIGHT.json',{'at':now(),'status':'passed','smoke_jobs':6,'single_side_only_in_alora_arms':True,'separate_hidden_lora_control':True})
    files=list(HERE.glob('*.py'))+[HERE/p for p in ['DESIGN.md','REUSE_AUDIT.json','models.json','MODEL_PROVENANCE.json','DATA_AUDIT.json']]
    files += [HERE/p for p in read(HERE/'REUSE_AUDIT.json')['copied_source_sha256']]
    files += list((HERE/'tokens').glob('*.json'))+[HERE/'BASE_SPECIAL_TOKEN_DIAGNOSTIC.json']+list(HERE.glob('hub_metadata_*.json'))
    write(HERE/'FROZEN_PROTOCOL.json',{'at':now(),'sha256':{str(p):sha(p) for p in sorted(set(files))},'confirmation_seeds':CONFIRMATION_SEEDS,'primary_family_size':PRIMARY_FAMILY_SIZE})

def select():
    assert read(HERE/'TUNING_COMPLETE.json')['jobs']==18;result={}
    for model in read(HERE/'models.json'):
        result[model]={}
        for arm in ARMS:
            candidates=[]
            for s in specs('tuning'):
                if s['model']==model and s['arm']==arm:
                    cp=Path(s['checkpoint']);r=read(cp/'validation_metrics.json')
                    candidates.append({'lr':s['lr'],'accuracy':r['primary']['accuracy'],'candidate_nll':r['primary']['candidate_nll'],'source':str(cp),'metrics_sha256':sha(cp/'validation_metrics.json')})
            chosen=sorted(candidates,key=lambda x:(-x['accuracy'],x['candidate_nll'],x['lr']))[0]
            result[model][arm]=chosen|{'all_candidates':candidates}
    assert not (HERE/'SELECTION.json').exists()
    write(HERE/'SELECTION.json',{'at':now(),'selected':result,'new_study_test_results_accessed':False,'prior_benchmark_exposure':True})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','freeze','tuning','select','confirmation','reference']);a=p.parse_args()
    if a.phase=='freeze':freeze()
    elif a.phase=='select':select()
    else:run_phase(a.phase)
