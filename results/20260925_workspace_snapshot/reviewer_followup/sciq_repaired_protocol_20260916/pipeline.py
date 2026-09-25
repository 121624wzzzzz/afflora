import argparse
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from settings import *


def verify_sealed_sources():
    reuse=read(HERE/'REUSE_AUDIT.json')
    for name,audit in reuse['sealed_sources'].items():
        source=HERE.parent/name;manifest=source/'ARTIFACT_MANIFEST.json'
        assert sha(manifest)==audit['manifest_sha256']
        for item in read(manifest)['files']:assert sha(source/item['path'])==item['sha256']
    for filename,record in reuse['copied'].items():
        if 'sha256' in record:assert sha(HERE/filename)==record['sha256']
    for cfg in read(HERE/'models.json').values():
        for filename,digest in cfg['files'].items():assert sha(filename)==digest


def verify_frozen():
    for name,digest in read(HERE/'FROZEN_PROTOCOL.json')['sha256'].items():assert sha(HERE/name)==digest,name


def new_job(phase,model,seed,lr):
    name=f'{phase}_{model}_small_q_sd{seed}_lr{lr:g}' + ('_attempt2' if phase=='smoke' else '')
    spec={'phase':phase,'name':name,'model':model,'arm':'small_q','seed':seed,'lr':lr,
          'checkpoint':str(HERE/'checkpoints'/name)}
    return {'kind':'train','spec':spec,'split':'smoke' if phase=='smoke' else 'validation' if phase=='tuning' else 'test'}


def run_jobs(jobs,label):
    assert not (HERE/'ARTIFACT_MANIFEST.json').exists(),'Sealed run: create a new output directory.'
    write(HERE/f'{label}_JOBS.json',jobs);tasks=queue.Queue();stop=threading.Event()
    for job in jobs:tasks.put(job)
    def worker(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0')
        while not stop.is_set():
            try:job=tasks.get_nowait()
            except queue.Empty:return
            spec=job['spec'];cp=Path(spec['checkpoint']);commands=[]
            if job['kind']=='train':
                assert not cp.exists(),cp
                path=HERE/'specs'/f"{spec['name']}.json";write(path,spec)
                commands.append([PYTHON,'-u',str(HERE/'train.py'),'--spec',str(path)])
            commands.append([PYTHON,'-u',str(HERE/'evaluate.py'),'--checkpoint',str(cp),'--split',job['split']])
            log=HERE/'logs'/f"{spec['name']}.log";log.parent.mkdir(exist_ok=True)
            with log.open('w') as output:
                for command in commands:
                    print(json.dumps({'at':now(),'gpu':gpu,'name':spec['name'],'stage':Path(command[2]).name}),flush=True)
                    result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT)
                    if result.returncode:
                        stop.set();write(HERE/'failures'/f"{spec['name']}.json",{'command':command,'returncode':result.returncode})
                        raise RuntimeError(str(log))
            write(HERE/'job_status'/f"{spec['name']}.json",{'at':now(),'status':'complete','gpu':gpu,'job':job})
            print(json.dumps({'at':now(),'name':spec['name'],'status':'complete'}),flush=True)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for future in as_completed([pool.submit(worker,gpu) for gpu in GPUS]):future.result()
    write(HERE/f'{label}_COMPLETE.json',{'at':now(),'jobs':len(jobs)})


def smoke():
    verify_sealed_sources()
    subprocess.run([PYTHON,'-m','unittest','test_scoring'],cwd=HERE,check=True)
    run_jobs([new_job('smoke',model,5000,5e-5) for model in BASE_MODELS],'SMOKE')


def freeze():
    assert read(HERE/'SMOKE_COMPLETE.json')['jobs']==2
    audit=[]
    for job in read(HERE/'SMOKE_JOBS.json'):
        spec=job['spec'];cp=Path(spec['checkpoint'])
        smoke=read(HERE/'evaluations'/spec['name']/'smoke.json');assert smoke['status']=='passed'
        train=read(cp/'TRAINING.json');assert train['frozen_parameters_bitwise_unchanged'] and train['optimizer_whitelist_verified']
        init=read(cp/'INITIALIZATION.json');audit.append({'spec':spec,'initialization':init,'checks':smoke})
    write(HERE/'PREFLIGHT.json',{'at':now(),'status':'passed','smoke':audit})
    paths=list(HERE.glob('*.py'))+[HERE/name for name in ['DESIGN.md','PREFLIGHT_NUMERICS.md','models.json','DATA_AUDIT.json','REUSE_AUDIT.json']]
    paths+=list((HERE/'data').glob('*'))+list((HERE/'tokens').glob('*'))+list((HERE/'source').glob('*.py'))
    write(HERE/'FROZEN_PROTOCOL.json',{'at':now(),'sha256':{str(path.relative_to(HERE)):sha(path) for path in sorted(paths)},
                                    'post_hoc_repair':True,'primary_family_size':PRIMARY_FAMILY_SIZE,
                                    'efficiency_family_size':EFFICIENCY_FAMILY_SIZE})


def evaluation_and_tuning():
    verify_frozen();jobs=[];reuse=[]
    for name in ['standalone_base_native_sciq_20260916','standalone_single_boundary_sciq_20260915']:
        source=HERE.parent/name
        for file in sorted((source/'checkpoints').glob('*/spec.json')):
            spec=read(file)
            if spec['phase'] not in ['confirmation','reference']:continue
            cp=file.parent
            assert Path(spec['checkpoint'])==cp
            if spec['phase']=='confirmation':
                train=read(cp/'TRAINING.json');assert train['frozen_parameters_bitwise_unchanged']
                assert sha(cp/'adapter.safetensors')==train['adapter_sha256']
            reuse.append({'checkpoint':str(cp),'spec_sha256':sha(file),
                          'adapter_sha256':sha(cp/'adapter.safetensors') if spec['phase']=='confirmation' else None})
            jobs.append({'kind':'reevaluate','spec':spec,'split':'test'})
    assert len(jobs)==54,len(jobs)
    write(HERE/'CHECKPOINT_REUSE.json',reuse)
    # Place tuning first so its training starts immediately; every old run still reruns.
    jobs=[new_job('tuning',model,TUNING_SEED,lr) for model in BASE_MODELS for lr in LRS]+jobs
    run_jobs(jobs,'REEVALUATION_AND_TUNING')


def select():
    verify_frozen();selected={}
    for model in BASE_MODELS:
        candidates=[]
        for lr in LRS:
            job=new_job('tuning',model,TUNING_SEED,lr);spec=job['spec']
            path=HERE/'evaluations'/spec['name']/'validation_metrics.json';metrics=read(path)['primary']
            candidates.append({'lr':lr,'accuracy':metrics['candidate_accuracy'],'candidate_nll':metrics['candidate_nll'],
                               'source':str(path),'metrics_sha256':sha(path)})
        chosen=sorted(candidates,key=lambda x:(-x['accuracy'],x['candidate_nll'],x['lr']))[0]
        selected[model]=chosen|{'candidates':candidates}
    assert not (HERE/'SELECTION.json').exists()
    write(HERE/'SELECTION.json',{'at':now(),'selected':selected,'rule':'validation candidate accuracy, NLL, LR',
                                'old_benchmark_exposure':True,'new_control_test_accessed':False})


def confirmation():
    verify_frozen();selection=read(HERE/'SELECTION.json')['selected']
    run_jobs([new_job('confirmation',model,seed,selection[model]['lr']) for model in BASE_MODELS for seed in CONFIRMATION_SEEDS],
             'CONFIRMATION')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','freeze','evaluate_tune','select','confirmation','all'])
    args=parser.parse_args()
    if args.phase=='all':
        smoke();freeze();evaluation_and_tuning();select();confirmation()
    else:{'smoke':smoke,'freeze':freeze,'evaluate_tune':evaluation_and_tuning,'select':select,'confirmation':confirmation}[args.phase]()
