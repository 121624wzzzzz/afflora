import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
from common import HERE, CHAT, PYTHON, checkpoint_hashes, now, read, sha, write
from experiment import jobs,variant,command,MODELS

def validate_train(j):
    cp=Path(j['checkpoint']);args=read(cp/'run_args.json');init=read(cp/'initialization_audit.json');metrics=read(cp/'train_results.json')
    count=32 if j['smoke'] else read(HERE/'DATA_AND_MODEL_AUDIT.json')['train_rows']
    assert args['variant']==variant(j) and args['seed']==j['seed'] and args['model_path']==MODELS[j['model']]
    assert args['learning_rate']==5e-5 and args['num_train_epochs']==1 and args['max_seq_len']==2048
    assert init['arm']==j['arm'] and init['train_rows']==count and init['train_batch']==8 and init['gradient_accumulation']==2
    assert metrics['global_step']==(2 if j['smoke'] else math.ceil(count/16)) and math.isfinite(metrics['train_loss'])
    assert args['corrected_data_pipeline']['implementation_sha256']==sha(HERE/'source/corrected_sft_experiment/data_pipeline.py')
    if j['arm']=='hidden_budget':assert init['budget_control']['actual_extra_parameters']==init['budget_control']['target_extra_parameters']
    for f in ['adapter_model.safetensors','adapter_config.json']:
        assert (cp/f).exists()
    if j['arm'] in ['both','output']:assert (cp/'affine_vocab_adapter.safetensors').exists()
    marker=cp/'TRAIN_COMPLETE.json';h=checkpoint_hashes(cp)
    if marker.exists():assert read(marker)['checkpoint_hashes']==h
    else:write(marker,{'validated_at':now(),'checkpoint_hashes':h,'global_step':metrics['global_step']})

def freeze():
    path=HERE/'manifest.json'
    if path.exists():
        m=read(path)
        for p,h in m['local_sha256'].items():assert sha(p)==h,p
        return m
    assert read(HERE/'smoke_state.json')['phase']=='complete'
    for model in MODELS:
        group=[j for j in jobs(True) if j['model']==model]
        init=[read(Path(j['checkpoint'])/'initialization_audit.json') for j in group]
        assert len({x['shared_hidden_init_sha256'] for x in init})==1
        by={j['arm']:v for j,v in zip(group,init)}
        assert by['both']['affine_components']['input']==by['both']['affine_components']['output']==by['output']['affine_components']['output']
        for j in group:
            validate_train(j)
            assert read(Path(j['checkpoint'])/'RELOAD_AUDIT.json')['status']=='passed'
            complete=read(HERE/'smoke_outputs'/j['name']/'COMPLETE.json')
            for p,h in complete['files'].items():assert sha(p)==h,p
            for p,h in complete['identity']['implementation_sha256'].items():assert sha(p)==h,p
            assert complete['identity']['protocol_sha256']==sha(HERE/'DESIGN.md')
    source_checks={}
    for p in (HERE/'source').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            original=CHAT/'source'/p.relative_to(HERE/'source');assert sha(p)==sha(original);source_checks[str(original)]=sha(original)
    for n in ['train.py','budget.py']:assert sha(HERE/n)==sha(CHAT/n);source_checks[str(CHAT/n)]=sha(CHAT/n)
    # Scoring uses the previously frozen independent loader and NLTK resources.
    original=read(CHAT/'manifest.json')
    for p,h in original['sha256'].items():
        if any(str(CHAT/sub)+'/' in p for sub in ['source','ifeval_deps','nltk_data']):assert sha(p)==h;source_checks[p]=h
    data_audit=read(HERE/'DATA_AND_MODEL_AUDIT.json')
    for p,h in data_audit['base_model_sha256'].items():assert sha(p)==h;source_checks[p]=h
    fixed=['common.py','experiment.py','train.py','budget.py','check_reload.py','evaluate.py','run.py','prepare_data.py',
        'cmrc_official_py3.py','DESIGN.md','DATA_AND_MODEL_AUDIT.json']
    local=[HERE/n for n in fixed]+[p for sub in ['source','data'] for p in (HERE/sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','safetensors','datasets']}
    m={'created_at':now(),'matrix':jobs(),'smoke_jobs':jobs(True),'train_rows':data_audit['train_rows'],'optimization_steps':570,
       'local_sha256':{str(p):sha(p) for p in local},'external_sha256':source_checks,'versions':versions,
       'training_commands':{j['name']:command(j) for j in jobs()},'scope':'CMRC task-specific fresh SFT; public-dev was previously evaluated for transfer.'}
    write(path,m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args()
    kind='smoke' if a.smoke else 'main';lockfile=(HERE/f'{kind}.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not a.smoke:freeze()
    statefile=HERE/f'{kind}_state.json';state=read(statefile) if statefile.exists() else {'jobs':{},'created_at':now()}
    state.update(phase='running',scheduler_pid=os.getpid());write(statefile,state)
    q=queue.Queue()
    for j in jobs(a.smoke):q.put(j)
    mutex=threading.Lock()
    def update(name,**fields):
        with mutex:
            state['jobs'].setdefault(name,{}).update(fields);state['updated_at']=now();write(statefile,state)
            print(now(),name,fields,flush=True)
    def execute(j,gpu,stage,cmd):
        name=j['name'];log=HERE/'logs'/f'{name}.{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),STACKING_ARM=j['arm'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            DS_IGNORE_CUDA_DETECTION='1',HF_DATASETS_OFFLINE='1')
        with log.open('a') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=HERE,env=env)
            update(name,status=stage,gpu=gpu,pid=proc.pid,log=str(log),stage_started_at=now())
            rc=proc.wait()
        assert rc==0,f'{stage} exit {rc}: {log}'
    def worker(gpu):
        while True:
            try:j=q.get_nowait()
            except queue.Empty:return
            try:
                cp=Path(j['checkpoint']);out=HERE/('smoke_outputs' if a.smoke else 'outputs')/j['name']
                if not (cp/'TRAIN_COMPLETE.json').exists():execute(j,gpu,'train',command(j))
                validate_train(j)
                if a.smoke and not (cp/'RELOAD_AUDIT.json').exists():execute(j,gpu,'reload',[PYTHON,str(HERE/'check_reload.py'),'--checkpoint',str(cp)])
                if not (out/'COMPLETE.json').exists():execute(j,gpu,'evaluate',[PYTHON,'-u',str(HERE/'evaluate.py'),'--name',j['name']]+(['--smoke'] if a.smoke else []))
                complete=read(out/'COMPLETE.json');assert complete['status']=='passed'
                for p,h in complete['files'].items():assert sha(p)==h,p
                update(j['name'],status='complete',gpu=None,pid=None,completed_at=now())
            except Exception as e:update(j['name'],status='failed',error=str(e),gpu=None,pid=None)
            finally:q.task_done()
    with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:list(pool.map(worker,a.gpus.split(',')))
    state['phase']='complete' if all(state['jobs'][j['name']]['status']=='complete' for j in jobs(a.smoke)) else 'failed'
    write(statefile,state);assert state['phase']=='complete'

if __name__=='__main__':main()
