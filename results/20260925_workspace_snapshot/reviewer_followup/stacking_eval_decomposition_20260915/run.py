import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import queue
import subprocess
import threading
from shared import D,PYTHON,read,sha,write,now,jobs

def freeze():
    prep=read(D/'PREPARATION.json');assert prep['status']=='passed'
    for m in prep['template_identity'].values():assert all(v['items']==v['matching'] for v in m['counts'].values())
    assert read(D/'smoke_state.json')['phase']=='complete'
    for j in jobs():
        if j['seed']!=42:continue
        marker=read(D/'smoke_outputs'/j['name']/'COMPLETE.json');assert marker['status']=='passed'
        for path,h in marker['files'].items():assert sha(path)==h,path
    for path,h in prep['source_sha256'].items():assert sha(path)==h,path
    names=['shared.py','evaluate_parts.py','prepare.py','run.py','DESIGN.md','PREPARATION.json']
    m={'created_at':now(),'jobs':jobs(),'code_sha256':{str(D/n):sha(D/n) for n in names}}
    assert not (D/'manifest.json').exists()
    write(D/'manifest.json',m)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    kind='smoke' if a.smoke else 'main';f=(D/f'{kind}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not a.smoke:freeze()
    matrix=[j for j in jobs() if not a.smoke or j['seed']==42]
    q=queue.Queue()
    for j in matrix:q.put(j)
    state={'phase':'running','started_at':now(),'jobs':{j['name']:{'status':'queued'} for j in matrix}};lock=threading.Lock()
    def update(name,**v):
        with lock:
            state['jobs'][name].update(v);state['updated_at']=now();write(D/f'{kind}_state.json',state)
            print(now(),name,v,flush=True)
    def worker(gpu):
        while True:
            try:j=q.get_nowait()
            except queue.Empty:return
            try:
                log=D/'logs'/f'{kind}_{j["name"]}.log';log.parent.mkdir(exist_ok=True)
                cmd=[PYTHON,'-u',str(D/'evaluate_parts.py'),'--name',j['name']]+(['--smoke'] if a.smoke else [])
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',DS_IGNORE_CUDA_DETECTION='1')
                with log.open('x') as stream:
                    proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=D)
                    update(j['name'],status='running',gpu=gpu,pid=proc.pid);rc=proc.wait()
                assert rc==0,f'exit {rc}: {log}'
                complete=read(D/('smoke_outputs' if a.smoke else 'outputs')/j['name']/'COMPLETE.json');assert complete['status']=='passed'
                update(j['name'],status='complete',gpu=None,pid=None)
            except Exception as e:update(j['name'],status='failed',error=str(e),gpu=None,pid=None)
            finally:q.task_done()
    with ThreadPoolExecutor(8) as pool:list(pool.map(worker,range(8)))
    state['phase']='complete' if all(x['status']=='complete' for x in state['jobs'].values()) else 'failed'
    write(D/f'{kind}_state.json',state);assert state['phase']=='complete'

if __name__=='__main__':main()
