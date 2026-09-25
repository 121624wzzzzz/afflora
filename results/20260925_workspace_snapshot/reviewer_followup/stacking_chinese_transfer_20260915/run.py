import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import queue
import subprocess
import threading
import time
from common import HERE, PYTHON, now, read, sha, write

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args()
    kind='smoke' if a.smoke else 'main'
    lockfile=(HERE/f'{kind}.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=read(HERE/'REUSE_AUDIT.json')['endpoints']
    if a.smoke:jobs=[j for j in jobs if j['seed'] in [42,None]]
    statefile=HERE/f'{kind}_state.json'
    state=read(statefile) if statefile.exists() else {'created_at':now(),'jobs':{}}
    state.update(scheduler_pid=os.getpid(),phase='running');write(statefile,state)
    q=queue.Queue()
    for j in jobs:q.put(j)
    mutex=threading.Lock()
    def update(name,**fields):
        with mutex:
            state['jobs'].setdefault(name,{}).update(fields);state['updated_at']=now();write(statefile,state)
            print(now(),name,fields,flush=True)
    def worker(gpu):
        while True:
            try:j=q.get_nowait()
            except queue.Empty:return
            name=j['name'];out=HERE/('smoke' if a.smoke else 'outputs')/name
            try:
                if (out/'COMPLETE.json').exists():
                    complete=read(out/'COMPLETE.json')
                    for path,h in complete['files'].items():assert sha(path)==h,path
                    update(name,status='complete',reused_completed_output=True);continue
                log=HERE/'logs'/f'{kind}.{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
                command=[PYTHON,'-u',str(HERE/'evaluate.py'),'--name',name]+(['--smoke'] if a.smoke else [])
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
                with log.open('a') as f:
                    proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=HERE,env=env)
                    update(name,status='running',gpu=gpu,pid=proc.pid,log=str(log),started_at=now())
                    rc=proc.wait()
                assert rc==0,f'exit {rc}: {log}'
                assert read(out/'COMPLETE.json')['status']=='passed'
                update(name,status='complete',gpu=None,pid=None,completed_at=now())
            except Exception as e:update(name,status='failed',error=str(e),gpu=None,pid=None)
            finally:q.task_done()
    with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:list(pool.map(worker,a.gpus.split(',')))
    state['phase']='complete' if all(state['jobs'][j['name']]['status']=='complete' for j in jobs) else 'failed'
    write(statefile,state);assert state['phase']=='complete'

if __name__=='__main__':main()
