import os,subprocess,queue
from concurrent.futures import ThreadPoolExecutor,as_completed
from support import *

tasks=[s for s in original_specs() if s['arm']=='both'];q=queue.Queue()
for s in tasks:q.put(s)
write(HERE/'EOS_FROZEN_PLAN.json',{'at':now(),'tasks':tasks,'script_sha256':sha(HERE/'eos_probe.py'),'plan_sha256':sha(HERE/'EOS_PROBE_PLAN.md')})
def worker(gpu):
    while True:
        try:s=q.get_nowait()
        except queue.Empty:return
        print(json.dumps({'at':now(),'gpu':gpu,'name':s['name'],'status':'starting'}),flush=True)
        with (HERE/'logs'/f"eos_{s['name']}.log").open('w') as f:
            p=subprocess.run([PYTHON,'-u',str(HERE/'eos_probe.py'),'--checkpoint',s['checkpoint']],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0,(s['name'],p.returncode)
        print(json.dumps({'at':now(),'gpu':gpu,'name':s['name'],'status':'complete'}),flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
    futures=[pool.submit(worker,g) for g in range(8)]
    for f in as_completed(futures):f.result()
write(HERE/'EOS_COMPLETE.json',{'at':now(),'jobs':len(tasks)})
