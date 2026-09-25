import os,subprocess,queue
from concurrent.futures import ThreadPoolExecutor,as_completed
from support import *

def main():
    tasks=[s for s in original_specs() if s['arm']=='both'];q=queue.Queue()
    for s in tasks:q.put(s)
    (HERE/'logs').mkdir(exist_ok=True)
    # Verify every reused model byte before starting any intervention.
    for cfg in read(HERE/'models.json').values():
        for p,h in cfg['files'].items():assert sha(p)==h,p
    write(HERE/'ABLATION_PLAN.json',{'at':now(),'tasks':tasks,'script_sha256':sha(HERE/'ablate.py'),'design_sha256':sha(HERE/'DESIGN.md'),
        'unadapted_base':'additional reference on seed2002 checkpoint with internal LoRA and both boundary maps disabled, once per model'})
    def worker(gpu):
        while True:
            try:s=q.get_nowait()
            except queue.Empty:return
            print(json.dumps({'at':now(),'gpu':gpu,'name':s['name'],'status':'starting'}),flush=True)
            with (HERE/'logs'/f"{s['name']}.log").open('w') as f:
                proc=subprocess.run([PYTHON,'-u',str(HERE/'ablate.py'),'--checkpoint',s['checkpoint']],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),stdout=f,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(s['name'],proc.returncode)
            print(json.dumps({'at':now(),'gpu':gpu,'name':s['name'],'status':'complete'}),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,g) for g in range(8)]
        for f in as_completed(futures):f.result()
    write(HERE/'ABLATIONS_COMPLETE.json',{'at':now(),'jobs':len(tasks)})

if __name__=='__main__':main()
