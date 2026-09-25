import os,subprocess,time
from common import *

def make_jobs():
    smoke=[];jobs=[]
    def spec(m,task,arm,seed,name,smoke=False):
        return dict(name=name,model=m,task=task,arm=arm,seed=seed,lr=2e-4,microbatch=TASK_SETTINGS[task]['microbatch'],smoke=smoke)
    for m in MODELS:
        for task,arm in [('cluener','hidden_both'),('wikisql','hidden_budget')]:
            smoke.append(spec(m,task,arm,TASK_SETTINGS[task]['seed_start']-1,f'smoke_{task}_{m}_{arm}',True))
    for task in TASKS:
        for m in MODELS:jobs.append(spec(m,task,'base',TASK_SETTINGS[task]['seed_start'],f'{task}_{m}_base'))
    for i in range(5):
        for task in TASKS:
            seed=TASK_SETTINGS[task]['seed_start']+i
            for m in MODELS:
                for arm in ['hidden','hidden_budget','hidden_both']:
                    jobs.append(spec(m,task,arm,seed,f'{task}_{m}_{arm}_s{seed}'))
    return smoke,jobs

def free_gpus():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(a) for line in s.splitlines() for a,b in [line.split(',')] if int(a) in range(1,7) and int(b)<512}

def run_stage(stage,jobs):
    write(HERE/f'{stage}_JOBS.json',jobs)
    pending=[];active=[];finished=[];failed=[]
    for spec in jobs:
        complete=HERE/'checkpoints'/spec['name']/'COMPLETE.json'
        if complete.exists():
            assert read(complete)['status']=='passed';assert read(complete.parent/'spec.json')==spec
            finished.append({'name':spec['name'],'code':0,'existing_complete':True})
        else:
            assert not complete.parent.exists(),f'Incomplete run requires inspection: {complete.parent}'
            pending.append(spec)
    while pending or active:
        available=free_gpus()-{r['gpu'] for r in active}
        for gpu in sorted(available):
            if not pending or failed:break
            spec=pending.pop(0);path=HERE/'specs'/f"{spec['name']}.json";write(path,spec)
            (HERE/'logs').mkdir(exist_ok=True);log=(HERE/'logs'/f"{spec['name']}.log").open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            proc=subprocess.Popen([PYTHON,'-u',str(HERE/'run.py'),'--spec',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append({'spec':spec,'gpu':gpu,'process':proc,'log':log});print(canonical({'launched':spec['name'],'pid':proc.pid,'gpu':gpu}),flush=True)
        for r in list(active):
            code=r['process'].poll()
            if code is not None:
                r['log'].close();active.remove(r);result={'name':r['spec']['name'],'code':code}
                (finished if code==0 else failed).append(result);print(canonical(result),flush=True)
        write(HERE/f'{stage}_STATE.json',{'at':now(),'pending':[r['name'] for r in pending],
            'active':[{'name':r['spec']['name'],'gpu':r['gpu'],'pid':r['process'].pid} for r in active],'finished':finished,'failed':failed})
        if failed and not active:break
        if pending or active:time.sleep(10)
    write(HERE/f'{stage}_COMPLETE.json',{'at':now(),'status':'passed' if not failed else 'failed','finished':finished,'failed':failed})
    assert not failed,failed

def main():
    assert read(HERE/'SCORER_TESTS.json')['status']=='passed'
    for manifest in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/manifest)['files'].items():assert sha(HERE/rel)==h,rel
    for cfg in read(HERE/'models.json').values():
        for path,h in cfg['files'].items():assert sha(path)==h,path
    smoke,jobs=make_jobs()
    run_stage('SMOKE',smoke);run_stage('FORMAL',jobs)
    subprocess.run([PYTHON,str(HERE/'audit_parameters.py')],check=True)
    subprocess.run([PYTHON,str(HERE/'audit_results.py')],check=True)
    subprocess.run([PYTHON,str(HERE/'analyze.py')],check=True)
    write(HERE/'EXPERIMENT_COMPLETE.json',{'at':now(),'status':'passed','formal_jobs':len(jobs),'smoke_jobs':len(smoke)})
if __name__=='__main__':main()
