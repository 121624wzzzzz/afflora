import argparse,os,subprocess,time
from common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','pilot']);args=parser.parse_args()
    jobs=[]
    if args.stage=='smoke':
        for i,m in enumerate(MODELS):
            for arm,task in [('hidden_both','toolace'),('hidden_budget','cluener')]:
                jobs.append({'name':f'smoke_{m}_{arm}','model':m,'task':task,'arm':arm,'seed':6099,'lr':2e-4,'smoke':True})
    else:
        assert read(HERE/'SMOKE_COMPLETE.json')['status']=='passed'
        for task in TASKS:
            for m in MODELS:
                for arm in ['base','hidden']:
                    jobs.append({'name':f'pilot_{task}_{m}_{arm}','model':m,'task':task,'arm':arm,'seed':6100,'lr':2e-4,'curve_steps':[16] if arm=='hidden' else []})
    write(HERE/f'{args.stage.upper()}_JOBS.json',jobs)
    (HERE/'logs').mkdir(exist_ok=True);pending=list(jobs);active=[];finished=[];failed=[];gpus=[1,2,3,4,5,6]
    while pending or active:
        used={r['gpu'] for r in active}
        for gpu in gpus:
            if not pending:break
            if gpu in used:continue
            spec=pending.pop(0);path=HERE/'specs'/f"{spec['name']}.json";write(path,spec)
            log=(HERE/'logs'/f"{spec['name']}.log").open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            proc=subprocess.Popen([PYTHON,'-u',str(HERE/'run.py'),'--spec',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append({'spec':spec,'gpu':gpu,'process':proc,'log':log});print(json.dumps({'launched':spec['name'],'pid':proc.pid,'gpu':gpu}),flush=True)
        for r in list(active):
            code=r['process'].poll()
            if code is not None:
                r['log'].close();active.remove(r);result={'name':r['spec']['name'],'code':code}
                (finished if code==0 else failed).append(result);print(json.dumps(result),flush=True)
        write(HERE/f'{args.stage.upper()}_STATE.json',{'at':now(),'pending':[r['name'] for r in pending],
            'active':[{'name':r['spec']['name'],'gpu':r['gpu'],'pid':r['process'].pid} for r in active],'finished':finished,'failed':failed})
        if pending or active:time.sleep(10)
    write(HERE/f'{args.stage.upper()}_COMPLETE.json',{'at':now(),'status':'passed' if not failed else 'failed','finished':finished,'failed':failed})
    assert not failed,failed
if __name__=='__main__':main()
