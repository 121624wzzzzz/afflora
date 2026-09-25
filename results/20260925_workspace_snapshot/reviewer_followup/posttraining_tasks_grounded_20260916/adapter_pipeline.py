import subprocess,os,time
from common import *

def main():
    while not (HERE/'PILOT_COMPLETE.json').exists():time.sleep(10)
    assert read(HERE/'PILOT_COMPLETE.json')['status']=='passed'
    while not (HERE/'TASK_GATE.json').exists():time.sleep(10)
    gate=read(HERE/'TASK_GATE.json');assert gate['status']=='passed'
    tasks=gate['tasks'];assert all(t in TASKS for t in tasks)
    protocol=read(HERE/'ADAPTER_PILOT_PROTOCOL.json');frozen=read(HERE/'CODE_FROZEN.json')
    for rel,h in frozen['files'].items():assert sha(HERE/rel)==h
    reuse=[]
    for task in tasks:
        for m in MODELS:
            name=f'pilot_{task}_{m}_hidden';cp=HERE/'checkpoints'/name
            assert read(cp/'COMPLETE.json')['status']=='passed'
            spec=read(cp/'spec.json');assert spec['seed']==6100 and spec['lr']==.0002
            t=read(cp/'TRAINING.json');assert t['reload_loss_error']==0 and t['adapter_sha256']==sha(cp/'adapter.safetensors')
            reuse.append({'name':name,'task':task,'model':m,'files':{str(p.relative_to(HERE)):sha(p) for p in cp.glob('*') if p.is_file()}})
    write(HERE/'PILOT_REUSE.json',{'at':now(),'status':'passed','runs':reuse})
    jobs=[]
    # Complete a full paired seed across tasks before advancing seeds.
    for seed in protocol['seeds']:
        for task in tasks:
            for m in MODELS:
                for arm in protocol['arms']:
                    if seed==6100 and arm=='hidden':continue
                    jobs.append({'name':f'adapter_{task}_{m}_{arm}_s{seed}','model':m,'task':task,'arm':arm,'seed':seed,'lr':.0002})
    write(HERE/'ADAPTER_PILOT_JOBS.json',jobs)
    write(HERE/'ADAPTER_PROTOCOL_FROZEN.json',{'at':now(),'files':{p:sha(HERE/p) for p in ['ADAPTER_PILOT_PROTOCOL.json','adapter_pipeline.py','PILOT_REUSE.json','CODE_FROZEN.json']}})
    pending=list(jobs);active=[];finished=[];failed=[]
    while pending or active:
        used={r['gpu'] for r in active}
        for gpu in read(HERE/'GPU_POOL.json')['gpus']:
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
        write(HERE/'ADAPTER_PILOT_STATE.json',{'at':now(),'pending':[r['name'] for r in pending],
            'active':[{'name':r['spec']['name'],'gpu':r['gpu'],'pid':r['process'].pid} for r in active],'finished':finished,'failed':failed})
        if pending or active:time.sleep(10)
    write(HERE/'ADAPTER_PILOT_COMPLETE.json',{'at':now(),'status':'passed' if not failed else 'failed','finished':finished,'failed':failed})
    assert not failed,failed
if __name__=='__main__':main()
