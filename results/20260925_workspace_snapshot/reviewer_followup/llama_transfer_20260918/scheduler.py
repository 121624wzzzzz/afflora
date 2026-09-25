"""Finite eight-GPU queue; scores cannot affect admission, retries or settings."""
import fcntl, os, subprocess, time
from common import *

def gpu_memory():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return {int(line.split(',')[0]):int(line.split(',')[1]) for line in r.stdout.splitlines()}

def event(kind,**kw):
    record={'at':now(),'event':kind,**kw}
    with (HERE/'EVENTS.jsonl').open('a') as f:f.write(canonical(record)+'\n')
    print(canonical(record),flush=True)

def main():
    lock=(HERE/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(HERE/'READY.json')['status']=='passed';assert not (HERE/'STATE.json').exists()
    records=[dict(j,state='pending') for j in read(HERE/'SMOKE_JOBS.json')+read(HERE/'FORMAL_JOBS.json')]
    active={};audits={};phase='smokes';start=now()
    (HERE/'logs').mkdir(exist_ok=True);(HERE/'specs').mkdir(exist_ok=True)
    while True:
        for name,(p,j,stream) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            stream.close();del active[name];j.update(returncode=rc,worker_finished_at=now())
            if rc!=0 or not (HERE/'checkpoints'/name/'COMPLETE.json').exists():
                j['state']='failed';event('worker_failed',name=name,returncode=rc);continue
            log=(HERE/'logs'/f'{name}.audit.log').open('w')
            p=subprocess.Popen([PYTHON,'-u',str(HERE/'audit_one.py'),'--name',name],cwd=HERE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT)
            j.update(state='auditing',audit_pid=p.pid);audits[name]=(p,j,log);event('audit_started',name=name,pid=p.pid)
        changed=False
        for name,(p,j,stream) in list(audits.items()):
            rc=p.poll()
            if rc is None:continue
            stream.close();del audits[name]
            ok=rc==0 and (HERE/'audits'/f'{name}.json').exists() and read(HERE/'audits'/f'{name}.json')['status']=='passed'
            j.update(state='passed' if ok else 'failed',audit_returncode=rc,finished_at=now());changed=True;event(j['state'],name=name)
        if changed:subprocess.run([PYTHON,str(HERE/'analyze.py')],cwd=HERE,check=True)
        smokes=[j for j in records if j['spec']['smoke']]
        failed=any(j['state']=='failed' for j in records)
        if phase=='smokes' and all(j['state']=='passed' for j in smokes):phase='formal';event('formal_admission_opened')
        if not failed:
            memory=gpu_memory();busy={j['gpu'] for p,j,f in active.values()}
            for gpu in range(8):
                if gpu in busy:continue
                candidates=[j for j in records if j['state']=='pending' and (j['spec']['smoke'] or phase=='formal') and memory.get(gpu,0)>=(49152 if j['spec']['model']=='llama31_8b_base' else 36864)]
                if not candidates:continue
                j=candidates[0];s=j['spec'];name=s['name'];write(HERE/'specs'/f'{name}.json',s)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
                stream=(HERE/'logs'/f'{name}.log').open('w')
                p=subprocess.Popen([PYTHON,'-u',str(HERE/'run.py'),'--spec',str(HERE/'specs'/f'{name}.json')],cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
                j.update(state='running',gpu=gpu,pid=p.pid,started_at=now());active[name]=(p,j,stream);event('started',name=name,gpu=gpu,pid=p.pid)
        state={'started_at':start,'at':now(),'phase':phase,'jobs':records,'counts':{s:sum(j['state']==s for j in records) for s in ['pending','running','auditing','passed','failed']}}
        write(HERE/'STATE.json',state)
        if not active and not audits and (failed or all(j['state']=='passed' for j in records)):
            write(HERE/'SCHEDULER_COMPLETE.json',{'at':now(),'status':'failed' if failed else 'passed','counts':state['counts']});event('scheduler_complete',status='failed' if failed else 'passed');break
        time.sleep(10)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
