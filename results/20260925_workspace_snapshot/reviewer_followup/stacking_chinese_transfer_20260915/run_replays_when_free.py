import fcntl
import os
import subprocess
import time
from common import HERE, PYTHON, now, read, write

lock=(HERE/'generation_replay.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    state=read(HERE/'main_state.json')
    assert state['phase']!='failed', 'Main run failed; inspect before extra checks.'
    if sum(x['status']=='complete' for x in state['jobs'].values())>=24:break
    time.sleep(5)
busy={int(x['gpu']) for x in state['jobs'].values() if x['status']=='running'}
free=sorted(set(range(8))-busy);assert len(free)>=6
names=[f'qwen3_06b_chat_{arm}_hr8_sd{seed}' for seed in [42,43] for arm in ['none','both']]
names += [f'qwen25_15b_chat_{arm}_hr8_sd43' for arm in ['none','both']]
processes=[]
for name,gpu in zip(names,free):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    log=HERE/'logs'/f'generation_replay.{name}.log'
    with log.open('w') as f:
        p=subprocess.Popen([PYTHON,'-u',str(HERE/'replay_generation.py'),'--name',name],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=HERE)
    print(now(),'replay',gpu,name,p.pid,flush=True);processes.append((name,p))
results={}
for name,p in processes:
    assert p.wait()==0,name
    results[name]=read(HERE/'generation_replay'/name/'AUDIT.json')
write(HERE/'GENERATION_REPLAY_AUDIT.json',{'status':'passed','completed_at':now(),'checks':results})
print('ALL SIX GENERATION REPLAYS PASSED',flush=True)
