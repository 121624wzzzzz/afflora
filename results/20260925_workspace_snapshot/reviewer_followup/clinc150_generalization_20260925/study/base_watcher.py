import time,subprocess,os
from common import *
while not (HERE/'FINAL_AUDIT.json').exists():
 if (HERE/'STATE.json').exists() and read(HERE/'STATE.json')['stage'] in ['scheduler_error','needs_technical_review']:raise RuntimeError('training needs review')
 time.sleep(30)
assert read(HERE/'FINAL_AUDIT.json')['status']=='passed'
while True:
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 avail=[int(a) for a,b in (line.split(',') for line in out.strip().splitlines()) if int(b)>30000]
 if avail:break
 time.sleep(30)
write(HERE/'BASE_STATE.json',dict(at=now(),stage='running',gpu=avail[0]))
code=subprocess.call([PYTHON,'-B',str(HERE/'base_eval.py')],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(avail[0]),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false'))
write(HERE/'BASE_STATE.json',dict(at=now(),stage='complete' if code==0 else 'failed',exit_code=code))
assert code==0
