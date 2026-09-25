"""Run sequential resource measurements only after this training queue, on an idle device."""
import time,subprocess,os
from common import *
write(HERE/'RESOURCE_BENCHMARK_STATE.json',dict(at=now(),stage='waiting_for_training_then_idle_gpu'))
while not (HERE/'STATE.json').exists() or read(HERE/'STATE.json')['stage']!='complete':time.sleep(60)
while True:
 result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 free=[int(row.split(',')[0]) for row in result.splitlines() if int(row.split(',')[1])>=78000 and int(row.split(',')[2])<=5]
 if free:break
 write(HERE/'RESOURCE_BENCHMARK_STATE.json',dict(at=now(),stage='waiting_for_idle_gpu'));time.sleep(60)
gpu=free[0];write(HERE/'RESOURCE_BENCHMARK_STATE.json',dict(at=now(),stage='running',gpu=gpu))
with (HERE/'logs/resource_profile.log').open('w') as f:
 result=subprocess.run([PYTHON,'-B',str(HERE/'profile_resources.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4'),cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
write(HERE/'RESOURCE_BENCHMARK_STATE.json',dict(at=now(),stage='passed' if result.returncode==0 else 'failed',exit_code=result.returncode,gpu=gpu))
