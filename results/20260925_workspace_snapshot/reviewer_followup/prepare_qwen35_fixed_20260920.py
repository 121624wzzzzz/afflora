"""Fresh per-size gates, with a distinct GPU/process for reference verification."""
import os,json,subprocess,concurrent.futures,fcntl,traceback,hashlib
from pathlib import Path
from datetime import datetime
PARENT=Path(__file__).resolve().parent
MASTER=PARENT/'qwen35_fixed_multiscale_20260920'
PYTHON='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def now():return datetime.now().astimezone().isoformat()
def one(pair):
 key,gpu=pair;root=PARENT/f'qwen35_fixed_{key}_20260920'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
  OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_COMPILE_THREADS='1')
 def run(script,device):
  write(root/'PREPARATION_STATE.json',dict(at=now(),script=script,gpu=device,status='running'))
  print(key,'starting',script,'GPU',device,flush=True)
  with (root/(script+'.log')).open('x') as f:
   subprocess.run([PYTHON,'-u',str(root/script)],cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=str(device)),stdout=f,stderr=subprocess.STDOUT,check=True)
  print(key,'passed',script,flush=True)
 try:
  for script in ['prepare.py','test_native_formula.py','test_accelerated_kernel.py','technical_probe.py','initial_references.py']:run(script,gpu)
  # GPU4 remains reserved until every size is READY; one verification at a time.
  write(root/'PREPARATION_STATE.json',dict(at=now(),script='test_initial_repeat.py',gpu=4,status='waiting_reference_verification_device'))
  with (MASTER/'reference_gpu4.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);run('test_initial_repeat.py',4)
  run('freeze.py',gpu)
  write(root/'PREPARATION_STATE.json',dict(at=now(),status='passed'))
 except Exception:
  write(root/'PREPARATION_FAILED.json',dict(at=now(),status='failed',traceback=traceback.format_exc()));raise
 return key
def main():
 assert not (MASTER/'PREPARATION_COMPLETE.json').exists()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(one,pair) for pair in [('08b',7),('2b',1),('4b',2),('9b',3)]]
  for f in concurrent.futures.as_completed(futures):print('READY',f.result(),flush=True)
 write(MASTER/'PREPARATION_COMPLETE.json',dict(at=now(),status='passed'))
if __name__=='__main__':main()
