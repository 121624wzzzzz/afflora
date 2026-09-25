"""Wait for verified downloads, then run independent per-size technical admission."""
import os,json,subprocess,time,concurrent.futures
from pathlib import Path
PARENT=Path(__file__).resolve().parent
PYTHON='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
def one(pair):
 key,gpu=pair;p=PARENT/f'qwen35_{key}_20260919'
 while not (p/'MODEL_IDENTITY_AUDIT.json').exists():time.sleep(10)
 assert json.loads((p/'MODEL_IDENTITY_AUDIT.json').read_text())['status']=='passed'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_COMPILE_THREADS='1')
 for script,log in [('prepare.py','prepare.log'),('test_accelerated_kernel.py','accelerated_kernel.log'),('technical_probe.py','technical_probe.log'),('test_initial_repeat.py','initial_repeat_gate.log'),('freeze.py','freeze.log')]:
  print(json.dumps({'model':key,'starting':script,'gpu':gpu}),flush=True)
  with (p/log).open('w') as f:r=subprocess.run([PYTHON,'-u',str(p/script)],cwd=p,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps({'model':key,'finished':script,'returncode':r.returncode}),flush=True)
  if r.returncode:raise RuntimeError((key,script,r.returncode))
 return key
def main():
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(one,pair) for pair in [('08b',7),('2b',1),('9b',2)]]
  for f in concurrent.futures.as_completed(fs):print('ready',f.result(),flush=True)
if __name__=='__main__':main()
