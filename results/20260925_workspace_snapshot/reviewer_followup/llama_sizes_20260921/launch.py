import os,json,subprocess,time,datetime,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/home/wz/anaconda3/envs/torch24/bin/python';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
def state(stage,**kw):(R/'LAUNCH_STATE.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),stage=stage,**kw),indent=2)+'\n')
try:
 state('waiting_verified_download')
 while not (R/'MODEL_1B.json').exists():time.sleep(30)
 state('preparing_and_reauditing')
 with (R/'prepare.log').open('w') as f:subprocess.run([PY,'-B',str(R/'prepare_study.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 state('running_study')
 with (R/'scheduler.log').open('w') as f:subprocess.run([PY,'-B',str(R/'study/scheduler.py')],cwd=R/'study',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 state('complete',audit=json.loads((R/'study/FINAL_AUDIT.json').read_text())['status'])
except Exception:
 state('failed',error=traceback.format_exc());raise
