import os,subprocess,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;S=R/'study';PY='/home/wz/anaconda3/envs/torch24/bin/python';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
while not (S/'REUSE_AUDIT.json').exists():
 if (R/'VERIFY_FAILED').exists():raise RuntimeError('Input verification failed')
 time.sleep(15)
assert json.loads((S/'REUSE_AUDIT.json').read_text())['status']=='passed'
assert json.loads((S/'SHARED_ALGEBRA_AUDIT.json').read_text())['status']=='passed'
while not (S/'DATA_FROZEN.json').exists():time.sleep(5)
files=list(S.glob('*.py'))+list((S/'source').glob('*.py'))+[S/n for n in ['PROTOCOL.md','models.json','BUDGET_PLAN.json','FORMAL_JOBS.json','SMOKE_JOBS.json','REUSE_AUDIT.json','MODEL_IDENTITY_AUDIT.json','SHARED_ALGEBRA_AUDIT.json','SOURCE_REUSE.json']]
(S/'CODE_FROZEN.json').write_text(json.dumps(dict(files={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
(S/'READY.json').write_text(json.dumps(dict(status='passed'))+'\n')
with (R/'scheduler.log').open('w') as f:subprocess.run([PY,'-B',str(S/'scheduler.py')],env=env,cwd=S,stdout=f,stderr=subprocess.STDOUT,check=True)
