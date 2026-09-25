import time,subprocess,os
from common import *
while not (HERE/'STATE.json').exists() or read(HERE/'STATE.json')['stage'] in ['running','disk_guard_waiting']:time.sleep(60)
s=read(HERE/'STATE.json');assert s['stage']=='complete' and not s['failed'] and not s['pending'] and not s['active'],s['stage']
expected=read(HERE/'PLAN_COUNTS.json')['training_total'];assert len(s['done'])==expected
subprocess.run([PYTHON,'-B',str(HERE/'report.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
responses=official=0;manifest={}
for name in s['done']:
 a=read(HERE/'audits'/f'{name}.json');assert a['status']=='passed';cp=HERE/'checkpoints'/name;tr=read(cp/'TRAINING.json')
 assert sha(cp/'adapter.safetensors')==tr['adapter_sha256']
 assert (cp/'initial_adapter.safetensors').exists(),str(cp)
 for tag,h in a['summary_sha256'].items():
  ep=HERE/'evaluations'/name/tag;assert sha(ep/'SUMMARY.json')==h;assert sha(ep/'responses.jsonl')==read(ep/'SUMMARY.json')['responses_sha256']
 responses+=a['responses'];official+=a['official_sql_executions']
for directory in ['checkpoints','evaluations','audits','specs']:
 for f in (HERE/directory).rglob('*'):
  if f.is_file():manifest[str(f.relative_to(HERE))]=sha(f)
for f in HERE.glob('*.py'):manifest[f.name]=sha(f)
write(HERE/'OUTPUT_MANIFEST.json',dict(at=now(),files=manifest))
write(HERE/'FINAL_AUDIT.json',dict(at=now(),status='passed',runs=expected,responses=responses,official_sql_executions=official,manifest_sha256=sha(HERE/'OUTPUT_MANIFEST.json')))
