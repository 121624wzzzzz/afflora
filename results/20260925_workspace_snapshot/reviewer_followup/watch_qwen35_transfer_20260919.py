"""Read-only monitor of the finite running experiment; exits on completion/failure."""
import json,time
from pathlib import Path
root=Path(__file__).resolve().parent/'qwen35_transfer_20260919'
def read(p):return json.loads(p.read_text())
seen=set();phase=None;tick=0
while True:
 d=read(root/'STATE.json');passed={j['spec']['name'] for j in d['jobs'] if j['state']=='passed'}
 new=passed-seen;out={'at':d['at'],'phase':d['phase'],'counts':d['counts']}
 if tick and new:
  out['new_completed']=[]
  for n in sorted(new):
   a=read(root/'audits'/f'{n}.json');scores={tag:read(root/'evaluations'/n/tag/'SUMMARY.json')['primary'] for tag in a['summary_sha256']}
   out['new_completed'].append({'name':n,'primary':scores})
 if new or phase!=d['phase'] or tick%2==0:print(json.dumps(out,ensure_ascii=False),flush=True)
 seen=passed;phase=d['phase'];tick+=1
 p=root/'SCHEDULER_COMPLETE.json'
 if p.exists() and read(p)['status']=='failed':print('SCHEDULER_FAILED',flush=True);raise SystemExit(1)
 if (root/'FINAL_AUDIT.json').exists():print('FINAL_AUDIT_READY',flush=True);break
 time.sleep(45)
