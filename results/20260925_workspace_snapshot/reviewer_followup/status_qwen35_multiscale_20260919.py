"""Compact read-only progress snapshot; no partial effect claims."""
import json
from pathlib import Path
from datetime import datetime
PARENT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def main():
 master=PARENT/'qwen35_multiscale_20260919'
 if (master/'RETIREMENT.json').exists():
  print(json.dumps(dict(status='retired_incomplete',historical_state_is_stale=True,archives=read(master/'PARTIAL_ARCHIVES.json'),current_study=str(PARENT/'qwen35_fixed_multiscale_20260920')),ensure_ascii=False),flush=True);return
 out={'at':datetime.now().astimezone().isoformat(timespec='seconds'),'models':{}}
 for key,size in [('08b','0.8B'),('2b','2B'),('9b','9B')]:
  p=PARENT/f'qwen35_{key}_20260919';s=read(p/'STATE.json');v=dict(phase=s['phase'],counts=s['counts'],active=[])
  for j in s['jobs']:
   if j['state'] not in ['running','failed']:continue
   name=j['spec']['name'];q=p/'checkpoints'/name;progress=read(q/'PROGRESS.json') if (q/'PROGRESS.json').exists() else {}
   ev=[read(e) for e in (p/'evaluations'/name).glob('*/PROGRESS.json')]
   entry=dict(name=name,gpu=j.get('gpu'),step=progress.get('step'),evaluation=[dict(done=x['done'],total=x['total']) for x in ev])
   if j['state']=='failed':entry['error_tail']=(p/'logs'/f'{name}.log').read_text()[-1500:]
   v['active'].append(entry)
  v['download_verified']=(p/'MODEL_IDENTITY_AUDIT.json').exists()
  if not v['download_verified']:
   mp=PARENT.parents[1]/f'models/Qwen3.5-{size}-Base'
   files=list(mp.glob('*.range_*'))+list(mp.glob('*.safetensors'))
   v['downloaded_GB']=round(sum(f.stat().st_size for f in files)/1e9,2)
  if s['phase']=='waiting_ready':
   v['passed_preparation']=[n for n in ['DATA_AUDIT','MODEL_FORMULA_AUDIT','ACCELERATED_KERNEL_AUDIT','CHECKPOINT_RUNTIME_AUDIT','INITIAL_REPEAT_GATE'] if (p/(n+'.json')).exists() and read(p/(n+'.json')).get('status')=='passed']
   if (p/'INITIAL_REPEAT_PROGRESS.json').exists():v['repeat_cases']=len(read(p/'INITIAL_REPEAT_PROGRESS.json')['records'])
  out['models'][size]=v
 print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
