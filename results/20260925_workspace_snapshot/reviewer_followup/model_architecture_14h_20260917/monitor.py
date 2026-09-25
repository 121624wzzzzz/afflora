import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
if not (ROOT/'STATE.json').exists():print('Preparation; no scheduler state yet.')
else:
 s=read(ROOT/'STATE.json');print(s['at'],s['counts'],'halted',s['halted_phases'])
 for key,j in s['jobs'].items():
  if j['status']!='running':continue
  path=ROOT/j['phase'];name=j['spec']['name'];parts=[];p=path/'checkpoints'/name/'PROGRESS.json'
  if p.exists():a=read(p);parts.append(f"step {a['step']}/{a['steps']}")
  for tag in ['smoke_dev','dev','test']:
   p=path/'evaluations'/name/tag/'PROGRESS.json'
   if p.exists():a=read(p);parts.append(f"{tag} {a['done']}/{a['total']}")
  print('GPU',j['gpu'],key,', '.join(parts) or 'initializing')
