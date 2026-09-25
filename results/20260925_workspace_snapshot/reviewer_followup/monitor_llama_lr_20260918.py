"""Read-only concise monitor for the finite learning-rate follow-up."""
import json,collections
from pathlib import Path

p=Path(__file__).resolve().parent/'llama_lr_20260918'
def read(p):return json.loads(p.read_text())
s=read(p/'STATE.json');print(s['at'],s['phase'],s['counts'])
for j in s['jobs']:
    name=j['spec']['name']
    if j['state'] in ['running','auditing','failed']:
        r=p/'checkpoints'/name/'PROGRESS.json';progress=read(r) if r.exists() else {}
        evaluations=[]
        for q in (p/'evaluations'/name).glob('*/PROGRESS.json'):
            x=read(q);evaluations.append(f'{q.parent.name}:{x["done"]}/{x["total"]}')
        print(name,j['state'],'gpu',j.get('gpu'),'step',progress.get('step'),'eval',','.join(evaluations))
        if j['state']=='failed':
            for suffix in ['.log','.audit.log']:
                q=p/'logs'/(name+suffix)
                if q.exists():print(q.read_text()[-1600:])
if (p/'COMPATIBILITY.json').exists():print('compatibility',read(p/'COMPATIBILITY.json')['status'])
if (p/'SELECTION.json').exists():print('selection',read(p/'SELECTION.json')['selected'])
if (p/'SCHEDULER_COMPLETE.json').exists():print('completion',read(p/'SCHEDULER_COMPLETE.json'))
