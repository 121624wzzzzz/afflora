"""Read-only compact status; this file is outside the frozen study code."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent/'llama_transfer_20260918'
def read(p):return json.loads(p.read_text())
if not (R/'STATE.json').exists():print('State not yet written');raise SystemExit()
s=read(R/'STATE.json');print(s['at'],s['phase'],s['counts'])
for j in s['jobs']:
    if j['state'] not in ['running','auditing','failed']:continue
    spec=j['spec'];name=spec['name'];progress={}
    p=R/'checkpoints'/name/'PROGRESS.json'
    if p.exists():
        tr=read(p);progress.update(train_step=tr['step'],train_total=tr['steps'],loss=round(tr['loss'],4))
    for tag in ['smoke_dev','dev','test']:
        p=R/'evaluations'/name/tag/'PROGRESS.json'
        if p.exists():
            v=read(p);progress.update(eval=tag,done=v['done'],total=v['total'])
    print(j['state'],j.get('gpu'),name,progress)
    if j['state']=='failed':
        for suffix in ['.log','.audit.log']:
            p=R/'logs'/(name+suffix)
            if p.exists():print('\n'.join(p.read_text().splitlines()[-12:]))
if (R/'TEST_ANALYSIS.json').exists():
    for c in read(R/'TEST_ANALYSIS.json')['conditions']:
        if c['complete']:print('COMPLETE_BLOCK',c['task'],c['model'],{k:round(v,3) for k,v in c['means'].items()},[{k:v for k,v in cmp.items() if k in ['control','mean','positive_seeds','bonferroni8_95_ci']} for cmp in c['comparisons']])
