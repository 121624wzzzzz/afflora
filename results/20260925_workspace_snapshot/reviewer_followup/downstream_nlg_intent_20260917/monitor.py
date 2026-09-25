"""Read-only compact progress view; not part of training/evaluation."""
from common import *

for stage in ['SMOKE','FORMAL']:
    p=HERE/f'{stage}_STATE.json'
    if not p.exists():continue
    s=read(p);print(stage,s['at'],'finished',len(s['finished']),'active',len(s['active']),'pending',len(s['pending']),'failed',s['failed'])
    for r in s['active']:
        cp=HERE/'checkpoints'/r['name'];progress=cp/'PROGRESS.json'
        desc=f"step {read(progress)['step']}/{read(progress)['steps']}" if progress.exists() else 'initializing'
        for split in ['dev','test']:
            path=HERE/'evaluations'/r['name']/split/'PROGRESS.json'
            if path.exists():
                v=read(path);desc+=f" {split} {v['done']}/{v['total']}"
        print(' GPU',r['gpu'],r['name'],desc)
if (HERE/'EXPERIMENT_COMPLETE.json').exists():print('COMPLETE',read(HERE/'EXPERIMENT_COMPLETE.json'))
