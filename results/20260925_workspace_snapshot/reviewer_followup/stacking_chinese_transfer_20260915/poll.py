from collections import Counter
from common import HERE, now, read

state=read(HERE/'main_state.json')
print(now(),'phase',state['phase'],dict(Counter(v['status'] for v in state['jobs'].values())))
for name,v in state['jobs'].items():
    if v['status']=='running':
        out=HERE/'outputs'/name
        counts={}
        for file in ['c3_likelihood.jsonl','cmrc_likelihood.jsonl','cmrc_generation.jsonl']:
            path=out/file
            if path.exists():
                with path.open() as f:counts[file]=sum(1 for _ in f)
        print('GPU',v['gpu'],name,counts)
    elif v['status']=='failed':print('FAILED',name,v.get('error'))

