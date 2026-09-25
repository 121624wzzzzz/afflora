from collections import Counter
from shared import D,read,now
s=read(D/'main_state.json');print(now(),s['phase'],dict(Counter(j['status'] for j in s['jobs'].values())))
for n,j in s['jobs'].items():
    if j['status']=='running':
        print('GPU',j['gpu'],n,{p.stem:sum(1 for _ in p.open()) for p in (D/'outputs'/n).glob('*.jsonl')})
    elif j['status']=='failed':print(n,j)
