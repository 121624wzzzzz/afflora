from collections import Counter
from pathlib import Path
import re
from common import HERE, now, read
s=read(HERE/'main_state.json');print(now(),s['phase'],dict(Counter(x['status'] for x in s['jobs'].values())))
for name,x in s['jobs'].items():
    if x['status']=='train':
        log=Path(x['log']).read_text(errors='replace')[-32000:]
        steps=re.findall(r'(\d+)/570',log);losses=re.findall(r"\{'loss':[^\r\n]+",log)
        print('GPU',x['gpu'],name,'train',steps[-1] if steps else 'starting','/570',losses[-1][:120] if losses else '')
    elif x['status']=='evaluate':
        out=HERE/'outputs'/name;counts={}
        for n in ['cmrc_likelihood.jsonl','cmrc_generation.jsonl']:
            if (out/n).exists():
                with (out/n).open() as f:counts[n]=sum(1 for _ in f)
        print('GPU',x['gpu'],name,'evaluate',counts)
    elif x['status']=='failed':print('FAILED',name,x.get('error'))
