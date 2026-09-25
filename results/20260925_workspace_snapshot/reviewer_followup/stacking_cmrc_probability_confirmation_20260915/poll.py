from collections import Counter
from pathlib import Path
import argparse
import re
from common import HERE,now,read
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
s=read(HERE/('smoke_state.json' if a.smoke else 'main_state.json'))
print(now(),s['phase'],dict(Counter(x['status'] for x in s['jobs'].values())))
for name,x in s['jobs'].items():
    if x['status']=='train':
        log=Path(x['log']).read_text(errors='replace')[-32000:];total=2 if a.smoke else 570
        steps=re.findall(r'(\d+)/'+str(total),log);losses=re.findall(r"\{'loss':[^\r\n]+",log)
        print('GPU',x['gpu'],name,'train',steps[-1] if steps else 'starting','/',total,losses[-1][:120] if losses else '')
    elif x['status']=='evaluate':
        out=HERE/('smoke_outputs' if a.smoke else 'outputs')/name;counts={}
        for n in ['internal_probability.jsonl','public_probability.jsonl','generation.jsonl']:
            if (out/n).exists():
                with (out/n).open() as f:counts[n]=sum(1 for _ in f)
        print('GPU',x['gpu'],name,'evaluate',counts)
    elif x['status']=='failed':print('FAILED',name,x.get('error'))
    elif x['status']!='complete':print('GPU',x['gpu'],name,x['status'])
