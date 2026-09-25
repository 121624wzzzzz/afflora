import hashlib, json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = '/home/wz/anaconda3/envs/torch24/bin/python'
MODELS = ['qwen3_06b_base', 'qwen25_15b_base']
TASKS = ['banking77', 'e2e_clean']
def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def write_rows(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(canonical(z)+'\n' for z in x))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def htext(s):return hashlib.sha256(s.encode()).hexdigest()

def prompt(row):
    if row['task']=='banking77':
        cats=read(HERE/'data/categories.json')
        labels='\n'.join(f'{i:02d} = '+c.replace('_',' ') for i,c in enumerate(cats))
        return ('Classify the banking customer request using the intent categories below. '
            'Return only the two-digit category code.\n\nCategories:\n'+labels+
            '\n\nCustomer request:\n'+row['text']+'\n\nCategory code:')
    assert row['task']=='e2e_clean'
    return ('Write a natural English description of the restaurant from the provided attributes. '
        'Include all provided attributes and do not invent additional facts. Return only the description.\n\nAttributes:\n'+
        '\n'.join(k+': '+v for k,v in sorted(row['attributes']))+'\n\nDescription:')
