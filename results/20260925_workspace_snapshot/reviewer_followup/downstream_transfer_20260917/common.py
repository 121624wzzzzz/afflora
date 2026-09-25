import hashlib, json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = '/home/wz/anaconda3/envs/torch24/bin/python'
MODELS = ['qwen3_06b_base', 'qwen25_15b_base']
TASKS = ['anli_r1', 'wikisql']
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
    if row['task']=='anli_r1':
        return ('Determine the relationship between the premise and hypothesis. '
            'A = entailment: the hypothesis must be true given the premise. '
            'B = neutral: the premise does not determine whether the hypothesis is true. '
            'C = contradiction: the hypothesis must be false given the premise. '
            'Return only A, B, or C.\n\nPremise:\n'+row['premise']+
            '\n\nHypothesis:\n'+row['hypothesis']+'\n\nAnswer:')
    assert row['task']=='wikisql'
    columns=[{'index':i,'name':n,'type':t} for i,(n,t) in enumerate(zip(row['header'],row['types']))]
    return ('Convert the question into a query over the table. Return only a JSON object '
        'with keys "sel", "agg", and "conds". "sel" is the selected column index. '
        '"agg" is 0=none, 1=MAX, 2=MIN, 3=COUNT, 4=SUM, 5=AVG. '
        '"conds" is a list of [column_index, operator, value] conditions combined with AND; '
        'operator is 0=equals, 1=greater than, 2=less than. Column indices start at 0. '
        'Use [] when there are no conditions.\n\nColumns:\n'+canonical(columns)+
        '\n\nQuestion:\n'+row['question']+'\n\nAnswer:')
