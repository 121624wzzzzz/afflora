import hashlib
import json
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
PYTHON='/home/wz/anaconda3/envs/torch24/bin/python'
ARMS=['none','both','interior','hidden_budget']
LRS=[5e-5,2e-4]

def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def prompt(row,rotation=0):
    options=row['choices'][rotation:]+row['choices'][:rotation]
    return ('Answer the science question. Select the best option and respond with only '
            'its letter (A, B, C, or D).\n\nQuestion: '+row['question']+'\n\n'
            +'\n'.join(f'{c}. {v}' for c,v in zip('ABCD',options)))
def encode(tok,row,rotation=0):
    ids=list(tok.apply_chat_template([{'role':'user','content':prompt(row,rotation)}],
        tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
    gold=(row['gold']-rotation)%4
    return {'id':row['id'],'input_ids':ids,'gold':gold,'rotation':rotation,'ambiguous_gold':row['ambiguous_gold']}
