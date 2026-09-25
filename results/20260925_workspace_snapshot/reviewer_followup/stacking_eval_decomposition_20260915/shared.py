import hashlib
import json
from pathlib import Path
from datetime import datetime

D = Path(__file__).resolve().parent
B = D.parent / 'stacking_cmrc_task_sft_20260915'
A = D.parent / 'stacking_chinese_transfer_20260915'
PYTHON = '/home/wz/anaconda3/envs/torch24/bin/python'

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(p, v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def jobs(): return read(B/'manifest.json')['matrix']
