import hashlib, json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = '/home/wz/anaconda3/envs/torch24/bin/python'
MODELS = ['qwen3_06b_base', 'qwen25_15b_base']
TASKS = ['toolace', 'cluener']
TYPES = ['address','book','company','game','government','movie','name','organization','position','scene']
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
    if row['task']=='toolace':
        return ('Select the appropriate tools and supply their arguments for the request. '
            'Return only a JSON array of objects with keys "name" and "arguments". '
            'Use the exact tool and parameter names from the definitions. '
            'If no tool applies, or required information is missing, return []. '
            'Do not execute tools or invent missing arguments.\n\nTools:\n'+canonical(row['tools'])+
            '\n\nRequest:\n'+row['text']+'\n\nAnswer:')
    return ('从文本中抽取所有命名实体。仅返回 JSON 数组，每个对象含 type、text、start、end 四个字段。'
        'start 和 end 是原文从 0 开始的字符位置，两端均包含。没有实体则返回 []。'
        '类别：address=地址，book=书名，company=公司，game=游戏，government=政府，'
        'movie=电影，name=姓名，organization=组织机构，position=职位，scene=景点。'
        '\n\n文本：\n'+row['text']+'\n\n答案：')
