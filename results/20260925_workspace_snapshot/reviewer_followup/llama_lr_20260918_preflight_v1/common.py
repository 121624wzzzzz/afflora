import hashlib, json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = '/home/wz/anaconda3/envs/torch24/bin/python'
MODELS = ['llama32_3b_base','llama31_8b_base']
TOKEN_BOS_ID = 128000
TOKEN_EOS_ID = 128001
TASKS = ['cluener', 'wikisql']
ARMS = ['hidden','hidden_budget','hidden_both']
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

TYPES = ['address','book','company','game','government','movie','name','organization','position','scene']
TASK_SETTINGS = {'cluener':{'seed_start':6100,'microbatch':2,'max_new_tokens':512,'dev_batch':8,'test_batch':32},'wikisql':{'seed_start':7100,'microbatch':4,'max_new_tokens':256,'dev_batch':16,'test_batch':16,'confirm_batch':16}}

def prompt_ner(row):
    if row['task']=='toolace':
        return ('Select the appropriate tools and supply their arguments for the request. '
            'Return only a JSON array of objects with keys "name" and "arguments". '
            'Use the exact tool and parameter names from the definitions. '
            'If no tool applies, or required information is missing, return []. '
            'Do not execute tools or invent missing arguments.\n\nTools:\n'+canonical(row['tools'])+
            '\n\nRequest:\n'+row['text']+'\n\nAnswer:')
    return ('从文本中抽取所有命名实体。仅返回 JSON 数组，每个对象含 type、text、occurrence 三个字段。'
        'text 必须是原文中的连续片段。occurrence 表示这个片段在原文中的第几次出现，从 0 开始；'
        '只出现一次时填 0。没有实体则返回 []。'
        '类别：address=地址，book=书名，company=公司，game=游戏，government=政府，'
        'movie=电影，name=姓名，organization=组织机构，position=职位，scene=景点。'
        '\n\n文本：\n'+row['text']+'\n\n答案：')

def prompt_sql(row):
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

def prompt(row):
    return prompt_ner(row) if row['task']=='cluener' else prompt_sql(row)
