import hashlib,json
from datetime import datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
PYTHON='/home/wz/anaconda3/envs/torch24/bin/python'
MODELS=['qwen25_05b_base','qwen25_15b_base','qwen25_3b_base','qwen25_7b_base','qwen3_06b_base','qwen3_17b_base','qwen3_4b_base','qwen3_8b_base']
TASKS=['trec50','squad2'];ARMS=['hidden','hidden_budget','hidden_both']
TASK_SETTINGS={'trec50':{'seed_start':9100,'microbatch':2,'max_new_tokens':0,'dev_batch':8,'test_batch':8},'squad2':{'seed_start':10100,'microbatch':2,'max_new_tokens':128,'dev_batch':8,'test_batch':8}}
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def write_rows(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(canonical(z)+'\n' for z in x))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def htext(s):return hashlib.sha256(s.encode()).hexdigest()
def target_text(row):return row['target'] if row['task']=='trec50' else canonical(row['target'])
def prompt(row):
 if row['task']=='trec50':
  cats=read(HERE/'data/categories.json');labels='\n'.join(f'{i:02d} = '+c['label']+' ('+c['description']+')' for i,c in enumerate(cats))
  return ('Classify the type of answer requested by the question using the categories below. '
   'Return only the two-digit category code; do not answer the question itself.\n\nCategories:\n'+labels+'\n\nQuestion:\n'+row['text']+'\n\nCategory code:')
 assert row['task']=='squad2'
 return ('Answer the question using only the supplied passage. Return only a JSON object with the key "answer". '
  'If the passage supports an answer, copy the shortest complete answer span from the passage. '
  'If the passage does not contain the answer, return {"answer":""}. Do not use outside knowledge.\n\nPassage:\n'+row['context']+'\n\nQuestion:\n'+row['question']+'\n\nAnswer:')
