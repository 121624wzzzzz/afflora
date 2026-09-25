import hashlib, json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = '/home/wz/anaconda3/envs/torch24/bin/python'
MODELS = ['qwen25_05b_base','qwen25_15b_base','qwen25_3b_base','qwen25_7b_base','qwen3_06b_base','qwen3_17b_base','qwen3_4b_base','qwen3_8b_base']
TASKS = ['cluener', 'wikisql']
ARMS = ['hidden_shared8','hidden_shared16','hidden_shared32','hidden','hidden_budget','hidden_input','hidden_output','hidden_both','input','output','both']
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

MODELS=['qwen3_06b_base','qwen25_15b_base','llama32_1b_base','llama32_3b_base']
TASKS=['clinc150','emotion']
TASK_SETTINGS={t:{'seed_start':1107100 if t=='clinc150' else 1108100,'microbatch':2,'max_new_tokens':32,'dev_batch':16,'test_batch':32} for t in TASKS}
def prompt(row):
    labels=read(HERE/f"data/{row['task']}_labels.json")
    if row['task']=='clinc150':
        return ('Classify the user request into exactly one of the intent labels below. '
            'Return only a JSON string containing the exact label, including the double quotes. '
            'Do not answer or execute the request.\n\nIntent labels:\n'+', '.join(labels)+
            '\n\nUser request:\n'+row['text']+'\n\nIntent:')
    return ('Identify the emotion expressed by the writer of the text. Choose exactly one of the labels below. '
            'Return only a JSON string containing the exact label, including the double quotes. '
            'Do not answer the text.\n\nEmotion labels:\n'+', '.join(labels)+
            '\n\nText:\n'+row['text']+'\n\nEmotion:')
