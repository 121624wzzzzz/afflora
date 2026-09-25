import hashlib
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAT = HERE.parent / 'stacking_claim_checks_20260914/chat'
PYTHON = '/home/wz/anaconda3/envs/torch24/bin/python'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines()]
def now(): return datetime.now().astimezone().isoformat(timespec='seconds')

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); tmp.replace(path)

write_json = write

def checkpoint_hashes(cp):
    return {p.name: sha(p) for p in Path(cp).iterdir() if p.is_file() and p.name in {
        'run_args.json', 'adapter_config.json', 'adapter_model.safetensors',
        'affine_vocab_config.json', 'affine_vocab_adapter.safetensors'}}

def prompt_text(task, row):
    if task == 'cmrc':
        return '请根据下面的文章回答问题。答案必须是文章中的连续原文片段，只输出答案，不要解释。\n\n文章：\n'+row['context']+'\n\n问题：'+row['question']
    return ('请阅读下面的材料，从候选答案中选择最合适的一项。只输出该选项的完整内容，不要输出选项编号或解释。\n\n材料：\n'
            +row['context']+'\n\n问题：'+row['question']+'\n\n候选答案：\n'
            +'\n'.join(f'{i+1}. {v}' for i,v in enumerate(row['choices'])))

def prompt_ids(tokenizer, task, row):
    return list(tokenizer.apply_chat_template([{'role':'user','content':prompt_text(task,row)}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))
