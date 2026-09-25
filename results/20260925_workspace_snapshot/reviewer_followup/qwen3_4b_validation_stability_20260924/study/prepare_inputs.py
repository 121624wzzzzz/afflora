from common import *
from plan import architecture
from scoring import score,aggregate
from prepare import official_engine
from transformers import AutoTokenizer
import torch,collections

def key(s):return tuple(s[k] for k in ['stage','model','task','method','placement','rank','bias','seed'])
def main():
 sources=read(HERE/'SOURCE.json');models=read(HERE/'models.json');files={};token_count=0
 for m,c in models.items():
  cfg=read(Path(c['path'])/'config.json');assert cfg['tie_word_embeddings'] is True
  for path,h in c['files'].items():assert sha(path)==h,path;files[path]=h
  tok=AutoTokenizer.from_pretrained(c['path'],local_files_only=True);eos=128001 if m.startswith('llama') else 151643
  assert cfg['eos_token_id']==eos
  for task in TASKS:
   split_ids=[]
   for split in ['train','dev','test']:
    dp=HERE/f'data/{task}_{split}.jsonl';tp=HERE/f'tokens/{task}_{m}_{split}.json';data=rows(dp);tokens=read(tp);assert len(data)==len(tokens)
    split_ids.append({r['id'] for r in data})
    for r,t in zip(data,tokens):
     assert r['id']==t['id'];expected=tok.encode(prompt(r),add_special_tokens=False)
     if m.startswith('llama'):expected=[128000]+expected
     assert t['prompt_ids']==expected and t['target_ids']==tok.encode(canonical(r['target']),add_special_tokens=False)+[eos];token_count+=1
    files[str(dp)]=sha(dp);files[str(tp)]=sha(tp)
   assert all(not split_ids[a]&split_ids[b] for a,b in [(0,1),(0,2),(1,2)])
  print('verified weights/native tokenizer',m,flush=True)
 write(HERE/'INPUT_AUDIT.json',dict(at=now(),status='passed',files=files,token_records_reencoded=token_count))

if __name__=='__main__':main()
