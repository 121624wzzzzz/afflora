from common import *
from transformers import AutoTokenizer
n=0
for m,c in read(HERE/'models.json').items():
 t=AutoTokenizer.from_pretrained(c['path'],local_files_only=True);eos=128001 if m.startswith('llama') else 151643
 for task in TASKS:
  ids={split:{r['id'] for r in rows(HERE/f'data/{task}_{split}.jsonl')} for split in ['train','dev','test']}
  assert not ids['train']&ids['dev'] and not ids['train']&ids['test'] and not ids['dev']&ids['test']
  for split in ['train','dev','test']:
   for row,token in zip(rows(HERE/f'data/{task}_{split}.jsonl'),read(HERE/f'tokens/{task}_{m}_{split}.json')):
    assert token['target_ids']==t.encode(canonical(row['target']),add_special_tokens=False)+[eos],(m,task,split,row['id'])
    n+=1
write(HERE/'TARGET_AUDIT.json',dict(status='passed',records=n,at=now(),native_eos_and_full_target_reencoding=True))
print('targets passed',n)
