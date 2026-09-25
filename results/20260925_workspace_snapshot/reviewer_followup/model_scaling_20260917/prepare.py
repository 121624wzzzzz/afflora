"""Preserve exact old datasets/prompts and independently re-encode larger models."""
import sys,importlib.util,collections,math
from transformers import AutoTokenizer
from common import *
from scoring import score,aggregate

def official_engine(split):
    sys.path[:0]=[str(HERE/'raw/reference_deps'),str(HERE/'raw/official_wikisql')]
    from lib.dbengine import DBEngine
    return DBEngine(str(HERE/f'raw/wikisql/data/{split}.db'))

def main():
    audit={'at':now(),'counts':{},'lengths':{},'prompt_and_target_reuse_verified':0,'anchor_outputs_rescored':0}
    original={}
    for key in ['ner','sql']:
        sp=importlib.util.spec_from_file_location('old_'+key,HERE/f'provenance/{key}_common.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);original[key]=m
    tokenizers={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in read(HERE/'models.json').items()}
    anchor_root=HERE.parent/'posttraining_tasks_grounded_20260916';old_models=read(anchor_root/'models.json');anchor=old_models['qwen25_15b_base']
    manifest=read(anchor_root/'ARTIFACT_MANIFEST.json');assert sha(anchor_root/'models.json')==manifest['files']['models.json']['sha256']
    for f,h in anchor['files'].items():assert sha(f)==h,f
    anchor_tok=AutoTokenizer.from_pretrained(anchor['path'],local_files_only=True);write(HERE/'ANCHOR_MODEL_IDENTITY.json',{'status':'passed','model':anchor})
    for task in TASKS:
        sets=[]
        for split in ['train','dev','test']:
            data=rows(HERE/f'data/{task}_{split}.jsonl');assert len(data)==({'train':2048,'dev':200,'test':1343} if task=='cluener' else {'train':2048,'dev':256,'test':1024})[split]
            ts=read(HERE/f'anchors/tokens/{task}_{split}.json');assert [r['id'] for r in data]==[r['id'] for r in ts]
            for r,z in zip(data,ts):
                assert prompt(r)==original['ner' if task=='cluener' else 'sql'].prompt(r)
                assert anchor_tok.encode(prompt(r),add_special_tokens=False)==z['prompt_ids']
                assert anchor_tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]==z['target_ids'];audit['prompt_and_target_reuse_verified']+=1
            sets.append({' '.join(prompt(r).casefold().split()) for r in data});audit['counts'][task+'_'+split]=len(data)
            if task=='wikisql':
                engine=official_engine(split);from lib.query import Query
                for r in data:assert engine.execute_query(r['table_id'],Query.from_dict(r['target']),lower=True)==r['gold_execution']
            for model,tok in tokenizers.items():
                assert tok.eos_token_id==151643
                tokens=[{'id':r['id'],'prompt_ids':tok.encode(prompt(r),add_special_tokens=False),'target_ids':tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]} for r in data]
                assert all(len(r['prompt_ids'])+len(r['target_ids'])<=4096 for r in tokens)
                assert all(len(r['target_ids'])<=TASK_SETTINGS[task]['max_new_tokens'] for r in tokens)
                assert all(r['target_ids'].count(151643)==1 and 151643 not in r['prompt_ids'] and not ({151644,151645}&set(r['prompt_ids']+r['target_ids'])) for r in tokens)
                name=f'{task}_{model}_{split}';write(HERE/f'tokens/{name}.json',tokens)
                audit['lengths'][name]={'max_prompt':max(len(r['prompt_ids']) for r in tokens),'max_target':max(len(r['target_ids']) for r in tokens)}
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    for entry in read(HERE/'ANCHOR_INDEX.json'):
        root=HERE/entry['directory'];s=read(root/'SUMMARY.json');rs=rows(root/'responses.jsonl');data=rows(HERE/f"data/{entry['task']}_test.jsonl")
        assert [r['id'] for r in rs]==[r['id'] for r in data] and sha(root/'responses.jsonl')==s['responses_sha256']
        for r,gold in zip(rs,data):
            assert anchor_tok.decode(r['token_ids'],skip_special_tokens=False)==r['text']
            for k,v in score(r['text'],gold).items():assert r[k]==v,(entry['directory'],r['id'],k)
            audit['anchor_outputs_rescored']+=1
        got=aggregate(rs,entry['task']);assert got=={k:s[k] for k in got}
        init=read(root/'INITIALIZATION.json');assert init['preflight']['zero_residual_max_abs_error']==0
        if entry['arm']!='base':
            tr=read(root/'TRAINING.json');assert tr['steps']==64 and tr['examples']==2048 and tr['frozen_before']==tr['frozen_after'] and tr['reload_loss_error']==0
            order=read(root/'TRAIN_ORDER.json');training=rows(HERE/f"data/{entry['task']}_train.jsonl");assert order['ids']==[training[i]['id'] for i in order['indices']]
    audit['status']='passed';write(HERE/'DATA_AND_REUSE_AUDIT.json',audit);print(canonical(audit))
if __name__=='__main__':main()
