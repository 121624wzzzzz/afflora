import hashlib
import json
import re
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
from common import HERE, CHAT, now, prompt_text, read, rows, sha, write

TRANSFER=HERE.parent/'stacking_chinese_transfer_20260915'

def key(text):return hashlib.sha256(re.sub(r'\s+',' ',text).strip().encode()).hexdigest()
def dump(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)

def main():
    assert not (HERE/'manifest.json').exists()
    frozen=read(TRANSFER/'manifest.json')
    for p,h in frozen['local_sha256'].items():assert sha(p)==h,p
    source=rows(TRANSFER/'data/cmrc_train.jsonl');evaluation=rows(TRANSFER/'data/cmrc_eval.jsonl')
    eval_contexts={key(r['context']) for r in evaluation}
    rejected=[r['id'] for r in source if key(r['context']) in eval_contexts]
    eligible=[r for r in source if key(r['context']) not in eval_contexts]
    contexts=sorted({key(r['context']) for r in eligible},key=lambda c:hashlib.sha256(('20260915:'+c).encode()).hexdigest())
    dev_contexts=set(contexts[:240])
    train=[r for r in eligible if key(r['context']) not in dev_contexts]
    dev=[r for r in eligible if key(r['context']) in dev_contexts]
    for split,data in [('train',train),('dev',dev)]:
        converted=[]
        for i,r in enumerate(data):
            assert len(r['answers'])==1 and r['answers'][0]
            converted.append({'record_id':r['id'],'source_file':'official_cmrc_train','source_index':i,
                'context_hash':key(r['context']),'conversations':[{'role':'user','content':prompt_text('cmrc',r)},
                    {'role':'assistant','content':r['answers'][0]}]})
        dump(HERE/f'data/{split}.jsonl',converted)
        dump(HERE/f'data/cmrc_{split}.jsonl',data)
    dump(HERE/'data/cmrc_eval.jsonl',evaluation)
    assert {key(r['context']) for r in train}.isdisjoint({key(r['context']) for r in dev}|eval_contexts)
    sys.path.insert(0,str(HERE/'source/corrected_sft_experiment'))
    from data_pipeline import tokenize_conversation
    from transformers import AutoTokenizer
    lengths={};base_files={}
    for model,desc in read(CHAT/'manifest.json')['models'].items():
        for p,h in desc['files'].items():assert sha(p)==h,p;base_files[p]=h
        tok=AutoTokenizer.from_pretrained(desc['path'],use_fast=True)
        lengths[model]={}
        for split in ['train','dev']:
            rs=rows(HERE/f'data/{split}.jsonl');encoded=[tokenize_conversation(r,tok,100000) for r in rs]
            maximum=max(len(x['input_ids']) for x in encoded);assert maximum<=2048
            counts=[sum(t!=-100 for t in x['labels']) for x in encoded]
            lengths[model][split]={'rows':len(rs),'max_tokens':maximum,'supervised_tokens':sum(counts),'minimum_supervised_tokens':min(counts)}
    provenance={'created_at':now(),'source_commit':read(TRANSFER/'DATA_PROVENANCE.json')['repositories']['cmrc2018']['commit'],
        'source_sha256':{str(TRANSFER/f'data/cmrc_{s}.jsonl'):sha(TRANSFER/f'data/cmrc_{s}.jsonl') for s in ['train','eval']},
        'split_rule':'240 training context-hash groups with smallest SHA256(20260915:context_hash) form internal dev; remaining train. Original order retained.',
        'train_rows':len(train),'dev_rows':len(dev),'train_contexts':len(contexts)-240,'dev_contexts':240,
        'public_dev_eval_rows':len(evaluation),'removed_train_ids_with_eval_context_overlap':rejected,
        'all_context_groups_disjoint':True,'lengths':lengths,'base_model_sha256':base_files,
        'evaluation_exposure':'Public dev already evaluated for prior generic-SFT transfer; this is an explicitly sequential task-specific follow-up, not an untouched new test.'}
    write(HERE/'DATA_AND_MODEL_AUDIT.json',provenance)
    print({k:v for k,v in provenance.items() if k not in ['source_sha256','base_model_sha256']},flush=True)

if __name__=='__main__':main()
