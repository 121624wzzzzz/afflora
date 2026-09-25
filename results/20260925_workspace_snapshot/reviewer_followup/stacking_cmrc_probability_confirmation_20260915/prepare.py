import sys
from pathlib import Path
from common import HERE,CHAT,read,rows,sha,write,now,prompt_text
from experiment import MODELS
sys.path.insert(0,str(HERE/'source/corrected_sft_experiment'))
from data_pipeline import tokenize_conversation
from transformers import AutoTokenizer

def main():
    previous=HERE.parent/'stacking_cmrc_task_sft_20260915';diag=HERE.parent/'stacking_eval_decomposition_20260915'
    sources={**read(previous/'FINAL_AUDIT.json')['source_sha256'],**read(diag/'ARTIFACT_MANIFEST.json')['sha256']}
    for p,h in sources.items():assert sha(p)==h,p
    for p in (HERE/'source').rglob('*'):
        if p.is_file():assert sha(p)==sha(previous/'source'/p.relative_to(HERE/'source'))
    for n in ['train.py','budget.py','check_reload.py','common.py','cmrc_official_py3.py','DATA_AND_MODEL_AUDIT.json']:
        assert sha(HERE/n)==sha(previous/n),n
    for p in (HERE/'data').iterdir():
        if p.is_file():assert sha(p)==sha(previous/'data'/p.name),p
    print('Reused files, original model weights, and upstream artifacts verified',flush=True)
    counts={};cache_hashes={}
    for m,path in MODELS.items():
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);eos=tokenizer.encode('<|im_end|>',add_special_tokens=False);assert len(eos)==1
        stats={}
        for split,filename in [('internal_dev','dev.jsonl'),('public_dev','cmrc_eval.jsonl')]:
            data=rows(HERE/'data'/filename);cache=[]
            for row in data:
                public=split=='public_dev'
                text=prompt_text('cmrc',row) if public else row['conversations'][0]['content']
                refs=row['answers'] if public else [row['conversations'][-1]['content']]
                prefix=list(tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
                unique=[]
                for answer in dict.fromkeys(refs):
                    ids=tokenizer.encode(answer,add_special_tokens=False);assert ids and tokenizer.decode(ids)==answer
                    full=tokenize_conversation({'conversations':[{'role':'user','content':text},{'role':'assistant','content':answer}]},tokenizer,8192)
                    ix=[i for i,x in enumerate(full['labels']) if x!=-100]
                    assert full['input_ids'][:ix[0]]==prefix and [full['labels'][i] for i in ix]==ids+eos
                    unique.append({'answer':answer,'answer_ids':ids,'reference_indices':[i for i,r in enumerate(refs) if r==answer]})
                assert len(prefix)+max(len(x['answer_ids']) for x in unique)+1<=2048
                assert len(prefix)+256<=8192
                cache.append({'id':row['id'] if public else row['record_id'],'cluster':row['cluster'] if public else row['context_hash'],
                    'prompt_ids':prefix,'references':refs,'unique_answers':unique,'eos':eos[0]})
            target=HERE/'token_cache'/m/f'{split}.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
            import json
            target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cache));cache_hashes[str(target)]=sha(target)
            stats[split]={'questions':len(cache),'unique_answers':sum(len(x['unique_answers']) for x in cache),'max_prompt_tokens':max(len(x['prompt_ids']) for x in cache)}
        counts[m]=stats;print(m,stats,flush=True)
    assert all(x['public_dev']['questions']==3219 and x['public_dev']['unique_answers']==4188 for x in counts.values())
    write(HERE/'PREPARATION.json',{'status':'passed','checked_at':now(),'upstream_sources_sha256':sources,'cache_sha256':cache_hashes,
        'counts':counts,'training_source':'Exact copies of the previously audited CMRC training split and code. No old trained adapter reused.'})

if __name__=='__main__':main()
