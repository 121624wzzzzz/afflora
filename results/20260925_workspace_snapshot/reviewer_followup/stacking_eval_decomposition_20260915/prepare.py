import sys
import numpy as np
from shared import D,B,A,read,rows,sha,write,now,jobs

def main():
    sources={}
    audit=read(B/'FINAL_AUDIT.json');assert audit['status']=='passed'
    for path,h in audit['source_sha256'].items():assert sha(path)==h,path;sources[path]=h
    for root in [A,B]:
        for path,h in read(root/'ARTIFACT_MANIFEST.json')['sha256'].items():assert sha(path)==h,path;sources[path]=h
    print('All reused artifacts and frozen model/data/source hashes verified',flush=True)
    sys.path.insert(0,str(B));import common as previous
    sys.path.insert(0,str(previous.CHAT/'source/corrected_sft_experiment'))
    from data_pipeline import tokenize_conversation
    from transformers import AutoTokenizer
    public=rows(B/'data/cmrc_eval.jsonl');dev=rows(B/'data/dev.jsonl')
    models={j['model']:read(__import__('pathlib').Path(j['checkpoint'])/'run_args.json')['model_path'] for j in jobs()}
    identity={}
    for model,path in models.items():
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        eos=tokenizer.encode('<|im_end|>',add_special_tokens=False);assert len(eos)==1
        counts={};bad=[]
        for split,data in [('internal_dev',dev),('public_dev',public)]:
            total=matches=0
            for row in data:
                answers=[row['conversations'][-1]['content']] if split=='internal_dev' else row['answers']
                prompt=row['conversations'][0]['content'] if split=='internal_dev' else previous.prompt_text('cmrc',row)
                prefix=list(tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
                for i,answer in enumerate(answers):
                    ids=tokenizer.encode(answer,add_special_tokens=False)
                    full=tokenize_conversation({'conversations':[{'role':'user','content':prompt},{'role':'assistant','content':answer}]},tokenizer,8192)
                    labeled=[k for k,x in enumerate(full['labels']) if x!=-100]
                    match=full['input_ids'][:labeled[0]]==prefix and [full['labels'][k] for k in labeled]==ids+eos
                    total+=1;matches+=match
                    if not match and len(bad)<10:bad.append({'split':split,'id':row.get('id',row.get('record_id')),'index':i,'answer':answer,'separate_tokens':ids,'template_tokens':[full['labels'][k] for k in labeled]})
            counts[split]={'items':total,'matching':matches}
        identity[model]={'counts':counts,'examples_of_mismatch':bad}
        print(model,counts,flush=True)
    weighting=[]
    for j in jobs():
        raw=rows(B/'outputs'/j['name']/'cmrc_likelihood.jsonl');old=read(B/'outputs'/j['name']/'cmrc_scores.json')
        by={r['id']:[] for r in public}
        for r in raw:by[r['row_id']].append(r)
        macro=[];unique=[];first=[]
        for r in public:
            v=by[r['id']];macro.append(np.mean([x['nll']/x['tokens'] for x in v]));first.append(v[0]['nll']/v[0]['tokens'])
            indices=[r['answers'].index(s) for s in dict.fromkeys(r['answers'])];unique.append(np.mean([v[i]['nll']/v[i]['tokens'] for i in indices]))
        m={'reference_macro_ce':float(np.mean(macro)),'token_micro_ce':sum(r['nll'] for r in raw)/sum(r['tokens'] for r in raw),
           'unique_reference_macro_ce':float(np.mean(unique)),'first_reference_macro_ce':float(np.mean(first))}
        assert abs(m['reference_macro_ce']-old['metrics']['answer_ce'])<1e-12
        weighting.append({**{k:j[k] for k in ['name','model','arm','seed']},**m})
    write(D/'PREPARATION.json',{'checked_at':now(),'status':'passed','source_sha256':sources,'template_identity':identity,
        'public_questions':len(public),'internal_questions':len(dev),'reference_unique_counts':{str(k):sum(len(set(r['answers']))==k for r in public) for k in [1,2,3]},'weighting_results':weighting})
    print('Saved PREPARATION.json',flush=True)

if __name__=='__main__':main()
