import hashlib,json,random,shutil,unicodedata
from collections import Counter
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download
from transformers import AutoTokenizer
from common import *

def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).lower().split())
def main():
    assert not (HERE/'DATA_AUDIT.json').exists()
    models=read(HERE/'models.json');checked={}
    for m,cfg in models.items():
        for p,h in cfg['files'].items():
            got=sha(p);assert got==h,(p,got,h);checked[p]=got
    api=HfApi();info=api.dataset_info('allenai/sciq');revision=info.sha
    raw={};source={};(HERE/'raw').mkdir(exist_ok=True)
    for split in ['train','validation','test']:
        paths=[x.rfilename for x in info.siblings if x.rfilename.startswith('data/'+split+'-') and x.rfilename.endswith('.parquet')]
        assert len(paths)==1,paths
        p=Path(hf_hub_download('allenai/sciq',paths[0],repo_type='dataset',revision=revision))
        dest=HERE/'raw'/p.name;shutil.copy2(p,dest)
        source[split]={'upstream':paths[0],'sha256':sha(dest)}
        raw[split]=pd.read_parquet(dest).to_dict('records')
    assert {k:len(v) for k,v in raw.items()}=={'train':11679,'validation':1000,'test':1000}
    keys={k:{norm(x['question']) for x in v} for k,v in raw.items()}
    overlaps={f'{a}:{b}':len(keys[a]&keys[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    assert overlaps['validation:test']==0,overlaps
    removed=[];audits={};(HERE/'data').mkdir(exist_ok=True)
    for split,rr in raw.items():
        out=[];duplicate_options=[];ambiguous=[]
        for i,r in enumerate(rr):
            if split=='train' and norm(r['question']) in keys['validation']|keys['test']:
                removed.append(i);continue
            opts=[r['correct_answer']]+[r[f'distractor{j}'] for j in [1,2,3]]
            assert all(isinstance(x,str) and x.strip() for x in opts)
            if len({norm(x) for x in opts})<4:duplicate_options.append(i)
            gold_ambiguous=norm(opts[0]) in [norm(x) for x in opts[1:]]
            if gold_ambiguous:
                ambiguous.append(i)
                if split=='train':continue
            perm=list(range(4));sd=int(hashlib.sha256(('sciq-placement-20260915:'+split+':'+str(i)).encode()).hexdigest(),16)
            random.Random(sd).shuffle(perm)
            out.append({'id':f'{split}_{i}','question':r['question'],
                        'choices':[opts[j] for j in perm],'gold':perm.index(0),'ambiguous_gold':gold_ambiguous})
        dest=HERE/'data'/f'{split}.jsonl';dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
        audits[split]={'rows':len(out),'unique_questions':len({norm(x['question']) for x in out}),
                       'gold_positions':dict(Counter(x['gold'] for x in out)),
                       'duplicate_option_indices':duplicate_options,'ambiguous_gold_indices':ambiguous,'sha256':sha(dest)}
    token_audit={};(HERE/'tokens').mkdir(exist_ok=True)
    for m,cfg in models.items():
        tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
        labels=[tok.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(x)==1 for x in labels)
        label_ids=[x[0] for x in labels];eos=tok.convert_tokens_to_ids('<|im_end|>')
        token_audit[m]={'label_ids':label_ids,'eos_id':eos,'pad_id':tok.pad_token_id,'splits':{}}
        for split in raw:
            rr=rows(HERE/'data'/f'{split}.jsonl');out=[encode(tok,r) for r in rr]
            for r,item in zip(rr,out):
                # Native full serialization must agree exactly with the masked prefix.
                full=tok.apply_chat_template([{'role':'user','content':prompt(r)},
                    {'role':'assistant','content':'ABCD'[r['gold']]}],tokenize=True,return_dict=False,enable_thinking=False)
                expected=item['input_ids']+[label_ids[r['gold']],eos]
                assert full[:len(expected)]==expected,(m,split,r['id'])
                assert len(expected)<=512,(m,r['id'],len(expected))
            write(HERE/'tokens'/f'{m}_{split}.json',out)
            token_audit[m]['splits'][split]={'rows':len(out),'max_prompt_tokens':max(len(x['input_ids']) for x in out)}
        write(HERE/'tokens'/f'{m}_test_rotations.json',[encode(tok,r,j) for r in rows(HERE/'data/test.jsonl') for j in range(4)])
    write(HERE/'DATA_AUDIT.json',{'created_at':now(),'revision':revision,'source':source,'splits':audits,
         'normalized_question_overlap_before_filter':overlaps,'removed_training_indices':removed,
         'tokenization':token_audit,'model_files_verified':checked,'evaluation_labels_used_for_preflight_only':True})
    print(json.dumps({'status':'prepared','revision':revision,'splits':audits,'tokenization':token_audit},indent=2))

if __name__=='__main__':main()
