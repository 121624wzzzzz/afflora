"""Recheck selected data against author files and every token record before freeze."""
from transformers import AutoTokenizer
from common import *
def main():
    original={}
    for split,file in [('train','train-v2.0.json'),('test','dev-v2.0.json')]:
        for a in read(HERE/'raw/squad'/file)['data']:
            for p in a['paragraphs']:
                for q in p['qas']:original[q['id']]={'context':p['context'],'question':q['question'],'answers':[x['text'] for x in q['answers']],'is_impossible':q['is_impossible'],'title':a['title']}
    trec={}
    for split,file in [('official_train','train_5500.label'),('official_test','TREC_10.label')]:
        for i,line in enumerate((HERE/'raw/trec'/file).read_bytes().decode('latin-1').splitlines()):
            label,text=line.split(' ',1);trec[f'trec50_{split}_{i}']=(label,text)
    source_records=encoded=0;cats=read(HERE/'data/categories.json');assert len(cats)==50
    for task in TASKS:
        sets=[]
        for split in ['train','dev','test']:
            rs=rows(HERE/f'data/{task}_{split}.jsonl');assert len(rs)==({'train':2048,'dev':256,'test':500} if task=='trec50' else {'train':2048,'dev':256,'test':1024})[split]
            for r in rs:
                if task=='squad2':
                    assert all(r[k]==v for k,v in original[r['id']].items())
                    assert (r['source_split']=='official_dev')==(split=='test')
                    assert r['target']['answer'] in (r['answers'] or [''])
                else:
                    assert trec[r['id']]==(r['label'],r['text'])
                    assert cats[int(r['target'])]['label']==r['label']
                    assert ('official_test' in r['id'])==(split=='test')
                source_records+=1
            if task=='squad2':assert sum(r['is_impossible'] for r in rs)==len(rs)//2
            sets.append({r['cluster_id'] for r in rs})
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    for m,cfg in read(HERE/'models.json').items():
        tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
        assert read(HERE/f'tokens/{m}_codes.json')==[tok.encode(f'{i:02d}',add_special_tokens=False) for i in range(50)]
        for task in TASKS:
            for split in ['train','dev','test']:
                rs=rows(HERE/f'data/{task}_{split}.jsonl');ts=read(HERE/f'tokens/{task}_{m}_{split}.json');assert len(rs)==len(ts)
                for r,z in zip(rs,ts):
                    assert z=={'id':r['id'],'prompt_ids':tok.encode(prompt(r),add_special_tokens=False),'target_ids':tok.encode(target_text(r),add_special_tokens=False)+[151643]};encoded+=1
    write(HERE/'INDEPENDENT_DATA_AUDIT.json',{'at':now(),'status':'passed','author_records_verified':source_records,'token_records_reencoded':encoded,'split_clusters_disjoint':True,'invalid_outputs_not_used_for_preparation':True})
    print('Independent author-data and token audit passed',source_records,encoded)
if __name__=='__main__':main()
