"""Deterministic dataset construction, prior to any model output."""
import csv,re,unicodedata,collections
from common import *
from transformers import AutoTokenizer

def key(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def rank(scope,s):return htext('v4-20260917|'+scope+'|'+s)
def main():
    audit={'at':now(),'banking77':{},'e2e_clean':{}}
    raw=HERE/'raw/banking77/banking_data';cats=sorted(read(raw/'categories.json'));assert len(cats)==77
    write(HERE/'data/categories.json',cats)
    test=list(csv.DictReader((raw/'test.csv').open()));train=list(csv.DictReader((raw/'train.csv').open()))
    blocked={key(r['text']) for r in test};seen=set();removed=[];kept=[]
    for i,r in enumerate(train):
        k=key(r['text'])
        if k in blocked or k in seen:removed.append({'index':i,'reason':'test_overlap' if k in blocked else 'train_duplicate'});continue
        seen.add(k);kept.append(dict(r,source_index=i))
    extra=set(sorted(cats,key=lambda c:rank('extra-class',c))[:46]);splits={'train':[],'dev':[],'test':[dict(r,source_index=i) for i,r in enumerate(test)]}
    for cat in cats:
        pool=sorted([r for r in kept if r['category']==cat],key=lambda r:rank('bank-split',key(r['text'])))
        splits['dev']+=pool[:5];splits['train']+=pool[5:5+26+(cat in extra)]
    for split,rs in splits.items():
        out=[{'task':'banking77','id':f"banking77_{'test' if split=='test' else 'official_train'}_{r['source_index']}",'text':r['text'],'category':r['category'],'target':f'{cats.index(r["category"]):02d}','cluster_id':htext(key(r['text']))} for r in rs]
        out=sorted(out,key=lambda r:rank('bank-final-order',r['id']));write_rows(HERE/f'data/banking77_{split}.jsonl',out)
    assert len(splits['train'])==2048 and len(splits['dev'])==385 and len(splits['test'])==3080
    groups=[{key(r['text']) for r in splits[s]} for s in ['train','dev','test']]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    audit['banking77']={'official_train':len(train),'official_test':len(test),'removed_from_train':removed,'test_unique_texts':len(blocked),'counts':{s:len(v) for s,v in splits.items()}}
    e2e={}
    for split,filename in [('train','train-fixed.no-ol.csv'),('dev','devel-fixed.no-ol.csv'),('test','test-fixed.csv')]:
        source=list(csv.DictReader((HERE/'raw/e2e_clean/cleaned-data'/filename).open()));g={}
        for i,r in enumerate(source):
            pairs=[(k.strip(),v.strip()) for k,v in re.findall(r'([^,\[\]]+)\[([^\]]*)\]',r['mr'])]
            assert pairs and ', '.join(k+'['+v+']' for k,v in pairs)==r['mr'],r['mr']
            attrs=sorted(pairs);c=canonical(attrs)
            if c not in g:g[c]={'task':'e2e_clean','id':'e2e_'+htext(c),'cluster_id':htext(c),'attributes':attrs,'mr':', '.join(k+'['+v+']' for k,v in sorted(pairs)),'references':[],'source_indices':[]}
            g[c]['references'].append(r['ref']);g[c]['source_indices'].append(i)
        for r in g.values():
            r['references']=sorted(set(r['references']));r['target']=min(r['references'],key=lambda ref:rank('e2e-train-reference',r['id']+'|'+ref))
        e2e[split]=g;audit['e2e_clean'][split]={'source_rows':len(source),'unique_mrs':len(g)}
    assert not(set(e2e['train'])&set(e2e['dev']) or set(e2e['train'])&set(e2e['test']) or set(e2e['dev'])&set(e2e['test']))
    for split,n in [('train',2048),('dev',256),('test',len(e2e['test']))]:
        out=sorted(e2e[split].values(),key=lambda r:rank('e2e-select',r['id']))[:n];write_rows(HERE/f'data/e2e_clean_{split}.jsonl',out);audit['e2e_clean'][split]['selected']=len(out)
    lengths={}
    for m,cfg in read(HERE/'models.json').items():
        tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
        codes=[tok.encode(f'{i:02d}',add_special_tokens=False) for i in range(77)];assert set(map(len,codes))=={2} and len(set(map(tuple,codes)))==77
        write(HERE/f'tokens/{m}_codes.json',codes)
        for task in TASKS:
            for split in ['train','dev','test']:
                out=[]
                for r in rows(HERE/f'data/{task}_{split}.jsonl'):
                    p=tok.encode(prompt(r),add_special_tokens=False);t=tok.encode(r['target'],add_special_tokens=False)+[tok.eos_token_id]
                    assert tok.eos_token_id==151643 and not ({151644,151645,151643}&set(p))
                    assert 151643 not in t[:-1] and len(p)+len(t)<4096 and len(t)<256
                    out.append({'id':r['id'],'prompt_ids':p,'target_ids':t})
                name=f'{task}_{m}_{split}';write(HERE/f'tokens/{name}.json',out);lengths[name]={'n':len(out),'max_prompt':max(len(r['prompt_ids']) for r in out),'max_target':max(len(r['target_ids']) for r in out)}
    audit['lengths']=lengths;write(HERE/'DATA_AUDIT.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
