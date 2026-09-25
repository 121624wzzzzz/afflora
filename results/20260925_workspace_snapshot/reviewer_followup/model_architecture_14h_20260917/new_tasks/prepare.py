"""Construct new task splits without looking at any model outputs."""
import collections,unicodedata
from transformers import AutoTokenizer
from common import *
from scoring import normalize

DESCRIPTIONS={
 'ABBR:abb':'abbreviation','ABBR:exp':'expanded expression',
 'DESC:def':'definition','DESC:desc':'description','DESC:manner':'procedure or manner','DESC:reason':'reason',
 'ENTY:animal':'animal','ENTY:body':'body part','ENTY:color':'color','ENTY:cremat':'creative work or invention','ENTY:currency':'currency name','ENTY:dismed':'disease or medicine','ENTY:event':'event','ENTY:food':'food','ENTY:instru':'musical instrument','ENTY:lang':'language','ENTY:letter':'letter','ENTY:other':'other entity','ENTY:plant':'plant','ENTY:product':'product','ENTY:religion':'religion','ENTY:sport':'sport','ENTY:substance':'substance','ENTY:symbol':'symbol','ENTY:techmeth':'technique or method','ENTY:termeq':'equivalent term','ENTY:veh':'vehicle','ENTY:word':'word',
 'HUM:desc':'description of a person','HUM:gr':'group of people or organization','HUM:ind':'individual person','HUM:title':'title or position',
 'LOC:city':'city','LOC:country':'country','LOC:mount':'mountain','LOC:other':'other location','LOC:state':'state or province',
 'NUM:code':'numeric code','NUM:count':'count','NUM:date':'date','NUM:dist':'distance','NUM:money':'money amount','NUM:ord':'rank or order','NUM:other':'other numeric value','NUM:perc':'percentage','NUM:period':'duration','NUM:speed':'speed','NUM:temp':'temperature','NUM:volsize':'volume or size','NUM:weight':'weight'}
def key(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def rank(scope,x):return htext('v6-new-20260917|'+scope+'|'+x)
def round_robin(pools,n):
    taken=[];cats=sorted(pools)
    while len(taken)<n:
        progress=False
        for c in cats:
            if pools[c] and len(taken)<n:taken.append(pools[c].pop(0));progress=True
        assert progress,'Insufficient stratified examples'
    return taken
def main():
    assert not (HERE/'checkpoints').exists()
    audit={'at':now(),'trec50':{},'squad2':{},'lengths':{}}
    def trec(filename,split):
        out=[]
        for i,line in enumerate((HERE/'raw/trec'/filename).read_bytes().decode('latin-1').splitlines()):
            label,text=line.split(' ',1);out.append({'task':'trec50','id':f'trec50_{split}_{i}','label':label,'text':text,'source_index':i,'cluster_id':htext(key(text))})
        return out
    train=trec('train_5500.label','official_train');test=trec('TREC_10.label','official_test');cats=sorted({r['label'] for r in train});assert len(cats)==50 and set(cats)==set(DESCRIPTIONS)
    write(HERE/'data/categories.json',[{'label':c,'description':DESCRIPTIONS[c]} for c in cats]);assert len(test)==500
    blocked={key(r['text']) for r in test};seen=set();pool=[];removed=[]
    for r in train:
        k=key(r['text'])
        if k in blocked or k in seen:removed.append({'id':r['id'],'reason':'test_overlap' if k in blocked else 'duplicate_train'});continue
        seen.add(k);pool.append(r)
    by_class={c:sorted([r for r in pool if r['label']==c],key=lambda r:rank('trec_split',r['id'])) for c in cats}
    dev_pool={c:rs[:min(len(rs)//5,len(rs)-1)] for c,rs in by_class.items()}
    dev=round_robin(dev_pool,256);dev_ids={r['id'] for r in dev}
    train_pool={c:[r for r in rs if r['id'] not in dev_ids] for c,rs in by_class.items()}
    chosen=round_robin(train_pool,2048)
    for split,rs in [('train',chosen),('dev',dev),('test',test)]:
        for r in rs:r['target']=f'{cats.index(r["label"]):02d}'
        write_rows(HERE/f'data/trec50_{split}.jsonl',sorted(rs,key=lambda r:rank('trec_order',r['id'])))
    assert len({r['label'] for r in chosen})==50
    groups=[{key(r['text']) for r in rs} for rs in [chosen,dev,test]];assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    audit['trec50']={'source_train':len(train),'source_test':len(test),'removed_train':removed,'counts':{'train':len(chosen),'dev':len(dev),'test':len(test)},'train_per_class':dict(collections.Counter(r['label'] for r in chosen)),'test_unique_texts':len(blocked),'decoding':'latin-1 author file; official test order-independent full 500'}
    models=read(HERE/'models.json');toks={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in models.items()};reference=toks['qwen25_15b_base']
    flat={};excluded={}
    for source_split,file in [('train','train-v2.0.json'),('test','dev-v2.0.json')]:
        out=[];counts=collections.Counter()
        for article in read(HERE/'raw/squad'/file)['data']:
            title=article['title']
            for para in article['paragraphs']:
                context=para['context']
                for q in para['qas']:
                    answers=[a['text'] for a in q['answers']]
                    for a in q['answers']:assert context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text']
                    assert bool(answers)!=q['is_impossible']
                    r={'task':'squad2','id':q['id'],'title':title,'context':context,'question':q['question'],'answers':answers,'is_impossible':q['is_impossible'],'cluster_id':htext(key(context)),'source_split':'official_train' if source_split=='train' else 'official_dev'}
                    gold=[a for a in answers if normalize(a)] or [''];r['target']={'answer':gold[0]}
                    if len(reference.encode(prompt(r),add_special_tokens=False))>768:counts['prompt_over_768']+=1;continue
                    if any(len(reference.encode(canonical({'answer':a}),add_special_tokens=False))+1>128 for a in gold):counts['gold_over_generation_cap']+=1;continue
                    out.append(r)
        flat[source_split]=out;excluded[source_split]=dict(counts)
    blocked_contexts={r['cluster_id'] for r in flat['test']};unique={};duplicate=[]
    for r in flat['train']:
        k=(r['cluster_id'],key(r['question']))
        if r['cluster_id'] in blocked_contexts or k in unique:duplicate.append(r['id']);continue
        unique[k]=r
    split_pool={'train':[],'dev':[],'test':flat['test']}
    for r in unique.values():
        split='dev' if int(rank('squad_article_split',r['title'])[:8],16)%10==0 else 'train';split_pool[split].append(r)
    for split,n in [('train',2048),('dev',256),('test',1024)]:
        selected=[]
        for impossible in [False,True]:
            pool=sorted([r for r in split_pool[split] if r['is_impossible']==impossible],key=lambda r:rank('squad_select',r['id']));assert len(pool)>=n//2
            selected+=pool[:n//2]
        write_rows(HERE/f'data/squad2_{split}.jsonl',sorted(selected,key=lambda r:rank('squad_order',r['id'])))
    groups=[{r['cluster_id'] for r in rows(HERE/f'data/squad2_{s}.jsonl')} for s in ['train','dev','test']]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    titles=[{r['title'] for r in rows(HERE/f'data/squad2_{s}.jsonl')} for s in ['train','dev','test']];assert all(not titles[i]&titles[j] for i in range(3) for j in range(i))
    audit['squad2']={'eligibility_reference':'Qwen2.5-1.5B pinned tokenizer; prompt <=768 tokens and all normalized-nonempty gold targets <=128 including EOS; no truncation','eligible_counts':{s:len(rs) for s,rs in flat.items()},'excluded_length':excluded,'duplicate_or_test_context_train_removed':duplicate,'counts':{'train':2048,'dev':256,'test':1024},'answerable_unanswerable_balance':'exact 50/50 each split','article_groups':{s:len(titles[i]) for i,s in enumerate(['train','dev','test'])},'public_dev_as_heldout':True}
    for m,tok in toks.items():
        codes=[tok.encode(f'{i:02d}',add_special_tokens=False) for i in range(50)];assert set(map(len,codes))=={2} and len(set(map(tuple,codes)))==50;write(HERE/f'tokens/{m}_codes.json',codes)
        assert tok.eos_token_id==151643
        for task in TASKS:
            for split in ['train','dev','test']:
                out=[]
                for r in rows(HERE/f'data/{task}_{split}.jsonl'):
                    p=tok.encode(prompt(r),add_special_tokens=False);t=tok.encode(target_text(r),add_special_tokens=False)+[151643]
                    assert len(p)<=768 and len(t)<=128 and len(p)+len(t)<1024,(m,task,split,r['id'],len(p),len(t))
                    assert not ({151643,151644,151645}&set(p)) and t.count(151643)==1 and not ({151644,151645}&set(t))
                    out.append({'id':r['id'],'prompt_ids':p,'target_ids':t})
                name=f'{task}_{m}_{split}';write(HERE/f'tokens/{name}.json',out);audit['lengths'][name]={'n':len(out),'max_prompt':max(len(r['prompt_ids']) for r in out),'max_target':max(len(r['target_ids']) for r in out)}
    jobs=[];smokes=[]
    for task in TASKS:
        seed0=TASK_SETTINGS[task]['seed_start']
        for m in MODELS:
            for arm in ['base']+ARMS:
                for seed in ([seed0] if arm=='base' else range(seed0,seed0+3)):
                    spec={'name':f'{task}_{m}_'+('base' if arm=='base' else f'{arm}_s{seed}'),'task':task,'model':m,'arm':arm,'seed':seed,'lr':2e-4,'microbatch':2,'smoke':False}
                    jobs.append({'spec':spec,'reused':False,'priority':20 if m in ['qwen25_15b_base','qwen25_7b_base','qwen3_8b_base'] else 40})
            smokes.append({'spec':{'name':f'smoke_{task}_{m}','task':task,'model':m,'arm':'hidden_both' if task=='squad2' else 'hidden_budget','seed':seed0-1,'lr':2e-4,'microbatch':2,'smoke':True},'reused':False,'priority':0})
    write(HERE/'FORMAL_JOBS.json',jobs);write(HERE/'SMOKE_JOBS.json',smokes);write(HERE/'INVENTORY.json',jobs)
    audit['status']='passed';write(HERE/'DATA_AND_REUSE_AUDIT.json',audit);print(canonical(audit),flush=True)
if __name__=='__main__':main()
