import sys
from common import *
from scoring import score
from transformers import AutoTokenizer

def official_engine(split):
    sys.path.insert(0,str(HERE/'raw/reference_deps'))
    sys.path.insert(0,str(HERE/'raw/official_wikisql'))
    from lib.dbengine import DBEngine
    return DBEngine(str(HERE/f'raw/wikisql/data/{split}.db'))

def main():
    assert not (HERE/'DATA_FROZEN.json').exists()
    root=HERE.parent;old=root/'downstream_transfer_20260917'
    assert sha(old/'ARTIFACT_MANIFEST.json')=='ddfcf8b6c43e39303d9900efeac54d68899bf23b51bfb59426a59637ffd1fdc6'
    manifest=read(old/'ARTIFACT_MANIFEST.json')['files'];sources={}
    previous=[];excluded_tables=set();excluded_ids=set();excluded_prompts=set()
    normalize=lambda r:' '.join(prompt(r).casefold().split())
    for p in sorted(root.rglob('wikisql_*.jsonl')):
        if p.parent.name!='data' or p.is_relative_to(HERE):continue
        data=rows(p);assert all(r['task']=='wikisql' for r in data)
        previous.append({'path':str(p),'sha256':sha(p),'n':len(data)})
        excluded_tables.update(r['table_id'] for r in data)
        excluded_ids.update(r['id'] for r in data)
        excluded_prompts.update(normalize(r) for r in data)
    assert previous
    cfg=read(HERE/'models.json')['llama31_8b_base']
    for p,h in cfg['files'].items():assert sha(p)==h,p
    tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
    audit={'at':now(),'prior_data_files':previous,'sources':sources,'model_files_verified':cfg['files'],'splits':{}}
    chosen_tables={};chosen_ids={};seen=set(excluded_prompts);reference_checks=0;mutation_checks=0
    for tag,source,n in [('dev','dev',1024),('confirm','test',2048)]:
        for filename in [f'{source}.jsonl',f'{source}.tables.jsonl']:
            p=old/f'raw/wikisql/data/{filename}';h=sha(p);assert manifest[str(p.relative_to(old))]['sha256']==h
            sources[str(p)]=h
        tables={r['id']:r for r in rows(old/f'raw/wikisql/data/{source}.tables.jsonl')}
        candidates=[]
        for i,r in enumerate(rows(old/f'raw/wikisql/data/{source}.jsonl')):
            t=tables[r['table_id']]
            item={'id':f'wikisql-{source}-{i}','task':'wikisql','source_split':source,'table_id':r['table_id'],
                  'question':r['question'],'header':t['header'],'types':t['types'],'target':r['sql']}
            if item['id'] in excluded_ids or item['table_id'] in excluded_tables or normalize(item) in excluded_prompts:continue
            candidates.append(item)
        candidates.sort(key=lambda r:htext('llama-lr-20260918-v1:'+r['id']))
        selected=[]
        for r in candidates:
            key=normalize(r)
            if key in seen:continue
            seen.add(key);selected.append(r)
            if len(selected)==n:break
        assert len(selected)==n,(tag,len(selected))
        engine=official_engine(source);from lib.query import Query
        for j,r in enumerate(selected):
            r['gold_execution']=engine.execute_query(r['table_id'],Query.from_dict(r['target']),lower=True)
            assert score(canonical(r['target']),r)['content_correct'];reference_checks+=1
            if j<128:
                q=r['target']
                for candidate in [dict(q,agg=(q['agg']+1)%6),dict(q,conds=[]),dict(q,sel=(q['sel']+1)%len(r['header']))]:
                    ours=score(canonical(candidate),r)
                    try:ref=engine.execute_query(r['table_id'],Query.from_dict(candidate),lower=True)==r['gold_execution']
                    except Exception:ref=False
                    assert ours['content_correct']==ref
                    assert ours['lf_correct']==(Query.from_dict(candidate)==Query.from_dict(q));mutation_checks+=1
                assert not score('{"sel":true,"agg":0,"conds":[]}',r)['content_correct']
                assert not score('{"sel":0,"sel":1,"agg":0,"conds":[]}',r)['content_correct']
        write_rows(HERE/f'data/wikisql_{tag}.jsonl',selected)
        tokens=[]
        for r in selected:
            text=prompt(r);ids=[TOKEN_BOS_ID]+tok.encode(text,add_special_tokens=False)
            assert ids==tok.encode(text,add_special_tokens=True)
            targets=tok.encode(canonical(r['target']),add_special_tokens=False)+[TOKEN_EOS_ID]
            assert len(ids)+len(targets)<=4096 and len(targets)<=256
            assert ids.count(TOKEN_BOS_ID)==1 and TOKEN_EOS_ID not in ids and targets.count(TOKEN_EOS_ID)==1
            assert not (set(tok.all_special_ids)-{TOKEN_BOS_ID})&set(ids)
            assert not (set(tok.all_special_ids)-{TOKEN_EOS_ID})&set(targets)
            tokens.append({'id':r['id'],'prompt_ids':ids,'target_ids':targets})
        write(HERE/f'tokens/wikisql_llama31_8b_base_{tag}.json',tokens)
        chosen_tables[tag]=set(r['table_id'] for r in selected);chosen_ids[tag]=set(r['id'] for r in selected)
        assert not chosen_tables[tag]&excluded_tables and not chosen_ids[tag]&excluded_ids
        audit['splits'][tag]={'source_split':source,'eligible_n':len(candidates),'selected_n':n,'tables':len(chosen_tables[tag]),
                              'max_prompt':max(len(r['prompt_ids']) for r in tokens),'max_target':max(len(r['target_ids']) for r in tokens)}
    assert not chosen_tables['dev']&chosen_tables['confirm']
    train=rows(HERE/'data/wikisql_train.jsonl');tokens=read(HERE/'tokens/wikisql_llama31_8b_base_train.json')
    assert len(train)==len(tokens)==2048
    engine=official_engine('train');from lib.query import Query
    for r,t in zip(train,tokens):
        assert r['id']==t['id'] and t['prompt_ids']==tok.encode(prompt(r),add_special_tokens=True)
        assert t['target_ids']==tok.encode(canonical(r['target']),add_special_tokens=False)+[TOKEN_EOS_ID]
        assert engine.execute_query(r['table_id'],Query.from_dict(r['target']),lower=True)==r['gold_execution'];reference_checks+=1
    audit.update(status='passed',gold_official_checks=reference_checks,sql_mutation_checks=mutation_checks,
                 excluded_prior_tables=len(excluded_tables),excluded_prior_ids=len(excluded_ids))
    write(HERE/'DATA_AUDIT.json',audit)
    files={str(p.relative_to(HERE)):sha(p) for directory in ['data','tokens','raw'] for p in (HERE/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    write(HERE/'DATA_FROZEN.json',{'at':now(),'files':files})
    print(canonical({'status':'passed','splits':audit['splits'],'gold_checks':reference_checks,'mutation_checks':mutation_checks}))
if __name__=='__main__':main()
