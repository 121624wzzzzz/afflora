import collections,sys,tarfile,zipfile
from common import *
from scoring import score,execute

SALT='downstream-transfer-20260917'
def normalized(s):return ' '.join(s.casefold().split())
def key(r):
    return normalized(prompt(r))
def ordered(data):return sorted(data,key=lambda r:htext(SALT+':'+r['id']))

def official_engine(split):
    sys.path.insert(0,str(HERE/'raw/reference_deps'))
    sys.path.insert(0,str(HERE/'raw/official_wikisql'))
    from lib.dbengine import DBEngine
    return DBEngine(str(HERE/f'raw/wikisql/data/{split}.db'))

def main():
    from transformers import AutoTokenizer
    assert not (HERE/'DATA_FROZEN.json').exists()
    raw=HERE/'raw';source={};all_data={};audit={'at':now(),'excluded':[],'splits':{},'sql_reference_checks':0}
    with zipfile.ZipFile(raw/'anli.zip') as z:
        source['anli']={'url':'https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip','sha256':sha(raw/'anli.zip')}
        all_data['anli_r1']={}
        for split in ['train','dev','test']:
            data=[json.loads(l) for l in z.read(f'anli_v1.0/R1/{split}.jsonl').decode().splitlines()]
            all_data['anli_r1'][split]=[{'task':'anli_r1','id':r['uid'],'premise':r['context'],
                'hypothesis':r['hypothesis'],'target':{'e':'A','n':'B','c':'C'}[r['label']],'source_split':split} for r in data]
    sqlroot=raw/'wikisql';sqlroot.mkdir(exist_ok=True)
    with tarfile.open(raw/'wikisql.tar.bz2') as z:
        for member in z.getmembers():
            p=(sqlroot/member.name).resolve();assert p.is_relative_to(sqlroot.resolve())
            assert member.isdir() or member.isfile()
            if member.isfile():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.extractfile(member).read())
    source['wikisql']={'url':'https://raw.githubusercontent.com/salesforce/WikiSQL/cffb423077756d04c1bac5bcd45167c86903fbcb/data.tar.bz2',
        'revision':'cffb423077756d04c1bac5bcd45167c86903fbcb','sha256':sha(raw/'wikisql.tar.bz2')}
    all_data['wikisql']={};tables={}
    for split in ['train','dev','test']:
        tables[split]={r['id']:r for r in rows(sqlroot/f'data/{split}.tables.jsonl')}
        all_data['wikisql'][split]=[]
        for i,r in enumerate(rows(sqlroot/f'data/{split}.jsonl')):
            t=tables[split][r['table_id']]
            all_data['wikisql'][split].append({'id':f'wikisql-{split}-{i}','task':'wikisql','source_split':split,
                'table_id':r['table_id'],'question':r['question'],'header':t['header'],'types':t['types'],'target':r['sql']})
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not set(tables[a])&set(tables[b]),(a,b,'shared tables')
    tok={m:AutoTokenizer.from_pretrained(read(HERE/'models.json')[m]['path'],local_files_only=True) for m in MODELS}
    for m,t in tok.items():assert [len(t.encode(x,add_special_tokens=False)) for x in 'ABC']==[1,1,1]
    for task,partitions in all_data.items():
        audit['splits'][task]={};seen=set()
        heldout_premises={normalized(r['premise']) for s in ['dev','test'] for r in partitions[s]} if task=='anli_r1' else set()
        for split in ['test','dev','train']:
            kept=[]
            for r in ordered(partitions[split]):
                if task=='anli_r1' and split=='train' and normalized(r['premise']) in heldout_premises:
                    audit['excluded'].append({'id':r['id'],'reason':'premise_in_dev_or_test','split':split});continue
                k=key(r)
                if k in seen:audit['excluded'].append({'id':r['id'],'reason':'duplicate_input','split':split});continue
                seen.add(k);kept.append(r)
            n=2048 if split=='train' else ((256 if split=='dev' else 1024) if task=='wikisql' else len(kept))
            assert len(kept)>=n
            selected=kept[:n];lengths={};official=official_engine(split) if task=='wikisql' else None
            if official:
                from lib.query import Query
                for r in selected:
                    gold=official.execute_query(r['table_id'],Query.from_dict(r['target']),lower=True)
                    assert execute(r['target'],r)==gold,(r['id'],'execution mismatch')
                    r['gold_execution']=gold
                    assert score(canonical(r['target']),r)['content_correct']
                    assert score(canonical(r['target']),r)['lf_correct']
                    audit['sql_reference_checks']+=1
            write_rows(HERE/f'data/{task}_{split}.jsonl',selected)
            for m,t in tok.items():
                tokens=[{'id':r['id'],'prompt_ids':t.encode(prompt(r),add_special_tokens=False),
                    'target_ids':t.encode(r['target'] if task=='anli_r1' else canonical(r['target']),add_special_tokens=False)+[151643]} for r in selected]
                assert all(len(r['prompt_ids'])+len(r['target_ids'])<=4096 for r in tokens)
                assert all(len(r['target_ids'])<=256 for r in tokens)
                assert all(r['target_ids'].count(151643)==1 and 151643 not in r['prompt_ids'] and not ({151644,151645}&set(r['prompt_ids']+r['target_ids'])) for r in tokens)
                write(HERE/f'tokens/{task}_{m}_{split}.json',tokens)
                lengths[m]={'max_prompt':max(len(r['prompt_ids']) for r in tokens),'max_target':max(len(r['target_ids']) for r in tokens)}
            audit['splits'][task][split]={'original_n':len(partitions[split]),'eligible_n':len(kept),'selected_n':n,'lengths':lengths,
                'labels':dict(collections.Counter(r['target'] for r in selected)) if task=='anli_r1' else None}
    write(HERE/'DATA_AUDIT.json',audit);write(HERE/'DATA_SOURCES.json',source)
    files={str(p.relative_to(HERE)):sha(p) for folder in ['data','tokens'] for p in (HERE/folder).glob('*')}
    files.update({str(p.relative_to(HERE)):sha(p) for p in sqlroot.glob('data/*.db')})
    write(HERE/'DATA_FROZEN.json',{'at':now(),'files':files,'protocol_sha256':sha(HERE/'PROTOCOL.md')})
    print(canonical(audit['splits']),flush=True)
if __name__=='__main__':main()
