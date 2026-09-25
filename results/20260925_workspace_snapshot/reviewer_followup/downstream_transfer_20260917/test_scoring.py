"""Behavioral evaluation checks, including rejection and official agreement."""
import copy
from common import *
from scoring import *
from prepare import official_engine

def main():
    count=0
    for split in ['train','dev','test']:
        engine=official_engine(split)
        from lib.query import Query
        for r in rows(HERE/f'data/wikisql_{split}.jsonl')[:32]:
            q=r['target'];text=canonical(q);s=score(text,r)
            assert s['content_correct'] and s['lf_correct'] and s['strict_json']
            for candidate in [q,dict(q,agg=(q['agg']+1)%6),dict(q,conds=[]),dict(q,sel=(q['sel']+1)%len(r['header']))]:
                ours=score(canonical(candidate),r)
                try:ref=engine.execute_query(r['table_id'],Query.from_dict(candidate),lower=True)==r['gold_execution']
                except Exception:ref=False
                assert ours['content_correct']==ref
                assert ours['lf_correct']==(Query.from_dict(candidate)==Query.from_dict(q));count+=1
            assert score('```json\n'+text+'\n```',r)['content_correct']
            assert not score('```json\n'+text+'\n```',r)['strict_json']
            assert score(text+' explanation',r)['content_correct']
            assert not score(text+' explanation',r)['strict_json']
            for bad in ['garbage '+text,'{}',text[:-1],'{"sel":0,"sel":1,"agg":0,"conds":[]}',
                        '{"sel":true,"agg":0,"conds":[]}','{"sel":0,"agg":0,"conds":[[0,0,NaN]]}']:
                assert not score(bad,r)['content_correct']
    write(HERE/'SCORER_TESTS.json',{'at':now(),'status':'passed','official_comparisons':count})
    print('Scorer checks passed',count)
if __name__=='__main__':main()
