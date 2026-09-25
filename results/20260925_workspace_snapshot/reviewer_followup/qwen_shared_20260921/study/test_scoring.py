from common import *
from scoring import score,aggregate
from prepare import official_engine

def main():
    ner=0;sql=0
    for split in ['train','dev','test']:
        for r in rows(HERE/f'data/cluener_{split}.jsonl'):
            s=score(canonical(r['target']),r);assert s['content_correct'] and aggregate([s],'cluener')['primary']==100;ner+=1
        engine=official_engine(split);from lib.query import Query
        for r in rows(HERE/f'data/wikisql_{split}.jsonl')[:32]:
            q=r['target']
            for candidate in [q,dict(q,agg=(q['agg']+1)%6),dict(q,conds=[]),dict(q,sel=(q['sel']+1)%len(r['header']))]:
                ours=score(canonical(candidate),r)
                try:ref=engine.execute_query(r['table_id'],Query.from_dict(candidate),lower=True)==r['gold_execution']
                except Exception:ref=False
                assert ours['content_correct']==ref
                assert ours['lf_correct']==(Query.from_dict(candidate)==Query.from_dict(q));sql+=1
            assert not score('{"sel":true,"agg":0,"conds":[]}',r)['content_correct']
            assert not score('{"sel":0,"sel":1,"agg":0,"conds":[]}',r)['content_correct']
            assert score('```json\n'+canonical(q)+'\n```',r)['content_correct']
    r={'task':'cluener','text':'张三看见张三','gold_spans':[{'type':'name','text':'张三','start':4,'end':5}]};good=[{'type':'name','text':'张三','occurrence':1}]
    assert score(canonical(good),r)['span_tp']==1 and score(canonical([dict(good[0],occurrence=0)]),r)['span_tp']==0
    s=score(canonical(good*2),r);assert s['span_tp']==1 and s['span_pred']==2
    assert not score(canonical([dict(good[0],occurrence=True)]),r)['schema_valid']
    assert aggregate([score('invalid',r)],'cluener')['primary']==0
    write(HERE/'SCORER_TESTS.json',{'at':now(),'status':'passed','ner_gold_roundtrips':ner,'sql_official_comparisons':sql,'negative_duplicate_occurrence_checks':True});print('scorer tests passed',ner,sql)
if __name__=='__main__':main()
