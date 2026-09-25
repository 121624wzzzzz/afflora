import importlib.util
from common import *
from scoring import *
def main():
    spec=importlib.util.spec_from_file_location('official_squad',HERE/'raw/squad/evaluate-v2.0.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    golds=checks=0
    for split in ['train','dev','test']:
        for r in rows(HERE/f'data/squad2_{split}.jsonl'):
            targets=[a for a in r['answers'] if official.normalize_answer(a)] or ['']
            for gold in targets:
                s=score(canonical({'answer':gold}),r);assert s['em']==1 and s['f1']==1 and s['schema_valid'];golds+=1
            for pred in [targets[0],'','unsupported answer',targets[0]+' extra',targets[0].upper()]:
                got=score(canonical({'answer':pred}),r)
                assert got['em']==max(official.compute_exact(g,pred) for g in targets)
                assert abs(got['f1']-max(official.compute_f1(g,pred) for g in targets))<1e-12;checks+=1
            assert score('invalid output',r)['f1']==0
            assert not score('{"answer":"","answer":"x"}',r)['schema_valid']
            assert not score('{"answer":null}',r)['schema_valid']
            assert not score('{"answer":NaN}',r)['schema_valid']
            assert not score('[]',r)['schema_valid']
    r={'task':'squad2','answers':[],'is_impossible':True,'context':'Some text.'}
    assert score('{"answer":""}',r)['f1']==1 and score('not json',r)['f1']==0
    assert score('```json\n{"answer":""}\n```',r)['f1']==1 and not score('```json\n{"answer":""}\n```',r)['strict_json']
    rs=[{'predicted_code':f'{i:02d}','gold_code':f'{i:02d}','content_correct':True} for i in range(50)]
    got=aggregate(rs,'trec50');assert got['primary']==got['macro_f1_all50']==got['coarse_accuracy']==100
    rs[0].update(predicted_code='01',content_correct=False);assert aggregate(rs,'trec50')['primary']==98
    write(HERE/'SCORER_TESTS.json',{'at':now(),'status':'passed','squad_gold_roundtrips':golds,'official_perturbation_comparisons':checks,'invalid_not_abstention':True,'trec_all_labels_and_error_check':True})
    print('Scorer checks passed',golds,checks)
if __name__=='__main__':main()
