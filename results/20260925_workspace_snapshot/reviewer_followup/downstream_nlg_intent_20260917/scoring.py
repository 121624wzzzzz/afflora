import sys,importlib.util,copy
from common import *
sys.path[:0]=[str(HERE/'vendor'),str(HERE/'raw/tgen')]
from sacrebleu.metrics import BLEU,CHRF
from tgen.data import DA

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ORIGINAL=module('slot_original',HERE/'raw/e2e_clean/slot_error.py')
CORRECTED=module('slot_corrected',HERE/'source/slot_error_corrected.py')
BLEU_METRIC=BLEU(lowercase=False,tokenize='13a',smooth_method='exp',effective_order=False)
CHRF_METRIC=CHRF(word_order=2)

def slots(text,row,impl):
    out,gold=impl.reclassify_mr(text,DA.parse_diligent_da(row['mr']))
    added,missing,wrong,repeated,_,_=impl.check_output(copy.deepcopy(gold),copy.deepcopy(out))
    return {'added':added,'missing':missing,'wrong':wrong,'repeated':repeated,'gold_slots':sum(sum(v.values()) for v in gold.values()),'exact':added+missing+wrong+repeated==0}

def score(text,row):
    refs=[[s] for s in row['references']]
    return {'bleu_stats':BLEU_METRIC._extract_corpus_statistics([text],refs)[0],
        'chrf_stats':CHRF_METRIC._extract_corpus_statistics([text],refs)[0],
        'slots':slots(text,row,CORRECTED),'slots_original':slots(text,row,ORIGINAL)}

def reference_streams(data):
    return [[r['references'][j] if j<len(r['references']) else None for r in data] for j in range(max(len(r['references']) for r in data))]

def aggregate(records,task):
    if task=='banking77':
        f1=[]
        for c in range(77):
            code=f'{c:02d}';tp=sum(r['predicted_code']==code and r['gold_code']==code for r in records)
            denom=sum(r['predicted_code']==code for r in records)+sum(r['gold_code']==code for r in records)
            f1.append(2*tp/denom if denom else 0)
        return {'n':len(records),'primary':100*sum(r['content_correct'] for r in records)/len(records),'macro_f1':100*sum(f1)/77}
    out={'n':len(records),'primary':BLEU_METRIC._aggregate_and_compute([r['bleu_stats'] for r in records]).score,
        'chrfpp':CHRF_METRIC._aggregate_and_compute([r['chrf_stats'] for r in records]).score}
    for key in ['slots','slots_original']:
        den=sum(r[key]['gold_slots'] for r in records)
        for kind in ['added','missing','wrong','repeated']:
            out[key+'_'+kind+'_pct']=100*sum(r[key][kind] for r in records)/den
        out[key+'_ser_pct']=sum(out[key+'_'+k+'_pct'] for k in ['added','missing','wrong','repeated'])
        out[key+'_exact_pct']=100*sum(r[key]['exact'] for r in records)/len(records)
    return out

def tests():
    rs=[{'references':['the food here is good','food here is very good'],'mr':'name[The Eagle], eatType[pub]'},
        {'references':['the place is very nice'],'mr':'name[The Eagle]'}]
    texts=['food here is very good','the place is nice'];ss=[score(t,r) for t,r in zip(texts,rs)];a=aggregate(ss,'e2e_clean');streams=reference_streams(rs)
    assert abs(a['primary']-BLEU_METRIC.corpus_score(texts,streams).score)<1e-10
    assert abs(a['chrfpp']-CHRF_METRIC.corpus_score(texts,streams).score)<1e-10
    assert abs(BLEU_METRIC.corpus_score(['the food here is good','the place is very nice'],streams).score-100)<1e-10
    assert BLEU_METRIC.corpus_score(['',''],streams).score==0
    row={'mr':'name[The Eagle], eatType[pub]'}
    assert slots('The Eagle is a pub.',row,CORRECTED)['exact']
    assert slots('The Eagle.',row,CORRECTED)['missing']==1
    assert slots('The Eagle is a restaurant.',row,CORRECTED)['wrong']==1
    assert slots('The Eagle is a pub. The Eagle is a pub.',row,CORRECTED)['repeated']==2
    write(HERE/'SCORER_TESTS.json',{'at':now(),'status':'passed','public_corpus_equals_sufficient_statistics':True,'multireference_gold_bleu':100,'empty_bleu':0,'slots_exact_missing_wrong_repeated_verified':True,'bleu_config':str(BLEU_METRIC.get_signature()),'chrf_config':str(CHRF_METRIC.get_signature())})
if __name__=='__main__':tests()
