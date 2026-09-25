import numpy as np
from scipy.stats import t
from settings import *

METRICS={'candidate_accuracy':('test_predictions.jsonl','correct'),
         'generated_answer_accuracy':('test_generation.jsonl','correct')}


def contrast(model,arm,left,right,metric,family_size):
    filename,field=METRICS[metric]
    reference=rows(right[0]/filename)
    vectors=[];changes=[]
    for i,path in enumerate(left):
        a=rows(path/filename);b=rows(right[i if len(right)>1 else 0]/filename)
        assert [r['id'] for r in a]==[r['id'] for r in b]==[r['id'] for r in reference]
        aa=np.array([r[field] for r in a if not r['ambiguous_gold']],dtype=int)
        bb=np.array([r[field] for r in b if not r['ambiguous_gold']],dtype=int)
        vectors.append(100*(aa-bb));changes.append({'seed':read(path/'test_metrics.json')['spec']['seed'],
            'corrected':int(((aa==1)&(bb==0)).sum()),'regressed':int(((aa==0)&(bb==1)).sum())})
    x=np.mean(vectors,axis=1);se=x.std(ddof=1)/np.sqrt(len(x));q=t.ppf(1-.05/(2*family_size),len(x)-1)
    ci=(x.mean()+np.array([-1,1])*q*se).tolist();v=np.mean(vectors,axis=0);rng=np.random.default_rng(20260916)
    bootstrap=np.concatenate([v[rng.integers(0,len(v),(1000,len(v)))].mean(1) for _ in range(10)])
    return {'model':model,'arm':arm,'metric':metric,'mean_delta_pp':float(x.mean()),'per_seed_delta_pp':x.tolist(),
            'family_size':family_size,'corrected_seed_ci95_pp':ci,'positive_corrected_seed_interval':ci[0]>0,
            'question_bootstrap95_pp':np.quantile(bootstrap,[.025,.975]).tolist(),'paired_changes':changes,
            'interpretation':'post hoc; seed CI conditional on fixed questions, question CI conditional on fitted seeds'}


def main():
    assert read(HERE/'REEVALUATION_AND_TUNING_COMPLETE.json')['jobs']==60
    assert read(HERE/'CONFIRMATION_COMPLETE.json')['jobs']==10
    reports=[(p.parent,read(p)) for p in (HERE/'evaluations').glob('*/test_metrics.json')]
    assert len(reports)==64,len(reports)
    result={'at':now(),'scope':'post hoc corrected SciQ task adaptation; no general-SFT or stacking inference',
            'models':{},'primary_contrasts':[],'efficiency_contrasts':[],'secondary_chat_contrasts':[]}
    for model in read(HERE/'models.json'):
        subset=[(p,r) for p,r in reports if r['spec']['model']==model]
        refs=[(p,r) for p,r in subset if r['spec']['arm']=='base'];assert len(refs)==1
        refpath,reference=refs[0]
        arms=ARMS if model in BASE_MODELS else ['input','output']
        output={'reference':reference,'arms':{}};paths={}
        for arm in arms:
            fitted=sorted([(p,r) for p,r in subset if r['spec']['arm']==arm],key=lambda x:x[1]['spec']['seed'])
            assert len(fitted)==5
            expected=CONFIRMATION_SEEDS if model in BASE_MODELS else list(range(3002,3007))
            assert [r['spec']['seed'] for _,r in fitted]==expected
            pp=[p for p,_ in fitted];paths[arm]=pp
            primary={k:float(np.mean([r['primary'][k] for _,r in fitted])) for k in reference['primary'] if k!='n'}
            generation={k:float(np.mean([r['generation'][k] for _,r in fitted])) for k,v in reference['generation'].items() if k!='n' and isinstance(v,(int,float))}
            output['arms'][arm]={'parameters':fitted[0][1]['parameters'],'primary':primary,'generation':generation,
                                 'runs':[str(p) for p in pp]}
            for metric in METRICS:
                family=PRIMARY_FAMILY_SIZE if model in BASE_MODELS else 8
                result['primary_contrasts' if model in BASE_MODELS else 'secondary_chat_contrasts'].append(
                    contrast(model,arm,pp,[refpath],metric,family))
        if model in BASE_MODELS:
            for metric in METRICS:result['efficiency_contrasts'].append(
                contrast(model,'input_minus_small_q',paths['input'],paths['small_q'],metric,EFFICIENCY_FAMILY_SIZE))
        result['models'][model]=output
    write(HERE/'RESULTS.json',result)
    for model,out in result['models'].items():
        print(model,'reference',out['reference']['primary']['candidate_accuracy'],out['reference']['generation']['answer_accuracy'])
        for arm,r in out['arms'].items():print(arm,r['parameters'],r['primary']['candidate_accuracy'],r['generation']['answer_accuracy'])
    for r in result['efficiency_contrasts']:print('EFFICIENCY',r['model'],r['metric'],r['mean_delta_pp'],r['corrected_seed_ci95_pp'])


if __name__=='__main__':main()
