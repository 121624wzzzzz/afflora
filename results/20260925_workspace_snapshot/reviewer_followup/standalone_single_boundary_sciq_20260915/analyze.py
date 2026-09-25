import numpy as np
from scipy.stats import t
from settings import *
from run import specs

def interval(values):
    x=np.array(values,dtype=float);se=x.std(ddof=1)/np.sqrt(len(x));mu=x.mean()
    return {'mean':float(mu),'seeds':x.tolist(),'ci95':(mu+np.array([-1,1])*t.ppf(.975,len(x)-1)*se).tolist(),
        'family8_ci95':(mu+np.array([-1,1])*t.ppf(1-.05/16,len(x)-1)*se).tolist()}

def main():
    assert read(HERE/'CONFIRMATION_COMPLETE.json')['jobs']==20
    assert read(HERE/'REFERENCE_COMPLETE.json')['jobs']==2
    references={s['model']:s for s in specs('reference')};result={'at':now(),'scope':'single-boundary-only on previously inspected SciQ, fresh fitted seeds','models':{},'primary_contrasts':[]}
    for model in read(HERE/'models.json'):
        basecp=Path(references[model]['checkpoint']);base=read(basecp/'test_metrics.json')
        basepred=[r for r in rows(basecp/'test_predictions.jsonl') if r['rotation']==0 and not r['ambiguous_gold']]
        basegen=[r for r in rows(basecp/'test_generation.jsonl') if not r['ambiguous_gold']]
        assert len(basepred)==len(basegen)==998 and [r['id'] for r in basepred]==[r['id'] for r in basegen]
        output={'base':base,'arms':{}}
        for arm in ARMS:
            runs=[s for s in specs('confirmation') if s['model']==model and s['arm']==arm]
            assert [s['seed'] for s in runs]==CONFIRMATION_SEEDS
            metrics=[read(Path(s['checkpoint'])/'test_metrics.json') for s in runs]
            train=[read(Path(s['checkpoint'])/'TRAINING.json') for s in runs]
            init=[read(Path(s['checkpoint'])/'INITIALIZATION.json') for s in runs]
            aggregate={group:{k:float(np.mean([r[group][k] for r in metrics])) for k in metrics[0][group] if k!='n'} for group in ['primary','generation','rotation_average']}
            aggregate['parameters']=init[0]['trainable_parameters'];aggregate['training_seconds']=float(np.mean([r['seconds'] for r in train]));aggregate['frozen_parameters_unchanged_all']=all(r['frozen_parameters_bitwise_unchanged'] for r in train)
            effects={k:[] for k in ['candidate_accuracy','strict_generation_accuracy']};vectors={k:[] for k in effects}
            changes={k:[] for k in effects}
            for s,metric in zip(runs,metrics):
                cp=Path(s['checkpoint']);pred=[r for r in rows(cp/'test_predictions.jsonl') if r['rotation']==0 and not r['ambiguous_gold']]
                gen=[r for r in rows(cp/'test_generation.jsonl') if not r['ambiguous_gold']]
                assert [r['id'] for r in pred]==[r['id'] for r in gen]==[r['id'] for r in basepred]
                for key,rr,bb,field in [('candidate_accuracy',pred,basepred,'correct'),('strict_generation_accuracy',gen,basegen,'strict_correct')]:
                    aa=np.array([r[field] for r in rr],dtype=int);b=np.array([r[field] for r in bb],dtype=int);v=100*(aa-b)
                    vectors[key].append(v);effects[key].append(float(v.mean()))
                    changes[key].append({'seed':s['seed'],'corrected':int(((aa==1)&(b==0)).sum()),'regressed':int(((aa==0)&(b==1)).sum())})
            for key in effects:
                seed_interval=interval(effects[key]);v=np.mean(vectors[key],axis=0);rng=np.random.default_rng(20260915);bs=[]
                for _ in range(50):
                    indices=rng.integers(0,len(v),size=(200,len(v)));bs.extend(v[indices].mean(1).tolist())
                c={'model':model,'arm':arm,'metric':key,'versus':'frozen_base','effect_pp':seed_interval,
                    'passes_positive_corrected_interval':seed_interval['family8_ci95'][0]>0,
                    'question_bootstrap95_conditioned_on_fitted_seeds':np.quantile(bs,[.025,.975]).tolist(),'per_seed_changes':changes[key]}
                result['primary_contrasts'].append(c)
            aggregate['passes_both_task_metrics']=all(c['passes_positive_corrected_interval'] for c in result['primary_contrasts'] if c['model']==model and c['arm']==arm)
            output['arms'][arm]=aggregate
        result['models'][model]=output
    # Only a descriptive larger-budget reference, kept distinct from standalone tests.
    old=Path(read(HERE/'REUSE_AUDIT.json')['source_root']);seal=read(old/'ARTIFACT_MANIFEST.json')['sha256']
    assert sha(old/'RESULTS.json')==seal[str(old/'RESULTS.json')]
    previous=read(old/'RESULTS.json')
    result['larger_budget_hidden_r8_reference']={m:previous['aggregates'][m]['none'] for m in result['models']}
    result['reference_note']='Earlier independently trained seeds2002–2006, same task and candidate scoring, 5.046M / 9.232M parameters. Not a paired new-seed or equal-budget comparison.'
    write(HERE/'RESULTS.json',result)
    for m,r in result['models'].items():
        print(m,'base',r['base']['primary'],r['base']['generation'])
        for arm,a in r['arms'].items():print(arm,a)
    for c in result['primary_contrasts']:print(c['model'],c['arm'],c['metric'],c['effect_pp'])

if __name__=='__main__':main()
