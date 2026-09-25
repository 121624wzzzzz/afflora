"""Fixed five-seed confirmation; secondary intervals are descriptive and unadjusted."""
from pathlib import Path
import csv
import math
import numpy as np
from scipy.stats import t
from common import HERE,read,rows,sha,write,now
from experiment import MODELS,SEEDS,ARMS,jobs

PROB_METRICS=['answer_set_probability','answer_set_nll','content_micro_ce','content_macro_ce','total_micro_ce','eos_ce','content_top1','first_ce','first_top1','rest_ce','rest_top1']
GEN_METRICS=['em','f1','avg','generated_tokens','hit_token_cap']
CLUSTER_METRICS=['answer_set_probability','answer_set_nll','em','f1','avg']

def interval(values,family=1):
    v=np.asarray(values,dtype=float);assert len(v)==5 and np.isfinite(v).all()
    mean=float(v.mean());se=float(v.std(ddof=1)/math.sqrt(len(v)));critical=float(t.ppf(1-.05/(2*family),4))
    p=float(2*t.sf(abs(mean/se),4)) if se else (0. if mean else 1.)
    return {'mean':mean,'per_seed':dict(zip(map(str,SEEDS),v.tolist())),'standard_error':se,
        'ci':[mean-critical*se,mean+critical*se],'family':family,'df':4,'p_two_sided':p,
        'bonferroni_adjusted_p':min(1.,p*family),'critical_t':critical}

def main():
    assert read(HERE/'main_state.json')['phase']=='complete'
    manifest=read(HERE/'manifest.json');assert manifest['matrix']==jobs()
    endpoints={};per={};sources={str(HERE/'manifest.json'):sha(HERE/'manifest.json')}
    for j in jobs():
        folder=HERE/'outputs'/j['name'];complete=read(folder/'COMPLETE.json');assert complete['status']=='passed'
        for p,h in complete['files'].items():assert sha(p)==h,p;sources[p]=h
        endpoints[(j['model'],j['seed'],j['arm'])]=read(folder/'scores.json')
        prob=rows(folder/'public_probability.jsonl');gen=rows(folder/'generation.jsonl')
        assert len(prob)==len(gen)==3219 and [r['id'] for r in prob]==[r['id'] for r in gen]
        per[(j['model'],j['seed'],j['arm'])]={k:np.array([r[k] for r in (prob if k in PROB_METRICS else gen)]) for k in CLUSTER_METRICS}
    clusters=[r['cluster'] for r in prob];unique=sorted(set(clusters));ix={c:i for i,c in enumerate(unique)}
    labels=np.array([ix[c] for c in clusters]);counts=np.bincount(labels,minlength=len(unique));assert len(unique)==848
    rng=np.random.default_rng(20260915);weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000)
    denominators=weights@counts
    contrasts=[];primary=[]
    for model in MODELS:
        for control in ['none','hidden_budget']:
            for split,metrics in [('public',PROB_METRICS+GEN_METRICS),('internal',PROB_METRICS)]:
                for metric in metrics:
                    deltas=[endpoints[(model,s,'both')][split]['metrics'][metric]-endpoints[(model,s,control)][split]['metrics'][metric] for s in SEEDS]
                    is_primary=split=='public' and metric=='answer_set_probability'
                    item={'model':model,'control':control,'split':split,'metric':metric,'primary':is_primary,
                        'seed_nominal95':interval(deltas),'direction':'lower' if metric.endswith('_ce') or metric=='answer_set_nll' else 'higher'}
                    if is_primary:
                        item['seed_bonferroni_family4']=interval(deltas,4)
                        item['confirmed']=item['seed_bonferroni_family4']['ci'][0]>0;primary.append(item)
                    if split=='public' and metric in CLUSTER_METRICS:
                        delta=np.mean([per[(model,s,'both')][metric]-per[(model,s,control)][metric] for s in SEEDS],axis=0)
                        sums=np.bincount(labels,weights=delta,minlength=len(unique))
                        boot=(weights@sums)/denominators
                        assert abs(float(delta.mean())-item['seed_nominal95']['mean'])<1e-12
                        item['conditional_document_bootstrap95']={'ci':np.quantile(boot,[.025,.975]).tolist(),
                            'draws':10000,'seed':20260915,'clusters':848,'conditioning':'Five fitted model pairs fixed; resample normalized contexts, weight questions equally.'}
                    contrasts.append(item)
    model_confirmations={m:all(x['confirmed'] for x in primary if x['model']==m) for m in MODELS}
    result={'complete':True,'created_at':now(),'fresh_training_runs':30,'seeds':SEEDS,'primary_family':4,
        'joint_two_model_confirmation':all(model_confirmations.values()),'model_confirmations':model_confirmations,
        'primary_contrasts':primary,'contrasts':contrasts,'endpoints':[endpoints[(j['model'],j['seed'],j['arm'])] for j in jobs()],
        'limitations':['Same previously examined public dev; independent new training seeds, not a new unseen test set.',
            'Only primary probability intervals adjust family=4. All supporting intervals are descriptive.',
            'FP32 inference differs from old BF16 inference; old and new scores are not pooled.',
            'Reference probability covers given unique strings and im_end, not all correct paraphrases or calibration.',
            'Five seed pairs imply t-distribution assumptions; cluster intervals condition on fitted models.'],
        'sources_sha256':sources}
    write(HERE/'RESULTS.json',result)
    with (HERE/'paired_effects.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','control','split','metric','primary','mean','nominal_lo','nominal_hi','corrected_lo','corrected_hi']+[f'seed_{s}' for s in SEEDS])
        for r in contrasts:
            v=r['seed_nominal95'];adjusted=r.get('seed_bonferroni_family4',{}).get('ci',['',''])
            writer.writerow([r['model'],r['control'],r['split'],r['metric'],r['primary'],v['mean'],*v['ci'],*adjusted,*v['per_seed'].values()])
    lines=['# CMRC five-new-seed confirmation','',f'Completed {len(endpoints)} fresh training runs. Joint two-model confirmation: **{result["joint_two_model_confirmation"]}**.',
        '', 'Primary: mean unique complete-reference probability (percentage-point differences below). Fixed Bonferroni family=4; paired t df=4.',
        '', '| Model | Control | Mean Δ pp | Corrected CI pp | Seed deltas pp | Confirmed |','|---|---|---:|---|---|---|']
    for r in primary:
        v=r['seed_bonferroni_family4'];lines.append(f'| {r["model"]} | {r["control"]} | {100*v["mean"]:+.6f} | [{100*v["ci"][0]:+.6f}, {100*v["ci"][1]:+.6f}] | '+', '.join(f'{100*x:+.6f}' for x in v['per_seed'].values())+f' | {r["confirmed"]} |')
    lines+=['','All endpoint scores and all supporting contrasts are retained in RESULTS.json and paired_effects.csv.','',*['- '+x for x in result['limitations']]]
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
