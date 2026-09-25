import csv,math
import numpy as np
from scipy.stats import t
from common import *
from run import specs

def interval(values,family=1):
    x=np.array(values,dtype=float);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    half=float(t.ppf(1-.05/(2*family),len(x)-1)*se)
    return {'mean':mean,'ci95':[mean-half,mean+half],'seed_deltas':list(values)}

def main():
    frozen=read(HERE/'FROZEN_PROTOCOL.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    matrix=specs('confirmation');results={};predictions={}
    for s in matrix:
        cp=Path(s['checkpoint']);assert (cp/'COMPLETE.json').exists()
        r=read(cp/'test_metrics.json');assert r['prediction_sha256']==sha(cp/'test_predictions.jsonl')
        assert r['checkpoint_sha256']==sha(cp/'adapter.safetensors')
        results[s['model'],s['arm'],s['seed']]=r
        predictions[s['model'],s['arm'],s['seed']]=[x for x in rows(cp/'test_predictions.jsonl') if x['rotation']==0 and not x['ambiguous_gold']]
    comparisons=[];aggregates={}
    for model in read(HERE/'models.json'):
        aggregates[model]={}
        for arm in ARMS:
            rr=[results[model,arm,s] for s in range(2002,2007)]
            aggregates[model][arm]={k:float(np.mean([x['primary'][k] for x in rr])) for k in rr[0]['primary'] if k!='n'}
            aggregates[model][arm]['rotation_accuracy']=float(np.mean([x['rotation_average']['accuracy'] for x in rr]))
            aggregates[model][arm]['all_four_correct']=float(np.mean([x['all_four_correct'] for x in rr]))
        for control in ['none','hidden_budget','interior']:
            primary=[results[model,'both',s]['primary']['accuracy']-results[model,control,s]['primary']['accuracy'] for s in range(2002,2007)]
            entry={'model':model,'contrast':'both-minus-'+control,'primary_nominal':interval(primary),'primary_adjusted':interval(primary,6),'supporting':{}}
            entry['passes_adjusted']=entry['primary_adjusted']['ci95'][0]>0
            entry['mean_reaches_0_5pp']=entry['primary_adjusted']['mean']>=.5
            for metric in ['candidate_nll','brier','margin','unrestricted_first_token_accuracy','valid_label_rate']:
                entry['supporting'][metric]=interval([results[model,'both',s]['primary'][metric]-results[model,control,s]['primary'][metric] for s in range(2002,2007)])
            entry['supporting']['rotation_accuracy']=interval([results[model,'both',s]['rotation_average']['accuracy']-results[model,control,s]['rotation_average']['accuracy'] for s in range(2002,2007)])
            entry['supporting']['full_1000_nominal_accuracy']=interval([results[model,'both',s]['full_1000_nominal_labels']['accuracy']-results[model,control,s]['full_1000_nominal_labels']['accuracy'] for s in range(2002,2007)])
            paired=[]
            for s in range(2002,2007):
                a=predictions[model,'both',s];b=predictions[model,control,s]
                assert [x['id'] for x in a]==[x['id'] for x in b]
                assert len(a)==998 and len(set(x['id'] for x in a))==998
                paired.append([int(x['correct'])-int(y['correct']) for x,y in zip(a,b)])
            per_question=np.array(paired).mean(0)*100;rng=np.random.default_rng(20260915)
            boot=[]
            for _ in range(10000):boot.append(float(rng.choice(per_question,len(per_question),replace=True).mean()))
            entry['question_bootstrap_conditional_on_fitted_seeds']={'ci95':np.quantile(boot,[.025,.975]).tolist(),'mean':float(per_question.mean())}
            comparisons.append(entry)
    outcome={'at':now(),'n_jobs':len(matrix),'seeds':list(range(2002,2007)),'aggregates':aggregates,'comparisons':comparisons,
        'model_stacking_pass':{m:all(c['passes_adjusted'] for c in comparisons if c['model']==m and c['contrast']!='both-minus-interior') for m in aggregates},
        'model_placement_pass':{m:all(c['passes_adjusted'] for c in comparisons if c['model']==m) for m in aggregates}}
    write(HERE/'RESULTS.json',outcome)
    lines=['# SciQ placement: prespecified confirmation','',f"40 fresh runs, five new seeds, primary 998 unambiguous test questions. Six-comparison corrected intervals.",'',
        '| Model | Arm | Accuracy % | Rotation average % | NLL |','|---|---|---:|---:|---:|']
    for m,aa in aggregates.items():
        for a,r in aa.items():lines.append(f"| {m} | {a} | {r['accuracy']:.4f} | {r['rotation_accuracy']:.4f} | {r['candidate_nll']:.6f} |")
    lines+=['','| Model | Contrast | Gain pp | Corrected 95% CI | Pass |','|---|---|---:|---|---|']
    for c in comparisons:
        x=c['primary_adjusted'];lines.append(f"| {c['model']} | {c['contrast']} | {x['mean']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {c['passes_adjusted']} |")
    lines+=['','Intervals across seeds condition on the fixed test set. Question bootstrap conditions on these fitted seeds. Secondary metrics cannot replace the primary decision. No open-ended generation claim.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
