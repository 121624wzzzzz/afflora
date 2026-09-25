"""Final chat gains versus matched adapters and the unadapted chat reference."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,stdev
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'chat'


def describe(values):
    assert len(values)==3
    m=mean(values);sd=stdev(values);half=4.302652729911275*sd/math.sqrt(3)
    return {'n':3,'values':values,'mean':m,'sd':sd,'ci95':[m-half,m+half]}


def prompt_interval(per_seed):
    # Resample the same prompt indices for all three fitted seed models.
    # This interval is conditional on these models; it does not replace seed CI.
    differences=np.asarray(per_seed,dtype=np.float64).mean(axis=0)
    generator=np.random.default_rng(0);samples=[]
    for _ in range(20):
        indices=generator.integers(0,len(differences),size=(500,len(differences)))
        samples.extend(differences[indices].mean(axis=1).tolist())
    return np.quantile(samples,[.025,.975]).tolist()


def validate_aggregates(ce, scores, responses):
    """Independently reconstruct reported totals from retained row-level records."""
    rows=ce['per_example'];tokens=sum(r['token_count'] for r in rows)
    assert ce['num_examples']==len(rows)==1000
    assert ce['supervised_tokens']==tokens
    assert math.isclose(ce['total_nll'],math.fsum(r['nll_sum'] for r in rows),rel_tol=1e-10,abs_tol=1e-8)
    assert math.isclose(ce['avg_ce'],ce['total_nll']/tokens,rel_tol=1e-10,abs_tol=1e-10)
    for row in rows:
        assert math.isclose(row['mean_ce'],row['nll_sum']/row['token_count'],rel_tol=1e-10,abs_tol=1e-10)
    if scores is None:return
    assert len(responses)==len(scores['per_example'])==scores['examples']==541
    assert [r['key'] for r in responses]==[r['key'] for r in scores['per_example']]
    assert math.isclose(scores['cap_hit_rate'],mean(int(r['hit_token_cap']) for r in responses),abs_tol=1e-12)
    assert math.isclose(scores['mean_generated_tokens'],mean(r['generated_tokens'] for r in responses),abs_tol=1e-12)
    for mode in ['strict','loose']:
        flags=[flag for row in scores['per_example'] for flag in row[mode+'_instructions']]
        assert len(flags)==scores['instructions']
        assert math.isclose(scores[mode+'_instruction_accuracy'],mean(flags),abs_tol=1e-12)
        assert math.isclose(scores[mode+'_prompt_accuracy'],mean(r[mode] for r in scores['per_example']),abs_tol=1e-12)
        valid=[r for r in scores['per_example'] if r['key'] not in [1122,1129]]
        assert scores['valid_prompt_sensitivity']['examples']==len(valid)==539
        assert math.isclose(scores['valid_prompt_sensitivity'][mode+'_prompt_accuracy'],mean(r[mode] for r in valid),abs_tol=1e-12)


def main():
    audit=json.loads((ROOT/'FINAL_AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['new_training_cells']==24 and audit['reference_cells']==2
    state=json.loads((ROOT/'state.json').read_text())
    obs={};sources=[ROOT/'FINAL_AUDIT.json',ROOT/'manifest.json',Path(__file__).resolve()]
    for j in state['matrix']+state['references']:
        ce_path=ROOT/'reports'/f"{j['name']}.test.json"
        gen_path=ROOT/'ifeval'/j['name']/'scores.json'
        ce=json.loads(ce_path.read_text());g=json.loads(gen_path.read_text())
        response_path=gen_path.parent/'responses.jsonl'
        assert hashlib.sha256(response_path.read_bytes()).hexdigest()==g['responses_sha256']
        validate_aggregates(ce,g,[json.loads(line) for line in response_path.read_text().splitlines()])
        dev_path=ROOT/'reports'/f"{j['name']}.dev.json"
        validate_aggregates(json.loads(dev_path.read_text()),None,None)
        arm='reference' if j.get('reference') else j['placement']
        obs[j['model'],arm,j['seed']]={'ce':ce['avg_ce'],'strict':g['strict_prompt_accuracy'],
            'valid_strict':g['valid_prompt_sensitivity']['strict_prompt_accuracy'],
            'cap':g['cap_hit_rate'],'length':g['mean_generated_tokens'],'details':g['per_example']}
        sources.extend([ce_path,dev_path,gen_path,response_path])
    result={'status':'complete','aggregate_audit':{'status':'passed','ce_reports':52,'ifeval_reports':26},
            'protocol_audit_exclusions':audit.get('documented_manifest_exclusions',[]),'references':{},'contrasts':{}}
    table=[];lines=['# Chat-model follow-up: full matched results','',
        'All 24 trained cells and both unadapted chat references passed the final audit. Hidden rank is 8; every trained arm uses seeds 42/43/44. All gains below are treatment minus control. Negative CE is favorable; positive accuracy is favorable.','',
        '| Model | Arm | Test CE | IFEval strict, % | Valid-539 strict, % | Cap hit, % | Mean generated tokens |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    if result['protocol_audit_exclusions']:
        lines[4:4]=['Audit scope: the original scheduler rejected the changing derived RESULTS.md, mistakenly included in its immutable-input manifest. A separate documented final audit excludes only that output, preserves the original manifest and error, and validates all other inputs and endpoints. See [the audit note](../REPORT_MANIFEST_NOTE.md) and FINAL_AUDIT.json.','']
    models=sorted({k[0] for k in obs})
    for model in models:
        ref=obs[model,'reference',None]
        result['references'][model]={k:v for k,v in ref.items() if k!='details'}
        for arm in ['reference','none','output','both','hidden_budget']:
            rows=[ref] if arm=='reference' else [obs[model,arm,s] for s in [42,43,44]]
            avg={k:mean(r[k] for r in rows) for k in ['ce','strict','valid_strict','cap','length']}
            lines.append(f"| {model} | {arm} | {avg['ce']:.7f} | {100*avg['strict']:.3f} | {100*avg['valid_strict']:.3f} | {100*avg['cap']:.3f} | {avg['length']:.2f} |")
        for a,b in [('output','none'),('both','none'),('both','hidden_budget'),('both','output'),('none','reference'),('both','reference')]:
            paired=[(obs[model,a,s],ref if b=='reference' else obs[model,b,s]) for s in [42,43,44]]
            stats={k:describe([left[k]-right[k] for left,right in paired]) for k in ['ce','strict','valid_strict','cap','length']}
            for metric,exclude in [('strict',False),('valid_strict',True)]:
                diffs=[]
                for left,right in paired:
                    assert [r['key'] for r in left['details']]==[r['key'] for r in right['details']]
                    diffs.append([int(l['strict'])-int(r['strict']) for l,r in zip(left['details'],right['details']) if not exclude or l['key'] not in [1122,1129]])
                stats[metric]['prompt_bootstrap_ci95_conditional_on_fitted_models']=prompt_interval(diffs)
            key=f'{model}: {a} - {b}';result['contrasts'][key]=stats
            for metric,s in stats.items():
                prompt=s.get('prompt_bootstrap_ci95_conditional_on_fitted_models',[None,None])
                table.append({'model':model,'treatment':a,'control':b,'metric':metric,'mean_delta':s['mean'],
                    'seed_sd':s['sd'],'seed_ci95_low':s['ci95'][0],'seed_ci95_high':s['ci95'][1],
                    'conditional_prompt_bootstrap_ci95_low':prompt[0],'conditional_prompt_bootstrap_ci95_high':prompt[1]})
    lines += ['','## Paired seed means','','| Contrast | CE difference | IFEval strict gain, pp | Valid-539 gain, pp |','| --- | ---: | ---: | ---: |']
    for key,r in result['contrasts'].items():
        lines.append(f"| {key} | {r['ce']['mean']:+.7f} | {100*r['strict']['mean']:+.3f} | {100*r['valid_strict']['mean']:+.3f} |")
    lines += ['','JSON/CSV report two distinct uncertainty calculations: nominal paired t 95% intervals over three training seeds, and 10,000 paired-prompt percentile bootstrap replicates conditional on the three fitted models. The latter averages seed differences per prompt and resamples identical prompt indices across all models. Neither includes multiple-comparison correction or establishes cross-task generalization.','',
        'Both checkpoints are tied, chat-ready models. All arms use the same fixed, untuned LR 5e-5, one epoch, greedy non-thinking generation, repetition penalty 1.0 and 1,024-token cap. An increment relative to hidden LoRA can still be below the unadapted reference; inspect both comparisons before making a practical-utility claim.','',
        'See MODEL_PROVENANCE.json for official model identity checks and the rejected local Qwen2.5 cache. P1 Base-model results remain separate and unchanged.','']
    (ROOT/'FINAL_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'FINAL_ANALYSIS.md').write_text('\n'.join(lines))
    with (ROOT/'final_contrasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    (ROOT/'ANALYSIS_SOURCES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(ROOT/'FINAL_ANALYSIS.md')


if __name__=='__main__':main()
