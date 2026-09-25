"""Report every calibration contrast without choosing favorable cells."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean,stdev
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE/'calibration'
MODELS=['qwen3_06b','qwen25_7b']
ARMS=['none','output','both','hidden_budget']
SEEDS=[42,43,44]
METRICS=['raw_ce','calibrated_ce','raw_content_ce','calibrated_content_ce','token_accuracy','content_token_accuracy']


def describe(values):
    if not values:return None
    result={'n':len(values),'values':values,'mean':mean(values)}
    if len(values)==3:
        sd=stdev(values);half=4.302652729911275*sd/math.sqrt(3)
        result.update(sd=sd,ci95=[mean(values)-half,mean(values)+half])
    return result


def main():
    observations={};files=[]
    manifest_hash=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    for model in MODELS:
        for arm in ARMS:
            for seed in SEEDS:
                name=f'{model}_{arm}_hr8_sd{seed}';p=ROOT/'reports'/f'{name}.json'
                if not p.exists():continue
                r=json.loads(p.read_text());assert r['protocol_sha256']==manifest_hash
                assert r['dev']['examples']==r['test']['examples']==1000
                observations[model,arm,seed]=r;files.append(p)
    summary={'completed':len(observations),'total':24,'models':{},'contrasts':{}}
    lines=['# Calibration control: complete dev selection and test evaluation','',
           f'{len(observations)}/24 checkpoints completed. Partial results are exploratory.','',
           'Every arm selects its scalar temperature independently on dev. All compared raw/calibrated CE values use the same contiguous FP32 kernel; legacy CE is independently replayed and recorded. Lower CE is better; higher token accuracy is better. This is teacher-forced accuracy, not free generation.','',
           '| Model | Arm | Seeds | Raw test CE | Calibrated test CE | Content raw CE | Content calibrated CE | Top-1 content accuracy | Dev temperatures |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    table=[]
    for model in MODELS:
        summary['models'][model]={}
        for arm in ARMS:
            values=[observations[model,arm,s] for s in SEEDS if (model,arm,s) in observations]
            if not values:continue
            data={metric:describe([r['metrics'][metric] for r in values]) for metric in METRICS}
            data['temperatures']=[r['selected_temperature'] for r in values]
            data['content_temperatures']=[r['selected_content_temperature'] for r in values]
            summary['models'][model][arm]=data
            seeds=[s for s in SEEDS if (model,arm,s) in observations]
            lines.append(f"| {model} | {arm} | {','.join(map(str,seeds))} | {data['raw_ce']['mean']:.7f} | {data['calibrated_ce']['mean']:.7f} | {data['raw_content_ce']['mean']:.7f} | {data['calibrated_content_ce']['mean']:.7f} | {data['content_token_accuracy']['mean']:.7f} | {data['temperatures']} |")
        for treatment,control in [('output','none'),('both','none'),('both','hidden_budget'),('both','output')]:
            paired=[s for s in SEEDS if (model,treatment,s) in observations and (model,control,s) in observations]
            if not paired:continue
            key=f'{model}: {treatment} - {control}'
            data={'seeds':paired}
            for metric in METRICS:
                values=[]
                for seed in paired:
                    a,b=observations[model,treatment,seed],observations[model,control,seed]
                    for split in ['dev','test']:
                        assert [(r['record_id'],r['tokens'],r['content_tokens']) for r in a[split]['per_example']]==[(r['record_id'],r['tokens'],r['content_tokens']) for r in b[split]['per_example']]
                    values.append(a['metrics'][metric]-b['metrics'][metric])
                data[metric]=describe(values)
                table.append({'model':model,'treatment':treatment,'control':control,'metric':metric,
                              'seeds':','.join(map(str,paired)), 'mean_delta':data[metric]['mean'],
                              'sd':data[metric].get('sd'), 'ci95_low':data[metric].get('ci95',[None,None])[0],
                              'ci95_high':data[metric].get('ci95',[None,None])[1]})
            data['raw_treatment_minus_calibrated_control_ce']=describe([
                observations[model,treatment,s]['metrics']['raw_ce']-observations[model,control,s]['metrics']['calibrated_ce'] for s in paired])
            summary['contrasts'][key]=data
    summary['precision']={'maximum_legacy_replay_per_example_ce_error':max((r[x]['replay_max_per_example_ce_error'] for r in observations.values() for x in ['dev','test']),default=None),
                          'uniform_minus_legacy_ce_range':([min(r['test']['uniform_kernel_minus_legacy_ce'] for r in observations.values()),max(r['test']['uniform_kernel_minus_legacy_ce'] for r in observations.values())] if observations else None),
                          'edge_temperature_choices':[r['name'] for r in observations.values() if r['grid_edge_selected'] or r['content_grid_edge_selected']]}
    lines += ['','## Paired differences (treatment minus control)','','| Contrast | Raw CE | Calibrated CE | Content calibrated CE | Content token accuracy, pp |','| --- | ---: | ---: | ---: | ---: |']
    for key,r in summary['contrasts'].items():
        lines.append(f"| {key} (n={len(r['seeds'])}) | {r['raw_ce']['mean']:+.7f} | {r['calibrated_ce']['mean']:+.7f} | {r['calibrated_content_ce']['mean']:+.7f} | {100*r['content_token_accuracy']['mean']:+.5f} |")
    lines += ['','Intervals in JSON/CSV are nominal paired t 95% intervals over three training seeds on fixed evaluation data, without multiple-comparison correction. A surviving CE difference rules out this finite-grid scalar control as a complete explanation; it does not prove semantic or free-generation improvement. A grid-edge optimum is explicitly flagged.','',
              'See PRECISION_NOTE.md for the numerical path audit and the archived preliminary implementations. No old P1 report was replaced.','']
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    if table:
        with (ROOT/'paired_contrasts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    source_files=files+[ROOT/'manifest.json',Path(__file__).resolve()]
    if (ROOT/'FINAL_AUDIT.json').exists():source_files.append(ROOT/'FINAL_AUDIT.json')
    (ROOT/'SUMMARY_SOURCES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},indent=2)+'\n')
    print('Calibration complete reports:',len(observations))


if __name__=='__main__':main()
