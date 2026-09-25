"""Exploratory descriptive decomposition; no new model runs or confirmatory tests."""
import math
from common import HERE,rows,read,sha,write,now
from experiment import MODELS,SEEDS,jobs

def main():
    assert read(HERE/'main_state.json')['phase']=='complete'
    data={};sources={}
    for j in jobs():
        root=HERE/'outputs'/j['name'];p=root/'public_probability.jsonl';g=root/'generation.jsonl'
        prob=rows(p);gen=rows(g);assert [r['id'] for r in prob]==[r['id'] for r in gen]
        data[(j['model'],j['seed'],j['arm'])]=list(zip(prob,gen));sources[str(p)]=sha(p);sources[str(g)]=sha(g)
    contrasts=[]
    for model in MODELS:
        for control in ['none','hidden_budget']:
            seeds=[];categories={k:[] for k in ['both_correct','both_wrong','corrected','regressed']}
            for seed in SEEDS:
                pairs=list(zip(data[(model,seed,'both')],data[(model,seed,control)]));counts={k:0 for k in categories}
                em_delta=0.;p_delta=0.
                for (p,g),(q,h) in pairs:
                    assert p['id']==q['id'];a=bool(g['em']);b=bool(h['em'])
                    key=('both_correct' if a else 'both_wrong') if a==b else ('corrected' if a else 'regressed')
                    counts[key]+=1;delta=p['answer_set_probability']-q['answer_set_probability'];em_delta+=g['em']-h['em'];p_delta+=delta
                    categories[key].append({'prob_delta':delta,'nll_delta':p['answer_set_nll']-q['answer_set_nll'],
                        'f1_delta':g['f1']-h['f1'],'same_text':g['response']==h['response']})
                assert em_delta==counts['corrected']-counts['regressed'] and sum(counts.values())==3219
                seeds.append({'seed':seed,'counts':counts,'em_delta':em_delta/3219,'probability_delta':p_delta/3219})
            summary={}
            for key,values in categories.items():
                n=len(values);summary[key]={'seed_question_pairs':n,'fraction':n/(5*3219),
                    'mean_probability_delta':math.fsum(x['prob_delta'] for x in values)/n if n else None,
                    'contribution_to_overall_probability_delta':math.fsum(x['prob_delta'] for x in values)/(5*3219),
                    'positive_probability_fraction':sum(x['prob_delta']>0 for x in values)/n if n else None,
                    'mean_nll_delta':math.fsum(x['nll_delta'] for x in values)/n if n else None,
                    'mean_f1_delta':math.fsum(x['f1_delta'] for x in values)/n if n else None,
                    'same_text_fraction':sum(x['same_text'] for x in values)/n if n else None}
            contrasts.append({'model':model,'control':control,'per_seed':seeds,'categories':summary})
    write(HERE/'GENERATION_PROBABILITY_LINK.json',{'created_at':now(),'status':'complete','analysis':'Exploratory descriptive association only; repeated seed-question pairs are not independent samples.',
        'interpretation_limit':'EM flip decomposition is exact bookkeeping. Within-category probability changes do not establish a causal mechanism or represent unseen-test validation.',
        'contrasts':contrasts,'sources_sha256':sources})
    lines=['# Exploratory probability / generation decomposition','','This analysis does not change the fixed four primary tests. Categories pool five seeds descriptively; no independence assumption or p-values.','',
        '| Model | Control | Category | Pairs | Fraction | Mean probability Δ pp | Contribution to overall Δ pp |','|---|---|---|---:|---:|---:|---:|']
    for c in contrasts:
        for key,v in c['categories'].items():
            mean=v['mean_probability_delta'];formatted=f'{100*mean:+.6f}' if mean is not None else 'NA'
            lines.append(f'| {c["model"]} | {c["control"]} | {key} | {v["seed_question_pairs"]} | {100*v["fraction"]:.2f}% | {formatted} | {100*v["contribution_to_overall_probability_delta"]:+.6f} |')
    (HERE/'GENERATION_PROBABILITY_LINK.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
