"""Supplementary table-cluster bootstrap, specified before confirmation opens."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'llama_lr_20260918'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def main():
    plan=read(ROOT/'TABLE_BOOTSTRAP_PLAN.json');assert plan['script_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert read(ROOT/'FINAL_AUDIT.json')['status']=='passed'
    analysis=read(ROOT/'ANALYSIS.json');data=rows(ROOT/'data/wikisql_confirm.jsonl');tables=sorted({r['table_id'] for r in data})
    index={t:i for i,t in enumerate(tables)};assignment=np.array([index[r['table_id']] for r in data]);counts=np.bincount(assignment,minlength=len(tables))
    roles={}
    for role,rs in analysis['results'].items():
        values=[]
        for r in rs:
            responses=rows(ROOT/'evaluations'/r['run']/'confirm/responses.jsonl')
            assert [x['id'] for x in responses]==[x['id'] for x in data]
            values.append([float(x['content_correct']) for x in responses])
        roles[role]=np.array(values).mean(axis=0)
    contrasts=analysis['comparisons'];sums=[]
    assert [(c['a'],c['b']) for c in contrasts]==[tuple(c) for c in plan['comparisons']]
    for c in contrasts:
        delta=roles[c['a']]-roles[c['b']]
        assert abs(100*delta.mean()-c['mean_pp'])<1e-12
        sums.append(np.bincount(assignment,weights=delta,minlength=len(tables)))
    sums=np.array(sums).T;rng=np.random.default_rng(plan['seed']);draws=[];n=plan['replicates']
    for start in range(0,n,500):
        weights=rng.multinomial(len(tables),np.full(len(tables),1/len(tables)),size=min(500,n-start))
        assert np.all(weights.sum(axis=1)==len(tables))
        draws.append(100*(weights@sums)/(weights@counts)[:,None])
    draws=np.concatenate(draws);out=[]
    for i,c in enumerate(contrasts):
        out.append({'a':c['a'],'b':c['b'],'original_mean_pp':c['mean_pp'],'bootstrap_sd_pp':float(draws[:,i].std(ddof=1)),
                    'percentile_ci95':np.percentile(draws[:,i],[2.5,97.5],method='linear').tolist(),
                    'bonferroni5_percentile_ci95':np.percentile(draws[:,i],[.5,99.5],method='linear').tolist(),
                    'identity':c['identity']})
    result={'status':'passed','plan_sha256':hashlib.sha256((ROOT/'TABLE_BOOTSTRAP_PLAN.json').read_bytes()).hexdigest(),
            'replicates':n,'seed':plan['seed'],'table_clusters':len(tables),'examples':len(data),'comparisons':out,
            'interpretation':'Resample tables and keep all sampled questions in each chosen table; paired across methods; average over the same five fitted models. Captures table-sampling variability conditional on fitted models, not training-seed or model-selection uncertainty.',
            'limitations':['Supplementary finite-sample percentile bootstrap, separate from the frozen primary seed t intervals.',
                          'Assumes the observed eligible tables represent the relevant filtered population; no claim about unfiltered WikiSQL or other tasks.',
                          'Does not combine every source of uncertainty or establish causal mechanisms.']}
    (ROOT/'TABLE_BOOTSTRAP.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
