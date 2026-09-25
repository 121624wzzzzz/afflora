"""Prespecified comparisons; no change to selection or run admission."""
import math,statistics
from scipy.stats import t
from common import *
from design import CONFIRM_SEEDS,SEARCH_SEEDS,CANDIDATES,spec

CONTRASTS=[('tuned_heu','tuned_hidden'),('tuned_heu','tuned_budget'),('tuned_heu','anchor_heu'),('tuned_budget','anchor_budget'),('anchor_heu','anchor_budget')]

def training_diagnostics(name):
    h=read(HERE/'checkpoints'/name/'TRAINING.json')['history']
    return {'clipped_steps':sum(r['grad_norm']>1 for r in h),'mean_joint_grad_norm':statistics.mean(r['grad_norm'] for r in h),
            'mean_group_grad_norm':{g:statistics.mean(r['group_grad_norms'][g] for r in h) for g in ['hidden','input','output']},
            'mean_group_gradient_energy_share':{g:statistics.mean(r['group_grad_norms'][g]**2/sum(v*v for v in r['group_grad_norms'].values()) for r in h) for g in ['hidden','input','output']},
            'sum_step_update_norms':{g:sum(r['group_update_norms'][g] for r in h) for g in ['hidden','input','output']},
            'tail8_training_step_loss':statistics.mean(r['loss'] for r in h[-8:])}

def main():
    assert read(HERE/'SCHEDULER_COMPLETE.json')['status']=='passed'
    selection=read(HERE/'SELECTION.json');roles=selection['roles'];results={}
    for role,names in roles.items():
        results[role]=[]
        for seed in CONFIRM_SEEDS:
            name=names[str(seed)];audit=read(HERE/'audits'/f'{name}.json');assert audit['status']=='passed'
            p=HERE/'evaluations'/name/'confirm/SUMMARY.json';assert sha(p)==audit['summary_sha256']['confirm']
            s=read(p);assert s['n']==2048
            results[role].append({'seed':seed,'run':name,**{k:s[k] for k in ['primary','lf_correct_pct','query_valid_pct','strict_json_pct','capped_pct']},'diagnostics':training_diagnostics(name)})
    comparisons=[]
    for a,b in CONTRASTS:
        delta=[x['primary']-y['primary'] for x,y in zip(results[a],results[b])]
        avg=statistics.mean(delta);sd=statistics.stdev(delta);se=sd/math.sqrt(len(delta));identity=roles[a]==roles[b]
        comparisons.append({'a':a,'b':b,'paired_seed_deltas_pp':delta,'mean_pp':avg,'sd_pp':sd,
                            'ci95':[avg-t.ppf(.975,4)*se,avg+t.ppf(.975,4)*se],
                            'bonferroni5_ci95':[avg-t.ppf(1-.05/(2*5),4)*se,avg+t.ppf(1-.05/(2*5),4)*se],
                            'identity':identity,'positive_seeds':sum(x>0 for x in delta)})
    search_diag={}
    for arm,cs in CANDIDATES.items():
        search_diag[arm]=[dict(c,seed_diagnostics=[training_diagnostics(spec('search',arm,i,s)['name']) for s in SEARCH_SEEDS]) for i,c in enumerate(cs)]
    within_lr=[]
    for i,j in [(0,1),(2,3),(4,5)]:
        a,b=selection['scores']['hidden_both'][i],selection['scores']['hidden_both'][j]
        within_lr.append({'hidden_lr':a['lr'],'boundary_ratio_change':'1 -> .25','dev_seed_deltas_pp':[y-x for x,y in zip(a['seed_scores'],b['seed_scores'])],
                          'mean_dev_delta_pp':b['mean']-a['mean'],
                          'clipped_steps_ratio1':[r['clipped_steps'] for r in search_diag['hidden_both'][i]['seed_diagnostics']],
                          'clipped_steps_ratio025':[r['clipped_steps'] for r in search_diag['hidden_both'][j]['seed_diagnostics']]})
    report={'at':now(),'selection_sha256':sha(HERE/'SELECTION.json'),'family':5,'n_seeds':5,'results':results,'comparisons':comparisons,
            'search_training_diagnostics':search_diag,'search_within_lr_boundary_ratio_diagnostics':within_lr,
            'limitations':['Two-seed development selection; limited, differently shaped search grids.',
              'Intervals cover seed variability conditional on the datasets, not dataset sampling uncertainty.',
              'Fresh project holdout excludes previous tables; not proof of absence from model pretraining.',
              'Different old/new test populations prohibit interpreting cross-study absolute-score differences as tuning gains.',
              'Training/clipping diagnostics are descriptive and do not prove a causal explanation.']}
    write(HERE/'ANALYSIS.json',report)
    lines=['# Llama-8B WikiSQL learning-rate follow-up','','Development selection is frozen before confirmation. Scores are execution accuracy (%).',
           '','## Full development grid','','| Arm | H LR | E/U ratio | Seed 7200 | Seed 7201 | Mean | Selected |','|---|---:|---:|---:|---:|---:|---|']
    for arm,rs in selection['scores'].items():
        for r in rs:lines.append(f'| {arm} | {r["lr"]:g} | {r["boundary_lr_ratio"]:g} | {r["seed_scores"][0]:.4f} | {r["seed_scores"][1]:.4f} | {r["mean"]:.4f} | {r["id"]==selection["selected"][arm]["id"]} |')
    lines+=['','## New confirmation set','','| Role | 7300 | 7301 | 7302 | 7303 | 7304 | Mean |','|---|---:|---:|---:|---:|---:|---:|']
    for role,rs in results.items():lines.append('| '+role+' | '+' | '.join(f'{r["primary"]:.4f}' for r in rs)+f' | {statistics.mean(r["primary"] for r in rs):.4f} |')
    lines+=['','| Paired contrast | Mean delta (pp) | SD | Marginal 95% CI | Family-5 95% CI | Identity |','|---|---:|---:|---|---|---|']
    for c in comparisons:
        ci=c['ci95'];adj=c['bonferroni5_ci95'];lines.append(f'| {c["a"]} − {c["b"]} | {c["mean_pp"]:.4f} | {c["sd_pp"]:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | [{adj[0]:.4f}, {adj[1]:.4f}] | {c["identity"]} |')
    lines+=['','Identity comparisons use the same saved fit and are not independent replications.','',*['- '+s for s in report['limitations']]]
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(canonical({'comparisons':comparisons}),flush=True)
if __name__=='__main__':main()
