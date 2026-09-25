"""All fixed blocks, audited outputs only; no selection of favorable results."""
import numpy as np
from scipy.stats import t
from common import *

def main():
    inventory=read(HERE/'INVENTORY.json');values={};coverage=[];conditions=[]
    for row in inventory:
        s=row['spec'];key=(s['task'],s['model'],s['arm'],None if s['arm']=='base' else s['seed'])
        audit=HERE/'audits'/f"{s['name']}.json";available=False
        if audit.exists():
            a=read(audit);assert a['status']=='passed'
            p=HERE/'evaluations'/s['name']/'test/SUMMARY.json';assert sha(p)==a['summary_sha256']['test']
            values[key]=read(p)['primary'];available=True
        coverage.append({'task':s['task'],'model':s['model'],'arm':s['arm'],'seed':key[3],'available':available})
    lines=['# Llama transfer results','',f'Updated: {now()}', '', 'Only independently audited results. Fixed three-seed blocks; missing values remain missing.','',
           '| Task | Model | Base | H | Budget H | H+E+U | ΔH | Δbudget |','|---|---|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        for model in MODELS:
            seeds=list(range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+3))
            keys=[(task,model,arm,seed) for arm in ['base']+ARMS for seed in ([None] if arm=='base' else seeds)]
            missing=[list(k) for k in keys if k not in values]
            c={'task':task,'model':model,'seeds':seeds,'complete':not missing,'missing':missing,'family':8}
            if not missing:
                vs={a:[values[task,model,a,s] for s in ([None] if a=='base' else seeds)] for a in ['base']+ARMS}
                means={a:float(np.mean(v)) for a,v in vs.items()};comparisons=[]
                for control in ['hidden','hidden_budget']:
                    delta=np.array(vs['hidden_both'])-vs[control];mean=float(delta.mean());sd=float(delta.std(ddof=1));se=sd/np.sqrt(3)
                    comparisons.append({'treatment':'hidden_both','control':control,'differences':delta.tolist(),'mean':mean,'sd':sd,'positive_seeds':int((delta>0).sum()),'zero_seeds':int((delta==0).sum()),'negative_seeds':int((delta<0).sum()),'descriptive_unadjusted_95_ci':[mean-t.ppf(.975,2)*se,mean+t.ppf(.975,2)*se],'bonferroni8_95_ci':[mean-t.ppf(1-.05/16,2)*se,mean+t.ppf(1-.05/16,2)*se]})
                c.update(values=vs,means=means,comparisons=comparisons)
                lines.append(f"| {task} | {model} | {means['base']:.3f} | {means['hidden']:.3f} | {means['hidden_budget']:.3f} | {means['hidden_both']:.3f} | {comparisons[0]['mean']:+.3f} | {comparisons[1]['mean']:+.3f} |")
            else:lines.append(f'| {task} | {model} | incomplete ({len(missing)} missing runs) | | | | | |')
            conditions.append(c)
    lines+=['','## Fixed planned contrasts','']
    for c in conditions:
        if not c['complete']:continue
        for cmp in c['comparisons']:
            lines.append(f"- {c['task']} / {c['model']} / HEU − {cmp['control']}: raw {cmp['differences']}; mean {cmp['mean']:+.4f}; SD {cmp['sd']:.4f}; unadjusted CI {cmp['descriptive_unadjusted_95_ci']}; Bonferroni-8 CI {cmp['bonferroni8_95_ci']}.")
    lines+=['','3B budget H has 1,024 extra trainable parameters (0.00829%); 8B is exactly matched. Positive signs/means alone do not establish corrected significance. All settings and model identities in PROTOCOL.md.']
    write(HERE/'TEST_ANALYSIS.json',{'at':now(),'scope':'exploratory fixed Llama extension, complete audited three-seed blocks only','coverage':coverage,'conditions':conditions})
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Audited formal runs',sum(r['available'] for r in coverage),'/',len(coverage),'complete blocks',sum(c['complete'] for c in conditions),flush=True)
if __name__=='__main__':main()
