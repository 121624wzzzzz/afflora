"""Frozen exploratory analysis; only pre-specified complete paired blocks."""
import numpy as np
from scipy.stats import t
from common import *
CONTRASTS=[('hidden_both','hidden'),('hidden_both','hidden_budget'),('hidden_input','hidden'),('hidden_output','hidden'),('hidden_both','hidden_input'),('hidden_both','hidden_output'),('input','base'),('output','base'),('both','base'),('both','input'),('both','output')]

def main():
    anchors={(a['task'],a['model'],a['arm'],a['seed']):a for a in read(HERE/'ANCHOR_INDEX.json')};inventory=read(HERE/'INVENTORY.json');values={};coverage=[]
    for row in inventory:
        s=row['spec'];key=(s['task'],s['model'],s['arm'],None if s['arm']=='base' else s['seed']);entry=None
        if row['reused']:
            a=anchors[key];entry={'value':a['summary']['primary'],'source':'verified_historical','summary':str(HERE/a['directory']/'SUMMARY.json')}
        elif (HERE/'audits'/f"{s['name']}.json").exists():
            audit=read(HERE/'audits'/f"{s['name']}.json");assert audit['status']=='passed'
            p=HERE/'evaluations'/s['name']/'test'/'SUMMARY.json';assert sha(p)==audit['summary_sha256']['test'];entry={'value':read(p)['primary'],'source':'new_audited','summary':str(p)}
        if entry:values[key]=entry
        coverage.append({'task':s['task'],'model':s['model'],'arm':s['arm'],'seed':key[3],'available':entry is not None,'reused':row['reused']})
    conditions=[]
    for task in TASKS:
        for model in MODELS:
            counts=[3,5] if model in ['qwen25_7b_base','qwen3_8b_base'] else [3]
            for n in counts:
                seeds=list(range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+n))
                keys=[(task,model,arm,seed) for arm in ['base']+ARMS for seed in ([None] if arm=='base' else seeds)]
                complete=all(k in values for k in keys);c={'task':task,'model':model,'seeds':seeds,'complete':complete,'family':176,'missing':[list(k) for k in keys if k not in values]}
                if complete:
                    vs={arm:[values[task,model,arm,s]['value'] for s in ([None] if arm=='base' else seeds)] for arm in ['base']+ARMS}
                    comparisons=[]
                    for treatment,control in CONTRASTS:
                        delta=np.array(vs[treatment])-np.array(vs[control]);mean=float(delta.mean());sd=float(delta.std(ddof=1));se=sd/np.sqrt(n)
                        comparisons.append({'treatment':treatment,'control':control,'differences':delta.tolist(),'mean':mean,'sd':sd,'positive_seeds':int((delta>0).sum()),'zero_seeds':int((delta==0).sum()),'negative_seeds':int((delta<0).sum()),'descriptive_unadjusted_95_ci':[mean-float(t.ppf(.975,n-1))*se,mean+float(t.ppf(.975,n-1))*se],'bonferroni176_95_ci':[mean-float(t.ppf(1-.05/(2*176),n-1))*se,mean+float(t.ppf(1-.05/(2*176),n-1))*se]})
                    c.update(values=vs,means={a:float(np.mean(v)) for a,v in vs.items()},seed_sd={a:float(np.std(v,ddof=1)) if len(v)>1 else None for a,v in vs.items()},comparisons=comparisons)
                conditions.append(c)
    write(HERE/'TEST_ANALYSIS.json',{'at':now(),'scope':'exploratory fixed matrix; complete common-seed blocks only; historical runs disclosed; no general superiority inference','coverage':coverage,'conditions':conditions})
    print('Complete paired blocks:',sum(c['complete'] for c in conditions),'/',len(conditions))
if __name__=='__main__':main()
