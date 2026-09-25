import numpy as np
from scipy.stats import t
from common import *
def main():
    conditions=[];coverage=[]
    for split in ['dev','test']:
        cs=[]
        for task in TASKS:
            seeds=list(range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+3))
            for model in MODELS:
                values={};secondary={};missing=[]
                for arm in ['base']+ARMS:
                    names=[f'{task}_{model}_base'] if arm=='base' else [f'{task}_{model}_{arm}_s{s}' for s in seeds]
                    ss=[]
                    for name in names:
                        p=HERE/'audits'/f'{name}.json';summary=HERE/'evaluations'/name/split/'SUMMARY.json'
                        if not p.exists():missing.append(name);continue
                        audit=read(p);assert audit['status']=='passed' and sha(summary)==audit['summary_sha256'][split]
                        ss.append(read(summary))
                    if len(ss)==len(names):
                        values[arm]=[s['primary'] for s in ss]
                        secondary[arm]={k:float(np.mean([s[k] for s in ss])) for k,v in ss[0].items() if isinstance(v,(float,int)) and k!='seconds'}
                c={'task':task,'model':model,'seeds':seeds,'complete':not missing,'missing':missing,'family':32}
                if not missing:
                    comparisons=[]
                    for control in ['hidden','hidden_budget']:
                        delta=np.array(values['hidden_both'])-values[control];mean=float(delta.mean());sd=float(delta.std(ddof=1));se=sd/np.sqrt(3)
                        comparisons.append({'control':control,'differences':delta.tolist(),'mean':mean,'sd':sd,'positive_seeds':int((delta>0).sum()),'zero_seeds':int((delta==0).sum()),'negative_seeds':int((delta<0).sum()),'descriptive_unadjusted_95_ci':[mean-float(t.ppf(.975,2))*se,mean+float(t.ppf(.975,2))*se],'bonferroni32_95_ci':[mean-float(t.ppf(1-.05/64,2))*se,mean+float(t.ppf(1-.05/64,2))*se]})
                    c.update(values=values,means={a:float(np.mean(v)) for a,v in values.items()},secondary_means=secondary,comparisons=comparisons)
                cs.append(c)
        write(HERE/f'{split.upper()}_ANALYSIS.json',{'at':now(),'scope':'pre-specified exploratory three-seed new-task matrix; all 32 planned comparisons retained in multiplicity denominator','conditions':cs})
        print(split,'complete',sum(c['complete'] for c in cs),'/',len(cs))
if __name__=='__main__':main()
