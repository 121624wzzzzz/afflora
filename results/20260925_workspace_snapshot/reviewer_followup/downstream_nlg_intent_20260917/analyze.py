import numpy as np
from scipy.stats import t
from common import *

def main():
    allout={}
    for split in ['dev','test']:
        result={'at':now(),'split':split,'family':8,'seeds':list(range(8100,8105)),'conditions':[]};lines=['# V4 '+split+' results','', '|Task|Model|Base|LoRA|Budget LoRA|LoRA+A-LoRA|','|---|---|---:|---:|---:|---:|']
        for task in TASKS:
            for model in MODELS:
                summaries={};values={}
                for arm in ['base','hidden','hidden_budget','hidden_both']:
                    names=[f'{task}_{model}_base'] if arm=='base' else [f'{task}_{model}_{arm}_s{s}' for s in range(8100,8105)]
                    ss=[read(HERE/'evaluations'/name/split/'SUMMARY.json') for name in names];summaries[arm]=ss;values[arm]=[s['primary'] for s in ss]
                comparisons=[]
                for control in ['hidden','hidden_budget']:
                    d=np.array(values['hidden_both'])-values[control];mean=float(d.mean());sd=float(d.std(ddof=1));radius=float(t.ppf(1-.05/16,4)*sd/np.sqrt(5))
                    comparisons.append({'control':control,'differences':d.tolist(),'mean':mean,'sd':sd,'bonferroni8_ci':[mean-radius,mean+radius],'all_seeds_positive':bool((d>0).all())})
                secondary={arm:{k:float(np.mean([s[k] for s in ss])) for k,v in ss[0].items() if isinstance(v,(int,float)) and k not in ['seconds']} for arm,ss in summaries.items()}
                result['conditions'].append({'task':task,'model':model,'values':values,'means':{a:float(np.mean(v)) for a,v in values.items()},'comparisons':comparisons,'secondary_means':secondary})
                lines.append('|'+task+'|'+model+'|'+'|'.join(f'{np.mean(values[a]):.3f}' for a in ['base','hidden','hidden_budget','hidden_both'])+'|')
        for c in result['conditions']:
            lines.extend(['',c['task']+' / '+c['model']])
            for x in c['comparisons']:lines.append(f"- stack − {x['control']}: {x['mean']:+.4f}, corrected CI [{x['bonferroni8_ci'][0]:+.4f}, {x['bonferroni8_ci'][1]:+.4f}], seeds {x['differences']}")
        lines.extend(['','Banking77: accuracy percentage points; E2E: corpus BLEU points. Five-seed final checkpoints; intervals conditional on frozen test examples. No hyperparameter search.'])
        write(HERE/f'{split.upper()}_ANALYSIS.json',result);(HERE/f'{split.upper()}_RESULTS.md').write_text('\n'.join(lines)+'\n');allout[split]=result
    print((HERE/'TEST_RESULTS.md').read_text())
if __name__=='__main__':main()
