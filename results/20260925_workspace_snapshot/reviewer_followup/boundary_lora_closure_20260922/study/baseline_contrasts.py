from common import *
import math,collections
from scipy.stats import t
old=Path(read(HERE/'SOURCE.json')['study']);prior=read(old/'RESULTS.json')['groups'];current=read(HERE/'RESULTS.json')['groups'];out=[]
for group in prior+current:
 if group['stage']!='confirmation' or group['method'] not in ['vocab','affine'] or group['n']!=5:continue
 historical=group in prior
 baseline=next((h for h in current if h['stage']==('prior_h_baseline' if historical else 'confirmation') and h['model']==group['model'] and h['task']==group['task'] and h['method']=='none' and h['n']==5),None)
 if baseline is None:continue
 assert set(group['seeds'])==set(baseline['seeds']);diff=[group['seeds'][s]-baseline['seeds'][s] for s in sorted(group['seeds'])];mean=sum(diff)/5;se=math.sqrt(sum((d-mean)**2 for d in diff)/4/5);margin=t.ppf(.975,4)*se
 out.append(dict(family=('prior_' if historical else 'new_')+group['method']+'_vs_H',model=group['model'],task=group['task'],method=group['method'],placement=group['placement'],mean_delta=mean,ci95=[mean-margin,mean+margin],p_unadjusted=float(2*t.sf(abs(mean/se),4)) if se else (0. if mean else 1.),differences=diff,baseline_mean=baseline['mean'],method_mean=group['mean']))
for f in {r['family'] for r in out}:
 cells=[r for r in out if r['family']==f];expected=8 if f.startswith('prior_') else 10
 if len(cells)!=expected:continue
 prev=0.
 for i,r in enumerate(sorted(cells,key=lambda r:r['p_unadjusted'])):prev=max(prev,min(1.,(expected-i)*r['p_unadjusted']));r['p_holm']=prev
write(HERE/'H_BASELINE_CONTRASTS.json',dict(at=now(),contrasts=out,note='Five paired seeds, H fixed LR 2e-4; this is not a tuned-H optimality comparison.'))
