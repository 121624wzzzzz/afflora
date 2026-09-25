import collections
from common import *
records=[]
for r in read(HERE/'REUSE_AUDIT.json')['runs']:records.append(dict(spec=r['spec'],primary=r['primary'],reused=True))
for ap in (HERE/'audits').glob('*.json'):
 a=read(ap);cp=HERE/'checkpoints'/a['name'];s=read(cp/'spec.json')
 if s['smoke'] or s['stage']=='tuning':continue
 p=HERE/'evaluations'/s['name']/'test/SUMMARY.json'
 if p.exists():records.append(dict(spec=s,primary=read(p)['primary'],reused=False))
groups=collections.defaultdict(list)
for r in records:
 s=r['spec'];groups[tuple(s[k] for k in ['stage','model','task','method','placement','rank','bias'])].append(r)
result=[]
for key,rs in groups.items():result.append(dict(zip(['stage','model','task','method','placement','rank','bias'],key),n=len(rs),mean=sum(r['primary'] for r in rs)/len(rs),seeds={str(r['spec']['seed']):r['primary'] for r in rs},reused=sum(r['reused'] for r in rs)))
write(HERE/'RESULTS.json',dict(at=now(),groups=result,records=records))
lines=['# 边界普通 LoRA 对照：阶段结果','仅纳入通过独立审计的结果；未完成种子组的均值不是最终结论。调参只读 dev，不列入此表。','', '|阶段|模型|任务|方法/位置/rank/bias|n|均值|复用|','|---|---|---|---|---:|---:|---:|']
for r in result:lines.append(f"|{r['stage']}|{r['model']}|{r['task']}|{r['method']}/{r['placement']}/{r['rank']}/{r['bias']}|{r['n']}|{r['mean']:.4f}|{r['reused']}|")
(HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
# Paired intervals are emitted only for complete five-seed main cells.
import math
from scipy.stats import t as student_t
contrasts=[]
lookup={tuple(r[k] for k in ['stage','model','task','method','placement','rank','bias']):r for r in result}
for key,a in lookup.items():
 stage,model,task,method,place,rank,bias=key
 if stage not in ['common','confirmation'] or method!='affine' or rank not in [16,30] or bias!=(place!='u'):continue
 for comparison,bkey in [('ordinary', (stage,model,task,'vocab',place,1,False)),('hidden',(stage,model,task,'none','none',0,False))]:
  b=lookup.get(bkey)
  if b is None:continue
  seeds=sorted(set(a['seeds'])&set(b['seeds']))
  if len(seeds)!=5:continue
  ds=[a['seeds'][s]-b['seeds'][s] for s in seeds];mean=sum(ds)/5;sd=math.sqrt(sum((x-mean)**2 for x in ds)/4);se=sd/math.sqrt(5);margin=student_t.ppf(.975,4)*se
  pvalue=float(2*student_t.sf(abs(mean/se),4)) if se else (0. if mean else 1.)
  contrasts.append(dict(stage=stage,model=model,task=task,placement=place,comparison=comparison,affine_rank=rank,seeds=seeds,differences=ds,mean=mean,ci95=[mean-margin,mean+margin],p_unadjusted=pvalue,positive=sum(x>0 for x in ds),negative=sum(x<0 for x in ds)))
for stage,comparison,total in [('confirmation','ordinary',4),('confirmation','hidden',4)]:
 family=[c for c in contrasts if c['stage']==stage and c['comparison']==comparison]
 if len(family)==total:
  prev=0.
  for i,c in enumerate(sorted(family,key=lambda c:c['p_unadjusted'])):
   prev=max(prev,min(1.,(total-i)*c['p_unadjusted']));c['p_holm']=prev
write(HERE/'PAIRED_CONTRASTS.json',dict(at=now(),note='Only complete paired five-seed cells. Holm appears only when the entire prespecified family is complete.',contrasts=contrasts))

from selection_diagnostics import main as selection_diagnostics
selection_diagnostics()
