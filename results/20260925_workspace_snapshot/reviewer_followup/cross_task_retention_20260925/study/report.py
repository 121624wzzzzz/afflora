from common import *
import collections,math
from scipy.stats import t

def main():
 base={};groups=collections.defaultdict(list);records=[]
 for root in (HERE/'results').iterdir():
  if not (root/'AUDIT.json').exists():continue
  assert read(root/'AUDIT.json')['status']=='passed';job=read(root/'JOB.json');scores=read(root/'SUMMARY.json')
  for task,s in scores.items():
   record=dict(model=job['model'],source_task=job['source_task'],method=job['method'],seed=job['seed'],target_score=job['source_primary'],retention_task=task,**s);records.append(record)
   if job['method']=='base':base[job['model'],task]=s['accuracy']
   else:groups[job['model'],job['source_task'],task,job['method']].append(record)
 result=[];lookup={}
 for (m,source,task,method),rs in sorted(groups.items()):
  b=base.get((m,task));mean=sum(r['accuracy'] for r in rs)/len(rs)
  row=dict(model=m,source_task=source,retention_task=task,method=method,n=len(rs),mean=mean,base=b,delta_base=mean-b if b is not None else None,target_mean=sum(r['target_score'] for r in rs)/len(rs),candidate_mass=sum(r['candidate_mass'] for r in rs)/len(rs),seeds={str(r['seed']):r['accuracy'] for r in rs});result.append(row);lookup[m,source,task,method]=row
 contrasts=[]
 for (m,source,task,method),a in lookup.items():
  if method!='affine':continue
  for comparator in ['none','vocab']:
   b=lookup.get((m,source,task,comparator))
   if b is None or a['n']!=5 or b['n']!=5:continue
   seeds=sorted(set(a['seeds'])&set(b['seeds']));assert len(seeds)==5
   ds=[a['seeds'][s]-b['seeds'][s] for s in seeds];mean=sum(ds)/5;se=math.sqrt(sum((x-mean)**2 for x in ds)/4/5);margin=float(t.ppf(.975,4))*se
   pv=float(2*t.sf(abs(mean/se),4)) if se else (0. if mean else 1.)
   contrasts.append(dict(model=m,source_task=source,retention_task=task,comparison='aLoRA-minus-'+comparator,mean=mean,ci95=[mean-margin,mean+margin],p_unadjusted=pv,positive=sum(x>0 for x in ds),negative=sum(x<0 for x in ds),seeds=seeds,differences=ds))
 for comparator in ['none','vocab']:
  family=[c for c in contrasts if c['comparison']=='aLoRA-minus-'+comparator]
  if len(family)==8:
   prev=0
   for i,c in enumerate(sorted(family,key=lambda x:x['p_unadjusted'])):prev=max(prev,min(1,(8-i)*c['p_unadjusted']));c['p_holm']=prev
 write(HERE/'RESULTS.json',dict(at=now(),groups=result,records=records));write(HERE/'PAIRED_CONTRASTS.json',dict(at=now(),contrasts=contrasts))
 lines=['# 跨任务保留评测','负的“相对base”表示该候选评分协议下下降；aLoRA−H/普通为负表示额外退化。n<5非完整结果。','','|模型|训练任务|保留任务|方法|n|目标任务分数|保留准确率|base|相对base|候选概率质量|','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
 for r in result:lines.append(f"|{r['model']}|{r['source_task']}|{r['retention_task']}|{r['method']}|{r['n']}|{r['target_mean']:.4f}|{r['mean']:.4f}|{r['base']:.4f}|{r['delta_base']:+.4f}|{r['candidate_mass']:.6f}|")
 (HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
