"""Report every fixed rank; rank selection uses validation only."""
from common import *
def main():
 if not (HERE/'SELECTION_FROZEN.json').exists():return
 selection=read(HERE/'SELECTION_FROZEN.json')['selection'];groups=read(HERE/'RESULTS.json')['groups'];rows=[]
 for r in selection:
  model,task,method,place,rank,bias=r['architecture']
  cfg=next(s for s in read(HERE/'TUNING_JOBS.json') if [s[k] for k in ['model','task','method','placement','rank','bias']]==r['architecture'])
  dev=sum(r['dev_scores'][str(r['ratio'])])/3
  g=next((g for g in groups if g['method']==method and g['rank']==rank),None)
  rows.append(dict(method=method,rank=rank,boundary_parameters=cfg['expected_boundary_parameters'],total_parameters=cfg['expected_total_parameters'],selected_boundary_lr_ratio=r['ratio'],validation_mean=dev,test_n=g['n'] if g else 0,test_mean=g['mean'] if g and g['n']==5 else None))
 best={m:max([r for r in rows if r['method']==m],key=lambda r:(r['validation_mean'],-r['boundary_parameters']))['rank'] for m in ['affine','vocab']}
 write(HERE/'PARAMETER_CURVE.json',dict(at=now(),rows=rows,validation_selected_rank_by_family=best,note='All ranks reported; no test-based rank selection. Partial five-seed test means withheld.'))
 lines=['# 参数—效果表','rank和学习率只按补充验证集选择；五seed确认齐全后才报告test均值。','','|方法|rank|边界参数|总可训练参数|验证均值|test n|test均值|','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"|{r['method']}|{r['rank']}|{r['boundary_parameters']}|{r['total_parameters']}|{r['validation_mean']:.4f}|{r['test_n']}|{r['test_mean'] if r['test_mean'] is not None else 'pending'}|")
 (HERE/'PARAMETER_CURVE_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
