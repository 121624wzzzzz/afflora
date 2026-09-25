"""Report and seal all four sizes only after complete, audited finite queues."""
import os,json,time,subprocess,statistics,csv,hashlib
from pathlib import Path
from datetime import datetime
PARENT=Path(__file__).resolve().parent
MASTER=PARENT/'qwen35_fixed_multiscale_20260920'
PYTHON='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 while not (MASTER/'SCHEDULER_COMPLETE.json').exists():time.sleep(10)
 assert read(MASTER/'SCHEDULER_COMPLETE.json')['status']=='passed'
 sources=[];records=[];contrasts=[]
 for key,size in [('08b','0.8B'),('2b','2B'),('4b','4B'),('9b','9B')]:
  root=PARENT/f'qwen35_fixed_{key}_20260920'
  for script,args,log in [('report_qwen35_fixed_size_20260920.py',[],'report'),('seal_qwen35_fixed_size_20260920.py',[],'seal'),('seal_qwen35_fixed_size_20260920.py',['--verify'],'verify')]:
   with (PARENT/f'{root.name}_{log}.log').open('x') as f:
    subprocess.run([PYTHON,'-u',str(PARENT/script),'--root',str(root),*args],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,check=True)
   print(size,log,'passed',flush=True)
  a=read(root/'ANALYSIS.json');manifest=read(root/'SEAL_MANIFEST.json');assert a['family']==16
  assert sha(root/'ANALYSIS.json')==manifest['files']['ANALYSIS.json']['sha256']
  seal=read(root/'SEAL.json');assert seal['manifest_sha256']==sha(root/'SEAL_MANIFEST.json')
  sources.append(dict(size=size,path=str(root),historical=False,family=16,manifest_sha256=seal['manifest_sha256']))
  for task in ['wikisql','trec50']:
   row=dict(size=size,task=task,historical=False,base=a['base'][task]['primary'])
   for arm in ['hidden','hidden_budget','hidden_both']:
    scores=[r['primary'] for r in a['results'][task][arm]];row[arm]=statistics.mean(scores);row[arm+'_sd']=statistics.stdev(scores)
   records.append(row)
  for c in a['comparisons']:
   contrasts.append(dict(size=size,historical=False,family=16,task=c['task'],control=c['b'],mean_pp=c['mean_pp'],seed_deltas=c['paired_seed_deltas_pp'],ci95=c['ci95'],corrected_ci95=c['bonferroni16_ci95']))
 old=PARENT/'qwen35_transfer_20260919';assert sha(old/'SEAL_MANIFEST.json')=='35af1951cb22c7c414927a847b3aa7848cc2ce0be052633d6b6b052543f114d9'
 olda=read(old/'ANALYSIS.json');assert sha(old/'ANALYSIS.json')==read(old/'SEAL_MANIFEST.json')['files']['ANALYSIS.json']['sha256']
 historical=dict(path=str(old),manifest_sha256=sha(old/'SEAL_MANIFEST.json'),family=4,comparisons=olda['comparisons'])
 result=dict(at=datetime.now().astimezone().isoformat(),status='passed',new_jobs=456,sources=sources,results=records,comparisons=contrasts,historical_4b=historical)
 write(MASTER/'RESULTS.json',result)
 with (MASTER/'RESULTS.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 lines=['# Qwen3.5 四个尺寸：统一数值设置后的完整比较','',
  '0.8B、2B、4B、9B 全部完成同等学习率搜索与五种子确认。四个尺寸均采用固定FLA执行配置和同一PyTorch数值设置，包含全部正负结果。',
  '', '|模型|任务|Base|普通 LoRA 均值±SD|严格等参 LoRA 均值±SD|LoRA＋E＋U 均值±SD|','|---|---|---:|---:|---:|---:|']
 for r in records:lines.append(f'|{r["size"]}|{r["task"]}|{r["base"]:.2f}|'+ '|'.join(f'{r[k]:.2f}±{r[k+"_sd"]:.2f}' for k in ['hidden','hidden_budget','hidden_both'])+'|')
 lines+=['','|模型|任务|叠加相对对照|均值差 pp|校正种子95%区间|','|---|---|---|---:|---|']
 for c in contrasts:
  lo,hi=c['corrected_ci95'];lines.append(f'|{c["size"]}|{c["task"]}|{c["control"]}|{c["mean_pp"]:+.3f}|[{lo:+.3f}, {hi:+.3f}]|')
 lines+=['', '四个尺寸共16项主要比较统一使用Bonferroni校正。五种子t区间依赖小样本假设；跨零不证明无效或等效。固定五个模型的数据簇重采样区间另见逐尺寸报告，不能替代种子不确定性，均未合并数据、训练和选参不确定性。',
  '', '只有指定适配器更新，原模型参数均冻结；等参组使用真实更新的额外LoRA秩。每次训练首批16项微损失精确匹配独立进程参考值，重复反向全部梯度一致。这些检查不构成全过程跨设备确定性证明。',
  '', '0.8B/2B/4B原生共享E/U，9B原生不共享；规模与共享机制同时变化，无法单独归因于共享。本轮仅检验指定的双侧叠加配置；参数位置、alpha、dropout和偏置并未被单独隔离。公开数据已在项目早期评估，并非新的未见确认集。',
  '', '旧4B及暂停前完成的全部旧设置实验原样保留，未拼入本轮。控制实验能精确复现首批数值差异，但无法据此确定旧运行选择了哪个内核，更不能直接解释最终效果涨跌。旧/新4B描述性对照见RESULTS.json；不是新增显著性检验或独立复制。',
  '', '完整文件及封存身份：']
 for source in sources:lines.append(f'- {source["size"]}: {source["path"]}; manifest `{source["manifest_sha256"]}`')
 (MASTER/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 with (PARENT/'plot_qwen35_fixed_20260920.log').open('x') as f:
  subprocess.run([PYTHON,str(PARENT/'plot_qwen35_fixed_20260920.py')],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
 plan=read(MASTER/'PLAN.json');plan.update(status='complete_sealed_verified',completed_at=result['at']);write(MASTER/'PLAN.json',plan)
 write(MASTER/'CLOSEOUT_COMPLETE.json',dict(at=result['at'],status='passed',jobs=456,result_sha256=sha(MASTER/'RESULTS.json')))
 print('All 456 jobs audited; four studies sealed and independently verified.',flush=True)
if __name__=='__main__':main()
