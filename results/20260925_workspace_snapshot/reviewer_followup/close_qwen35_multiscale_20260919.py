"""Complete reports and seals only after all 342 jobs and final audits pass."""
import os,json,time,subprocess,statistics,csv,hashlib
from pathlib import Path
from datetime import datetime
PARENT=Path(__file__).resolve().parent
MASTER=PARENT/'qwen35_multiscale_20260919'
PYTHON='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 while not (MASTER/'SCHEDULER_COMPLETE.json').exists():time.sleep(10)
 assert read(MASTER/'SCHEDULER_COMPLETE.json')['status']=='passed'
 sources=[]
 for key,size in [('08b','0.8B'),('2b','2B'),('9b','9B')]:
  root=PARENT/f'qwen35_{key}_20260919'
  for script,extra,log in [('report_qwen35_multiscale_size_20260919.py',[],'report'),('seal_qwen35_multiscale_size_20260919.py',[],'seal'),('seal_qwen35_multiscale_size_20260919.py',['--verify'],'verify')]:
   with (PARENT/f'{root.name}_{log}.log').open('w') as f:
    subprocess.run([PYTHON,'-u',str(PARENT/script),'--root',str(root),*extra],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,check=True)
   print(size,log,'passed',flush=True)
  sources.append((size,root,12,False))
 old=PARENT/'qwen35_transfer_20260919'
 assert sha(old/'SEAL_MANIFEST.json')=='35af1951cb22c7c414927a847b3aa7848cc2ce0be052633d6b6b052543f114d9'
 sources.insert(2,('4B',old,4,True))
 records=[];contrasts=[];identities=[]
 for size,root,family,historical in sources:
  a=read(root/'ANALYSIS.json');manifest=read(root/'SEAL_MANIFEST.json')
  assert sha(root/'ANALYSIS.json')==manifest['files']['ANALYSIS.json']['sha256']
  seal=read(root/'SEAL.json');assert seal['manifest_sha256']==sha(root/'SEAL_MANIFEST.json')
  identities.append(dict(size=size,path=str(root),historical=historical,family=family,manifest_sha256=seal['manifest_sha256']))
  for task in ['wikisql','trec50']:
   record=dict(size=size,task=task,historical=historical,base=a['base'][task]['primary'])
   for arm in ['hidden','hidden_budget','hidden_both']:
    scores=[r['primary'] for r in a['results'][task][arm]];record[arm]=statistics.mean(scores);record[arm+'_sd']=statistics.stdev(scores)
   records.append(record)
  for c in a['comparisons']:
   contrasts.append(dict(size=size,historical=historical,family=family,task=c['task'],control=c['b'],mean_pp=c['mean_pp'],seed_deltas=c['paired_seed_deltas_pp'],ci95=c['ci95'],corrected_ci95=c[f'bonferroni{family}_ci95']))
 lines=['# Qwen3.5 多尺寸完整结果','',
  '新增 0.8B、2B、9B 均完成同等学习率搜索与五种子确认，包含全部正负结果。4B 为已封存历史参照。',
  '', '|模型|任务|Base|H 均值±SD|严格等参 H 均值±SD|H+E+U 均值±SD|','|---|---|---:|---:|---:|---:|']
 for r in records:
  lines.append(f'|{r["size"]}{"（历史）" if r["historical"] else ""}|{r["task"]}|{r["base"]:.2f}|'+ '|'.join(f'{r[k]:.2f}±{r[k+"_sd"]:.2f}' for k in ['hidden','hidden_budget','hidden_both'])+'|')
 lines+=['','|模型|任务|叠加相对对照|均值差 pp|校正种子95%区间|校正比较数|','|---|---|---|---:|---|---:|']
 for c in contrasts:
  lo,hi=c['corrected_ci95'];lines.append(f'|{c["size"]}|{c["task"]}|{c["control"]}|{c["mean_pp"]:+.3f}|[{lo:+.3f}, {hi:+.3f}]|{c["family"]}|')
 lines+=['','三个新增尺寸共 12 项主要比较使用 Bonferroni 校正；4B 保留原先 family-4 分析，不计为新增复制。每个对比固定数据和开发集选出的配置，五种子区间依赖小样本 t 假设。跨零不证明无效或等效。固定五个模型的数据簇重采样区间另见各尺寸报告，不能替代种子不确定性。',
  '', '新增所有非 Base 训练都重复首批前向/反向并保存数值核验；通过局部检查不证明全过程跨设备逐位确定。历史4B没有这一启动门禁，曾记录初始BF16损失跨运行差异。原生共享/不共享 E/U 随尺寸同时变化，不能单独归因于共享机制；此处仅验证 H+E+U 的具体双侧配置。',
  '', '全部原参数冻结、只有指定适配器训练。等参组使用真实更新的额外 LoRA 秩。公开确认数据已在早期模型上使用，不是项目全新未见数据。',
  '', '逐尺寸完整数据与封存身份：']
 for x in identities:lines.append(f'- {x["size"]}: {x["path"]}; manifest `{x["manifest_sha256"]}`')
 (MASTER/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 result=dict(at=datetime.now().astimezone().isoformat(),status='passed',new_jobs=342,sources=identities,results=records,comparisons=contrasts)
 (MASTER/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 with (MASTER/'RESULTS.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 plan=read(MASTER/'PLAN.json');plan.update(stage='complete: all 342 jobs passed, all three size studies sealed and independently verified',completed_at=result['at'])
 (MASTER/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
 p=MASTER/'README.md';text=p.read_text();start=text.index('当前：');end=text.index('查看父目录',start)
 text=text[:start]+'当前：全部342项运行已完成并通过审计，三个新增尺寸均已封存和独立复核。完整汇总见 RESULTS_ZH.md 与 RESULTS.json。'+text[end:]
 p.write_text(text.replace('（进行中）','（已完成）'))
 p=MASTER/'RUN_HANDOFF.md';p.write_text('COMPLETED: all new size studies passed, sealed and independently verified. See RESULTS_ZH.md. Previous operational notes follow for provenance.\n\n'+p.read_text())
 print('All new size studies sealed and independently verified; combined report ready.',flush=True)
if __name__=='__main__':main()
