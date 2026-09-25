"""Post-hoc descriptive old/new 4B comparison, only after both studies are sealed.

No extra significance tests, model selection, fits or checkpoint replacement.
The intervention is the entire numerical/startup policy, not one isolated kernel.
"""
import json,hashlib,csv,statistics
from pathlib import Path
from datetime import datetime
PARENT=Path(__file__).resolve().parent
ROOTS=[PARENT/'qwen35_transfer_20260919',PARENT/'qwen35_fixed_4b_20260920']
OUT=PARENT/'qwen35_policy_comparison_20260920'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 assert sha(ROOTS[0]/'SEAL_MANIFEST.json')=='35af1951cb22c7c414927a847b3aa7848cc2ce0be052633d6b6b052543f114d9'
 manifests=[read(r/'SEAL_MANIFEST.json') for r in ROOTS];inputs={}
 for root in ROOTS:assert read(root/'SEAL.json')['manifest_sha256']==sha(root/'SEAL_MANIFEST.json')
 def checked(i,rel):
  p=ROOTS[i]/rel;h=sha(p);assert h==manifests[i]['files'][rel]['sha256'];inputs[str(p)]=h;return read(p)
 for task in ['wikisql','trec50']:
  for split in ['train','dev','confirm']:
   for rel in [f'data/{task}_{split}.jsonl',f'tokens/{task}_qwen35_4b_base_{split}.json']:
    hashes=[]
    for i in [0,1]:
     p=ROOTS[i]/rel;h=sha(p);assert h==manifests[i]['files'][rel]['sha256'];inputs[str(p)]=h;hashes.append(h)
    assert hashes[0]==hashes[1],rel
 old,new=[checked(i,'SEARCH_JOBS.json') for i in [0,1]];assert old==new and len(old)==72
 assert checked(0,'BUDGET_PLAN.json')==checked(1,'BUDGET_PLAN.json')
 assert checked(0,'BUDGET_ALLOCATION.json')['rank_pattern']==checked(1,'BUDGET_ALLOCATION.json')['rank_pattern']
 records=[]
 def compare(specs,stage):
  inits=[checked(i,f'checkpoints/{s["name"]}/INITIALIZATION.json') for i,s in enumerate(specs)]
  for field in ['initialization_sha256','shared_hidden_initialization_sha256','frozen_before','trainable_parameters']:
   assert inits[0][field]==inits[1][field],(specs,field)
  orders=[checked(i,f'checkpoints/{s["name"]}/TRAIN_ORDER.json') for i,s in enumerate(specs)];assert orders[0]==orders[1]
  training=[checked(i,f'checkpoints/{s["name"]}/TRAINING.json') for i,s in enumerate(specs)]
  split='dev' if stage=='search' else 'confirm'
  values=[checked(i,f'evaluations/{s["name"]}/{split}/SUMMARY.json')['primary'] for i,s in enumerate(specs)]
  s=specs[0];records.append(dict(stage=stage,task=s['task'],arm=s['arm'],seed=s['seed'],old_lr=s['lr'],new_lr=specs[1]['lr'],
   same_lr=s['lr']==specs[1]['lr'],old_primary=values[0],new_primary=values[1],difference_pp=values[1]-values[0],
   old_first_loss=training[0]['history'][0]['loss'],new_first_loss=training[1]['history'][0]['loss']))
 for s in old:compare([s,s],'search')
 confirms=[checked(i,'CONFIRMATION_JOBS.json') for i in [0,1]]
 maps=[{(s['task'],s['arm'],s['seed']):s for s in jobs} for jobs in confirms];assert maps[0].keys()==maps[1].keys()
 for key in sorted(maps[0]):compare([m[key] for m in maps],'confirmation')
 analysis=[checked(i,'ANALYSIS.json') for i in [0,1]];grouped=[]
 for task in ['wikisql','trec50']:
  for arm in ['hidden','hidden_budget','hidden_both']:
   rows=[r for r in records if r['stage']=='confirmation' and r['task']==task and r['arm']==arm]
   assert len(rows)==5 and len({r['old_lr'] for r in rows})==len({r['new_lr'] for r in rows})==1
   grouped.append(dict(task=task,arm=arm,old_lr=rows[0]['old_lr'],new_lr=rows[0]['new_lr'],same_lr=rows[0]['same_lr'],
    old_mean=statistics.mean(r['old_primary'] for r in rows),new_mean=statistics.mean(r['new_primary'] for r in rows),
    mean_change_pp=statistics.mean(r['difference_pp'] for r in rows)))
 result=dict(at=datetime.now().astimezone().isoformat(),status='passed',scope='Descriptive post-hoc comparison. All72 development pairs match LR, seed, initial parameters and training order. Confirmation pairs may use different independently selected LRs. Whole numerical/startup-policy change, not isolated L2 causality; not an independent replication and no added significance tests.',
  sources=[dict(root=str(r),manifest_sha256=sha(r/'SEAL_MANIFEST.json')) for r in ROOTS],inputs_sha256=inputs,
  comparisons=records,confirmation_summary=grouped,base=[a['base'] for a in analysis])
 OUT.mkdir(exist_ok=False);(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 with (OUT/'PAIRED_RUNS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 lines=['# 旧/新4B执行设置的描述性对照','',
  '两个完整封存研究的全部72项开发训练均核对相同学习率、种子、初始适配器、冻结基座和训练顺序。确认阶段各自重新选参，因此部分比较同时包含学习率变化。此次改变的是整套数值与启动检查设置，不能将差异归因于单一归一化内核，也不计为独立复制。本报告不新增显著性检验。',
  '', '|任务|方法|旧LR|新LR|旧确认均值|新确认均值|变化pp|','|---|---|---:|---:|---:|---:|---:|']
 for r in grouped:lines.append(f'|{r["task"]}|{r["arm"]}|{r["old_lr"]:g}|{r["new_lr"]:g}|{r["old_mean"]:.4f}|{r["new_mean"]:.4f}|{r["mean_change_pp"]:+.4f}|')
 lines+=['','全部逐运行差值和首批损失见PAIRED_RUNS.csv，完整输入哈希见RESULTS.json。主要方法增量和校正区间仍以各研究原先固定的分析为准。']
 (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print('Sealed old/new4B descriptive comparison written.',flush=True)
if __name__=='__main__':main()
