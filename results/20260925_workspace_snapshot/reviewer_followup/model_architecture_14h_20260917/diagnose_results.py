"""Post-hoc descriptive diagnostics. Does not add hypothesis tests or change scoring."""
from reporting_common import *
import csv

METRICS={
 'cluener':['primary','span_precision_pct','span_recall_pct','text_micro_f1','content_correct','schema_valid','strict_json','native_eos_pct','capped_pct','mean_generated_tokens'],
 'wikisql':['primary','lf_correct_pct','query_valid_pct','execution_ok_pct','empty_prediction_pct','strict_json_pct','native_eos_pct','capped_pct','mean_generated_tokens'],
 'trec50':['primary','macro_f1_all50','coarse_accuracy'],
 'squad2':['primary','em_pct','answerable_f1_pct','answerable_em_pct','unanswerable_f1_pct','answerability_correct_pct','predicts_empty_pct','schema_valid_pct','strict_json_pct','nonempty_extractive_pct','native_eos_pct','capped_pct','mean_generated_tokens']}

def enrich(s,task):
 s=dict(s)
 if task=='cluener':
  counts=s['span_counts'];s['span_precision_pct']=100*counts['tp']/counts['pred'] if counts['pred'] else 0
  s['span_recall_pct']=100*counts['tp']/counts['gold'] if counts['gold'] else 0
 return s

def main():
 assert read(ROOT/'FINAL_AUDIT.json')['status']=='passed'
 data={p:collect(p) for p in TASKS};lines=['# 评分分解与描述性架构诊断','',
  '此附录在查看首个7B CLUENER配对种子后设计，属于事后描述性诊断。它不增加显著性检验，不改变冻结主指标、种子、训练预算或预定对照；不是新的确认性证据。','',
  '## 内容、格式与停止行为','',
  '以下只汇总每个方案完整的预定前三个种子，Base评测一次。所有指标的逐次记录在 SECONDARY_METRICS.csv。模型内主指标配对检验仍以冻结分析为准。']
 records=[];training=[]
 for phase,ds in data.items():
  anchors={a['directory']:a for a in read(ROOT/phase/'ANCHOR_INDEX.json')} if (ROOT/phase/'ANCHOR_INDEX.json').exists() else {}
  inventory={e['spec']['name']:e for e in read(ROOT/phase/'INVENTORY.json')}
  for (task,m,arm,seed),r in ds.items():
   if arm=='base':continue
   if r['reused']:
    a=anchors[inventory[r['name']]['anchor_directory']];ck=ROOT/phase/a['directory'];eval_ck=Path(a['source_checkpoint'])
   else:ck=ROOT/phase/'checkpoints'/r['name'];eval_ck=ck
   p=ck/'TRAINING.json';tr=read(p);history=tr['history'];assert len(history)==64
   dev=eval_ck.parent.parent/'evaluations'/eval_ck.name/'dev/SUMMARY.json'
   dev_score=read(dev)['primary'] if dev.exists() else ''
   training.append([phase,task,m,arm,seed,r['reused'],np.mean([h['loss'] for h in history[:8]]),
      np.mean([h['loss'] for h in history[-8:]]),np.mean([h['grad_norm'] for h in history[:8]]),
      sum(h['grad_norm']>1 for h in history)/len(history),dev_score,r['primary'],str(p),sha(p),str(dev) if dev.exists() else '',sha(dev) if dev.exists() else ''])
 for phase,tasks in TASKS.items():
  for task in tasks:
   keys=METRICS[task];lines+=['','### '+task,'','|模型 / 方案|'+'|'.join(keys)+'|','|---|'+'|'.join(['---:']*len(keys))+'|']
   for m in MODELS:
    for arm in (ARMS if phase=='core' else ['base','hidden','hidden_budget','hidden_both']):
     if series(data[phase],task,m,arm,3) is None:continue
     seeds=[None] if arm=='base' else range(SEEDS[task],SEEDS[task]+3)
     summaries=[enrich(data[phase][task,m,arm,s]['summary'],task) for s in seeds]
     for seed,s in zip(seeds,summaries):
      for k in keys:records.append([phase,task,m,arm,seed,k,s[k]])
     lines.append('|'+LABELS[m]+' / '+ARM_LABELS[arm]+'|'+'|'.join(f'{np.mean([s[k] for s in summaries]):.3f}' for k in keys)+'|')
 lines+=['',
  'CLUENER精确跨度与实体文本F1分开呈现，precision/recall由每次运行的micro计数计算后取种子均值。WikiSQL执行正确率与逻辑形式完全匹配分开，执行等价可能产生不同SQL。TREC是候选类别概率评测，不包含自由生成格式和停止诊断。','',
  '逐次训练的前/后8步loss、前8步梯度范数、触发全局梯度裁剪的步数比例，以及可取得的开发/测试主分数保留在 TRAINING_DIAGNOSTICS.csv。8步窗口在看到8B前两个种子的差异后选择，仅用于事后诊断；它不是完整训练集loss，不宜跨种子当作同一批样本比较。历史开发分数沿用其已封存摘要，此附录未重新评分历史开发输出。梯度范数为所有可训练参数合并、裁剪前的范数，不能据此证明某一分支导致了泛化差异。所有已审计拟合都导出，不按涨跌筛选。','',
  'SQuAD2的回答/拒答子集各占50%；合法常数空答案整体F1为50、可回答F1为0。schema_valid只要求评分器可解析到合法answer字段；strict_json还要求整个原始输出为该JSON。capped保留命中生成长度上限的比例。评分固定解析开头的合法结构而非重试生成，不能把主F1改善单独解释为完整生成行为改善。predicts_empty和answerability_correct采用原始answer字符串是否严格为空，不采用官方文本归一化；nonempty_extractive_pct的分母是全体样本，不是非空答案子集。','',
  '## 双侧相对单侧的指标差分','',
  'I_H = score(H+E+U) − score(H+E) − score(H+U) + score(H)。I_0 = score(E+U) − score(E) − score(U) + score(Base)。这是当前有界、可能非线性指标上的算术交互；正值不等于已证明机制协同，负值也不否定双侧优于各单侧。参数容量和训练轨迹同时改变。不给出事后p值或置信区间。','',
  '|任务 / 模型 / n|I_H均值|I_H逐种子|I_0均值|I_0逐种子|','|---|---:|---|---:|---|']
 interactions=[]
 for task in TASKS['core']:
  for m in MODELS:
   for n in ([3,5] if m in ['qwen25_7b_base','qwen3_8b_base'] else [3]):
    if not all(series(data['core'],task,m,a,n) is not None for a in ARMS):continue
    v={a:np.array(series(data['core'],task,m,a,n)) for a in ARMS}
    ih=v['hidden_both']-v['hidden_input']-v['hidden_output']+v['hidden'];i0=v['both']-v['input']-v['output']+v['base']
    interactions.append({'task':task,'model':m,'n':n,'I_H':ih.tolist(),'I_0':i0.tolist()})
    lines.append(f'|{task} / {LABELS[m]} / {n}|{np.mean(ih):+.4f}|'+', '.join(f'{x:+.4f}' for x in ih)+f'|{np.mean(i0):+.4f}|'+', '.join(f'{x:+.4f}' for x in i0)+'|')
 lines+=['','三种子与五种子行共享前三个种子，不是两份独立重复。缺失完整预定块时不生成交互行。','']
 (ROOT/'DESCRIPTIVE_DIAGNOSTICS.md').write_text('\n'.join(lines))
 with (ROOT/'SECONDARY_METRICS.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['phase','task','model','arm','seed','metric','value']);w.writerows(records)
 with (ROOT/'TRAINING_DIAGNOSTICS.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['phase','task','model','arm','seed','reused','first8_step_loss','last8_step_loss','first8_preclip_grad_norm','clip_step_fraction','dev_primary_if_available','test_primary','training_metadata','training_metadata_sha256','dev_summary_if_available','dev_summary_sha256_if_available']);w.writerows(training)
 write(ROOT/'DESCRIPTIVE_INTERACTIONS.json',{'at':now(),'scope':'post-hoc descriptive arithmetic contrasts; no inferential tests','conditions':interactions})
 from trec_error_analysis import build as trec_error_analysis
 trec_groups=trec_error_analysis()
 write(ROOT/'DIAGNOSTICS_STATUS.json',{'at':now(),'status':'passed','metric_rows':len(records),'training_rows':len(training),'interaction_conditions':len(interactions),'trec_error_groups':trec_groups})
 print('Wrote descriptive diagnostics:',len(records),'secondary metric rows')
if __name__=='__main__':main()
