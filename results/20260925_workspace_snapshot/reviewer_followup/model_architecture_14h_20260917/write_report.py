"""Presentation after final audits; never changes frozen training or inference."""
from reporting_common import *
import csv,collections
def main_arm_descriptions(data):
 rows=[]
 for task in TASKS['core']:
  for model in MODELS:
   values={a:series(data,task,model,a,3) for a in ['base','hidden','hidden_budget','hidden_both']}
   row={'task':task,'model':model,'seeds':list(range(SEEDS[task],SEEDS[task]+3)),'complete_main_arms':all(v is not None for v in values.values())}
   if row['complete_main_arms']:
    row.update(values=values,means={a:float(np.mean(v)) for a,v in values.items()},
     paired_differences={a:(np.array(values['hidden_both'])-values[a]).tolist() for a in ['hidden','hidden_budget']},
     reused_main_fits=sum(data[task,model,a,s]['reused'] for a in ['hidden','hidden_budget','hidden_both'] for s in row['seeds']))
   rows.append(row)
 return rows

def main():
 audit=read(ROOT/'FINAL_AUDIT.json');assert audit['status']=='passed';state=read(ROOT/'STATE.json')
 data={p:collect(p) for p in TASKS};analyses={p:read(ROOT/p/'TEST_ANALYSIS.json') for p in TASKS}
 formal=[j for j in state['jobs'].values() if j['stage']=='FORMAL' and j['status']=='passed'];fits=sum(j['spec']['arm']!='base' for j in formal);bases=len(formal)-fits;smokes=sum(j['stage']=='SMOKE' and j['status']=='passed' for j in state['jobs'].values());missing=audit['uncompleted_jobs']
 core3=[c for c in analyses['core']['conditions'] if len(c['seeds'])==3 and c['complete']];core5=[c for c in analyses['core']['conditions'] if len(c['seeds'])==5 and c['complete']];new=[c for c in analyses['new_tasks']['conditions'] if c['complete']]
 main_rows=main_arm_descriptions(data['core']);main_count=sum(r['complete_main_arms'] for r in main_rows)
 write(ROOT/'CORE_MAIN_ARMS_DESCRIPTIVE.json',{'at':now(),'scope':'Descriptive presentation of the preselected first three seeds for Base/H/budgetH/HEU. No additional inferential tests; frozen nine-arm complete-block analysis is unchanged.','conditions':main_rows})
 responses=sum(v['responses_independently_audited'] for v in audit['phases'].values())
 lines=['# 14小时模型、下游任务与E/U架构实验','',
  f'窗口：2026-09-17 21:07:10 至 2026-09-18 11:07:10（北京时间）。本轮完成并通过独立审计的新正式训练 **{fits}** 次、Base评测 **{bases}** 组、短程检查 **{smokes}** 次；另有104条计划内历史结果先核验再复用。共核验 **{responses:,}** 条新输出。','',
  f'完整配对覆盖：核心三种子主对照（Base/H/预算H/双侧）**{main_count}/16** 个模型/任务条件；核心三种子九方案全架构 **{len(core3)}/16**；untied五种子扩展 **{len(core5)}/4**；新任务三种子 **{len(new)}/16**。准备矩阵中仍有 **{len(missing)}** 个新运行未达到通过状态，全部列在文末与 [FINAL_AUDIT.json](FINAL_AUDIT.json)，不能把计划量写成完成量。','',
  'H是Transformer内部LoRA；E是输入embedding侧A-LoRA；U是输出unembedding侧A-LoRA。原模型所有权重冻结。E、U、E+U三组不含内部LoRA；H+E/H+U只增加对应一侧。下面统一三种子均值；未凑齐预定种子的单元格显示完成数，不用不同种子数量的均值混比。untied五种子单独列出。','',
  '## 模型和参数范围','',
  '|模型|原权重绑定|H参数|预算H参数|H+E+U参数|E-only|U-only|E+U（无H）|预算超额|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 budget=read(ROOT/'core/BUDGET_PLAN.json')
 for m in MODELS:
  b=budget[m];lines.append('|'+LABELS[m]+'|'+('tied' if b['tied'] else '**untied**')+'|'+'|'.join(f'{b[a]:,}' for a in ['hidden','hidden_budget','hidden_both','input','output','both','budget_excess'])+'|')
 lines+=['','预算超额为零才是严格等参数；超额不为零时对照拥有更多参数。单侧E含bias、U不含bias，E/U之间和单/双侧之间容量不同。本轮比较完整固定训练方案，不能把差异单独归为放置位置。','',
  '## 核心任务：内部LoRA与边界叠加（三种子）','',
  '|任务|模型|Base|H|预算H|H+E|H+U|H+E+U|','|---|---|---:|---:|---:|---:|---:|---:|']
 for task in TASKS['core']:
  for m in MODELS:lines.append('|'+task+'|'+LABELS[m]+'|'+'|'.join(cell(data['core'],task,m,a) for a in ['base','hidden','hidden_budget','hidden_input','hidden_output','hidden_both'])+'|')
 lines+=['','CLUENER是跨度micro-F1，WikiSQL是官方执行正确率%，均为越高越好。历史复用也严格只取同一预定三种子，与前几轮五种子均值可能不同；原始五种子结论不被改写。主对照的逐种子值、差值与复用数量见 [描述性主对照数据](CORE_MAIN_ARMS_DESCRIPTIVE.json)。主对照齐全不等于九方案齐全；该展示不增加统计检验，校正区间仍只按冻结的完整九方案分析输出。','',
  '![模型系列与叠加结果](figures/core_model_series.png)','',
  '## 核心任务：没有H的边界适配（三种子）','',
  '|任务|模型|Base|E-only|U-only|E+U|','|---|---|---:|---:|---:|---:|']
 for task in TASKS['core']:
  for m in MODELS:lines.append('|'+task+'|'+LABELS[m]+'|'+'|'.join(cell(data['core'],task,m,a) for a in ['base','input','output','both'])+'|')
 lines+=['','这些行检验只训练边界模块能否适配任务。没有设置新的同预算standalone内部LoRA对照，不能单凭优于Base就声称低预算最优。Base分数包含提示、结构输出和停止行为影响，不能解释成知识量。','',
  '![单独边界适配](figures/standalone_model_series.png)','',
  '## Untied模型五种子扩展','',
  '|任务|模型|Base|H|预算H|H+E|H+U|H+E+U|E-only|U-only|E+U|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for task in TASKS['core']:
  for m in ['qwen25_7b_base','qwen3_8b_base']:lines.append('|'+task+'|'+LABELS[m]+'|'+'|'.join(cell(data['core'],task,m,a,5) for a in ARMS)+'|')
 lines+=['','![Untied分侧叠加的种子增量](figures/untied_side_effects.png)','',
  '图中每个小点是预定配对种子，柱为均值；它不是置信区间图。五种子完整时展示五种子，否则只有完整三种子才展示三种子。缺失不被补值。所有预定对照及区间见 [CORE_CONTRASTS.md](CORE_CONTRASTS.md)。','',
  '## 新下游任务（三种子）','',
  '|任务|模型|Base|H|预算H|H+E+U|','|---|---|---:|---:|---:|---:|']
 for task in TASKS['new_tasks']:
  for m in MODELS:lines.append('|'+task+'|'+LABELS[m]+'|'+'|'.join(cell(data['new_tasks'],task,m,a) for a in ['base','hidden','hidden_budget','hidden_both'])+'|')
 lines+=['','TREC50采用完整500题、50个类别代码的两token联合概率分类正确率，不是自由生成问答。SQuAD2为固定1024条public-dev子集，回答/拒答各半，以官方归一化多参考token-F1为主，未用隐藏官方测试集。常数空答案可在该平衡子集获得50 F1，必须结合下面的可回答子集指标解读。','',
  '![新任务的叠加增量](figures/new_task_effects.png)','',
  '新任务的开发集另按相同冻结方案汇总在 [DEV_ANALYSIS.json](new_tasks/DEV_ANALYSIS.json)，不据此选择checkpoint或调整已固定超参数。','',
  '|SQuAD2模型 / 方案|总F1|EM%|可回答F1|不可回答F1|schema%|严格JSON%|','|---|---:|---:|---:|---:|---:|---:|']
 for m in MODELS:
  for arm in ['base','hidden','hidden_budget','hidden_both']:
   v=series(data['new_tasks'],'squad2',m,arm)
   if v is None:continue
   seeds=[None] if arm=='base' else range(SEEDS['squad2'],SEEDS['squad2']+3);ss=[data['new_tasks']['squad2',m,arm,s]['summary'] for s in seeds]
   lines.append('|'+LABELS[m]+' / '+ARM_LABELS[arm]+'|'+'|'.join(f"{np.mean([s[k] for s in ss]):.3f}" for k in ['primary','em_pct','answerable_f1_pct','unanswerable_f1_pct','schema_valid_pct','strict_json_pct'])+'|')
 lines+=['','## 配对统计与证据边界','',
  '这是预先固定矩阵下的广覆盖探索。核心三种子对全部176个计划对照保留Bonferroni分母；untied五种子扩展同样保留176。新任务保留32。仅完整共同种子块产生统计汇总，时间预算未完成的块不会使校正分母缩小。未校正95%种子t区间仅为描述性辅助；完整校正区间、逐种子差值与正/零/负计数分别在 [CORE_CONTRASTS.md](CORE_CONTRASTS.md) 和 [NEW_TASK_CONTRASTS.md](NEW_TASK_CONTRASTS.md)。','',
  '三种子与嵌套五种子不是独立重复，各阶段区间也不是覆盖整个历史研究所有选择的统一确认性检验。公共数据预训练暴露没有被排除。核心两任务因旧正结果而选择；新任务在首个输出前冻结。共同LR、小数据、最终单checkpoint、模型宽度/深度/权重绑定差异，以及分支缩放/bias/dropout限制机制和规模因果解释。保留所有负值和不确定结果，不能从少数正例外推普遍收益。','',
  '## 审计与复用','',
  f"成功的新输出记录共{responses:,}条。生成记录重新解码并按固定评分核对摘要；有效SQL预测用官方执行器核对，有效QA答案用作者EM/F1核对。TREC分类记录独立复算候选预测和指标；每次评测另对最短/最长两个例子运行独立完整前向，核对缓存联合概率与argmax，不把它表述成对全体样本重跑模型。新任务独立复核6132条作者记录、49056条token；核心55352条token在最后再次编码。选定104条历史运行的初始/最终adapter哈希重新检查，复用输出不用于初始化任何新模型。",'',
  '每个新训练核验可训练参数白名单、实际权重绑定指针、零残差与独立loss、原始权重无梯度且前后digest一致、指定adapter分组确实更新、优化器FP32和保存恢复严格一致。跨方案共同H/E/U初始化及样本顺序另在 [PAIR_AUDIT_PROGRESS.json](PAIR_AUDIT_PROGRESS.json) 检查。模型身份和全部冻结代码/数据在最终审计重新哈希。','',
  '格式、内容、停止行为和事后描述性交互见 [诊断附录](DESCRIPTIVE_DIAGNOSTICS.md)；实际参数化与对照能回答的问题见 [架构说明](ARCHITECTURE_READOUT.md)。诊断附录不增加显著性检验或机制因果结论。另在 [矩阵审计](BOUNDARY_ALGEBRA_AUDIT.json) 用实际权重和固定CLUENER首种子adapter核对抽取词表行上的分支/合并恒等式；这是CPU局部数值检查，不是全词表或端到端合并评测，缺失的固定检查单元明确列出。','',
  '逐次运行与资源记录见 [STATE.json](STATE.json) 和三个版本的scheduler日志；原协议见 [PLAN.md](PLAN.md)、[核心协议](core/PROTOCOL.md)、[新任务协议](new_tasks/PROTOCOL.md)。22:36因共享显存变化采用 [资源准入修订](RESOURCE_AMENDMENT_V2.json)，让小模型使用剩余显存；六个运行保持原PID而未重启。01:09另有 [覆盖排队修订](COVERAGE_AMENDMENT_V3.json)：40组尚未启动的1.7B/4B主对照提前，52组额外untied种子后排，八个在跑实验保留PID与进度。修订时已看到部分核心分数；规则依据覆盖缺口与耗时、按完整类别调整，不能表述为原始排队顺序始终未变。所有模型/任务/种子仍留在完整计划并报告缺失，不按涨跌选运行。两个接管中的非子进程退出码都记为unknown，成功以完整保存结果及新独立审计为准。科学拟合、评分、种子、统计族与时限未改；共享环境wall-clock不能作为方法速度优越性的依据。','',
  'TREC开发/测试方向差异另作 [逐例错误核对](TREC_ERROR_ANALYSIS.md)：保留所有完整模型块与50个类别，统计原来答错后改善、原来答对后退步的例数。该分析在看到8B反向结果后设计，只用于描述，未新增检验或调整实验。','',
  '## 未完成与失败记录','',
  '|阶段 / 状态|数量|','|---|---:|']
 count=collections.Counter((r['name'].split('/')[0],r['status']) for r in missing)
 if count:
  for (phase,status),n in sorted(count.items()):lines.append(f'|{phase} / {status}|{n}|')
 else:lines.append('|无：全部新队列任务通过|0|')
 lines+=['','全部未完成任务的模型、任务、架构、种子和状态保留在FINAL_AUDIT.json；发生训练失败、审计失败或截止中断时不将其当作成功结果。未完成的比较不给出方法优劣结论。','']
 (ROOT/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines))
 for phase,title in [('core','CORE_CONTRASTS'),('new_tasks','NEW_TASK_CONTRASTS')]:
  ls=['# '+title,'','All values use the frozen phase analysis; unadjusted intervals are descriptive.','',
      '|Task / model / n|Treatment − control|Mean|SD of paired differences|Raw differences|Unadjusted95%|Corrected95%|','|---|---|---:|---:|---|---|---|']
  for c in analyses[phase]['conditions']:
   if not c['complete']:continue
   for x in c['comparisons']:
    family=c['family'];ci=x[f'bonferroni{family}_95_ci'];raw=x['descriptive_unadjusted_95_ci'];treatment=x.get('treatment','hidden_both')
    ls.append(f"|{c['task']} / {LABELS[c['model']]} / {len(c['seeds'])}|{ARM_LABELS[treatment]} − {ARM_LABELS[x['control']]}|{x['mean']:+.4f}|{x['sd']:.4f}|"+', '.join(f'{v:+.4f}' for v in x['differences'])+f"|[{raw[0]:+.4f},{raw[1]:+.4f}]|[{ci[0]:+.4f},{ci[1]:+.4f}]|")
  (ROOT/(title+'.md')).write_text('\n'.join(ls)+'\n')
 with (ROOT/'ALL_AVAILABLE_TEST_RUNS.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['phase','task','model','arm','seed','primary','reused','summary'])
  for phase,ds in data.items():
   for k,v in sorted(ds.items(),key=lambda x:str(x[0])):w.writerow([phase,*k,v['primary'],v['reused'],str(v['path'].relative_to(ROOT))])
 write(ROOT/'REPORT_STATUS.json',{'at':now(),'status':'passed','new_fits':fits,'new_base_evals':bases,'smokes':smokes,'core_main_three_seed_blocks':main_count,'core_complete_three_seed_blocks':len(core3),'core_complete_five_seed_blocks':len(core5),'new_task_complete_three_seed_blocks':len(new),'uncompleted_jobs':len(missing),'new_responses':responses})
 print('Wrote full report and complete seed-level CSV.')
if __name__=='__main__':main()
