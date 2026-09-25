"""Present the fixed analysis; no model selection or additional significance tests."""
import csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent/'gemma_transfer_20260918'
ARMS=['hidden','hidden_budget','hidden_both']
LABEL={'hidden':'普通 H','hidden_budget':'严格等参 H','hidden_both':'H+E+U'}
EN={'hidden':'H','hidden_budget':'Exact-budget H','hidden_both':'H+E+U'}
TASKS=['wikisql','trec50']
def read(p):return json.loads(p.read_text())
def ci(v):return f'[{v[0]:+.4f}, {v[1]:+.4f}]'
def conclusion(c):
 lo,hi=c['bonferroni4_ci95']
 if lo>0:return '本轮校正种子区间高于零'
 if hi<0:return '本轮校正种子区间低于零'
 return '本轮校正种子区间跨零，增量方向仍不确定'
def main():
 assert read(ROOT/'FINAL_AUDIT.json')['status']=='passed' and not (ROOT/'SEAL.json').exists()
 a=read(ROOT/'ANALYSIS.json');sel=read(ROOT/'SELECTION.json');boot=read(ROOT/'CLUSTER_BOOTSTRAP.json');audit=read(ROOT/'FINAL_AUDIT.json')
 with (ROOT/'ALL_RESULTS.csv').open('w') as f:
  fields=['stage','task','arm','seed','lr','primary','lf_correct_pct','query_valid_pct','macro_f1_all50','coarse_accuracy','run']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for j in read(ROOT/'STATE.json')['jobs']:
   s=j['spec']
   if s['stage']=='smoke':continue
   split='dev' if s['stage']=='search' else 'confirm';v=read(ROOT/'evaluations'/s['name']/split/'SUMMARY.json')
   row={k:s[k] for k in ['stage','task','arm','seed','lr']};row.update(run=s['name'])
   row.update({k:v[k] for k in fields if k in v and k not in row});writer.writerow(row)
 colors={'hidden':'#526176','hidden_budget':'#d58421','hidden_both':'#166c80'}
 figures=ROOT/'figures';figures.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,task in zip(axes,TASKS):
  for arm in ARMS:
   rs=sorted(sel['scores'][task][arm],key=lambda r:r['lr'])
   ax.plot([r['lr'] for r in rs],[r['mean'] for r in rs],marker='o',color=colors[arm],label=EN[arm])
  ax.set_xscale('log');ax.set_xlabel('Learning rate (all trainable groups)');ax.set_ylabel('Development accuracy (%)')
  ax.set_xticks([5e-5,1e-4,2e-4,3e-4,4e-4,8e-4],['5e-5','1e-4','2e-4','3e-4','4e-4','8e-4'],fontsize=8,rotation=25)
  ax.set_title(task.upper()+' — two development seeds');ax.grid(alpha=.18);ax.legend(fontsize=8)
 fig.savefig(figures/'development_search.pdf');fig.savefig(figures/'development_search.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
 for ax,field,title in [(axes[0],'bonferroni4_ci95','Paired training seeds (5 seeds)'),(axes[1],'bonferroni4_percentile_ci95','Data clusters (fixed five models)')]:
  cs=a['comparisons'] if ax is axes[0] else boot['comparisons']
  for i,c in enumerate(cs):
   m=c['mean_pp'];lo,hi=c[field]
   ax.hlines(i,lo,hi,color='#166c80',lw=2);ax.plot([lo,hi],[i,i],marker='|',color='#166c80',linestyle='none');ax.plot(m,i,'o',color='#166c80')
  ax.axvline(0,color='#777777',linestyle='--',lw=1);ax.set_yticks(range(len(cs)),[c['task'].upper()+': H+E+U − '+EN[c['b']] for c in cs],fontsize=8)
  ax.invert_yaxis();ax.set_title(title+'\nFamily-4 95% intervals',fontsize=10);ax.set_xlabel('Accuracy difference (percentage points)');ax.grid(axis='x',alpha=.18)
 fig.savefig(figures/'confirmation_intervals.pdf');fig.savefig(figures/'confirmation_intervals.png',dpi=180);plt.close(fig)
 lines=['# Gemma-2-9B Base 跨模型族叠加实验','','本轮两任务、三种适配方法的开发选参、五种子验证与最终审计均已完成。Gemma 未新增稳定叠加优势：WikiSQL 相对普通 / 严格等参为 +0.4590 / −0.2637 pp，TREC 为 −1.6000 / −1.4000 pp。四项校正种子区间均跨零；固定五个模型时，TREC 两项校正数据重采样区间均低于零。此前 Qwen 和 Llama 的正、负结果继续保留。',
  '', '## 设置与公平性', '',
  '普通 H 为 27,009,024 个训练参数；严格等参 H 与 H+E+U 都为 27,241,984 个。原模型权重始终冻结；仅指定的 LoRA/边界模块接受更新。E/U 各 rank16，E 有偏置、U 无偏置。冻结 E/U 物理权重共享，但独立适配后的有效矩阵不要求继续共享。',
  '', '严格等参控制的是可训练参数数量。内部 H 为 alpha/r=2、dropout=0.05，E/U 为 alpha/r=8、dropout=0；参数放置、分支缩放、初始化及偏置结构并未被逐项隔离。比较验证的是这些固定完整方案在同等搜索次数下的表现，不能把差异单独归因于 E/U 的放置位置。',
  '', '三种方法在两个任务上均使用同一组六个学习率、各两个开发种子；每任务每方法独立选出开发均值最高配置，再用五个新种子训练和验证。共 72 次开发训练、30 次确认训练、2 项 Base 评测和 10 次技术试跑。每次正式训练均为 2,048 条样本、单轮 64 次更新；不比较跨任务指标或宣称搜索达到全局最优。',
  '', '采用官方预训练 Gemma-2-9B。原始权重为 FP32，所有组共同使用其 BF16 舍入后的冻结权重，评测时转为 FP32 计算；不能写成原始未舍入 FP32 checkpoint 的分数。保留原生 embedding 缩放和两处 soft-capping，E 适配放在原生 embedding 缩放之前。',
  '', '| 任务 | 方法 | 选定学习率 | 开发集两种子均值 |','|---|---|---:|---:|']
 for task in TASKS:
  for arm in ARMS:
   c=sel['selected'][task][arm];r=next(x for x in sel['scores'][task][arm] if x['id']==c['id'])
   lines.append(f'| {task} | {LABEL[arm]} | {c["lr"]:g} | {r["mean"]:.4f} |')
 lines+=['', '开发阶段存在一个必须保留的负例：TREC 叠加组在 LR=8e-4、seed=7400 时将全部 256 条问题预测为代码 45，准确率 2.734375%；另一种子为 83.593750%。该次训练 loss 偏高、梯度峰值较大，参数与评分审计通过；这与优化不稳定一致，但不能直接归因于 E、U、裁剪或 soft-capping。全部 36 次 TREC 开发拟合的事后描述性核对见 [开发优化敏感性](DEVELOPMENT_INSTABILITY_ZH.md)。未因此改变网格、种子或选择规则。',
  '', '## 五种子确认结果','','WikiSQL 为执行正确率，TREC50 为完整两 token 候选概率的分类正确率；数值为百分比。Base 未训练，只评估一次，不构造虚假的五种子方差。',
  '', '| 任务 | 方法 | 均值 | 种子标准差 |','|---|---|---:|---:|']
 for task in TASKS:
  lines.append(f'| {task} | Base | {a["base"][task]["primary"]:.4f} | — |')
  for arm in ARMS:
   rs=a['results'][task][arm];lines.append(f'| {task} | {LABEL[arm]} | {statistics.mean(r["primary"] for r in rs):.4f} | {statistics.stdev(r["primary"] for r in rs):.4f} |')
 lines+=['','| 任务 | 叠加相对对照 | 平均差（百分点） | 五个种子差值 | 校正种子 95% 区间 | 正向种子 |','|---|---|---:|---|---|---:|']
 for c in a['comparisons']:
  lines.append(f'| {c["task"]} | {LABEL[c["b"]]} | {c["mean_pp"]:+.4f} | '+', '.join(f'{v:+.4f}' for v in c['paired_seed_deltas_pp'])+f' | {ci(c["bonferroni4_ci95"])} | {c["positive_seeds"]}/5 |')
 lines+=['','## 如何解释','','本轮四项比较使用同一个预先指定的多重比较族。','']
 for c in a['comparisons']:lines.append(f'- {c["task"]} 相对{LABEL[c["b"]]}：均值 {c["mean_pp"]:+.4f} 个百分点，{conclusion(c)}。')
 lines+=['','| 任务 | 叠加相对对照 | 条件于已训练模型的校正 cluster bootstrap 区间 |','|---|---|---|']
 for c in boot['comparisons']:lines.append(f'| {c["task"]} | {LABEL[c["b"]]} | {ci(c["bonferroni4_percentile_ci95"])} |')
 for c in boot['comparisons']:
  if c['bonferroni4_percentile_ci95'][1]<0:
   lines+=['',f'{c["task"]} 相对{LABEL[c["b"]]}的校正数据重采样区间低于零：在固定这五个已训练模型时，负向差异对题目抽样具有一致性。这不替代训练种子的不确定性区间，也不证明所有未来训练都会下降。']
 lines+=['','种子区间描述固定数据和已选配置下的训练随机性；bootstrap 描述固定这五个模型时的表格/问题抽样变化。两者不等同，也不覆盖共同的选参、训练和数据不确定性；五个种子不足以验证 t 区间分布假设。跨零不等于证明没有收益，均值为正也不等于确认稳定优越。',
  '', '数据与提示来自此前已校验的共享公开基准。模型文件、数据、代码可复用，旧模型预测和训练后的适配器没有复用。确认集此前已被其他模型族评估，不能称为项目从未见过的数据，也不能排除预训练暴露。任务数量、样本预算和固定搜索范围均限制外推。',
  '', '## 描述性诊断','','WikiSQL 的逻辑形式、合法查询、严格 JSON 与截断率，以及 TREC 的 macro F1 和粗类别正确率保存在 `ANALYSIS.json`；全部搜索与逐种子结果见 `ALL_RESULTS.csv`。`PAIRED_CASES.json` 记录恢复/回退样本数。合法性分区使用固定全体题目分母，不构成机制的因果分解。',
  '', 'WikiSQL 相对严格等参组，叠加的合法查询率较高（99.8047% 对 99.5313%），但逻辑形式匹配较低（78.9941% 对 79.8340%），执行正确率也较低；双方查询均有效样本贡献 −0.4980 pp，其他合法性分区贡献 +0.2344 pp。故本轮均值下降不能简单解释为格式更差。TREC 相对等参组，macro F1 从 68.9611 降至 67.0742，粗类别正确率从 96.24% 降至 95.32%；这些是描述性次要指标，未另做显著性检验。',
  '', '| 任务 | 方法 | 平均触发裁剪步数 / 64 | 最后八步训练 loss 均值 | H 梯度能量占比 | E 占比 | U 占比 |','|---|---|---:|---:|---:|---:|---:|']
 for task in TASKS:
  for arm in ARMS:
   ds=[r['diagnostics'] for r in a['results'][task][arm]]
   gs={g:100*statistics.mean(d['mean_group_gradient_energy_share'][g] for d in ds) for g in ['hidden','input','output']}
   lines.append(f'| {task} | {LABEL[arm]} | {statistics.mean(d["clipped_steps"] for d in ds):.2f} | {statistics.mean(d["tail8_training_loss"] for d in ds):.5f} | {gs["hidden"]:.2f}% | {gs["input"]:.2f}% | {gs["output"]:.2f}% |')
 selected_code_counts=[]
 for r in a['results']['trec50']['hidden_both']:
  rs=[json.loads(v) for v in (ROOT/'evaluations'/r['run']/'confirm/responses.jsonl').read_text().splitlines() if v]
  selected_code_counts.append(len({v['predicted_code'] for v in rs}))
 lines+=['',f'选中配置的 TREC 叠加组五个确认种子分别预测了 {selected_code_counts} 个不同类别；未出现全部预测同一类别的情形。开发阶段 LR=8e-4 的崩塌属于未选中的配置，不能直接作为确认集差异的原因。这项类别计数是事后描述性核对，不增加显著性检验或改变任何选择。',
  '', '这些数值只描述选中配置的优化过程。梯度能量集中、裁剪频率或较低训练 loss 本身不能证明收益来源，也不能独立诊断某个模块有害。',
  '', '## 审计范围','',
  '首版技术预检在高学习率 TREC 的缓存评分一致性检查中停止：最大联合 logprob 差为 0.000336647，超过预设 0.0002；两条检查样本的类别选择相同。零更新复核显示原生完整前向与原生 causal-LM loss 的联合概率一致。正式搜索开始前，统一改用完整前向，保持阈值、训练代码、数据、搜索候选和统计方案不变，重复全部十项预检。首版十次试跑和失败记录完整封存在相邻 `gemma_transfer_20260918_preflight_v1`，不并入正式结果。',
  '',
  f'最终审计通过 {audit["jobs"]} 项运行、{audit["responses"]:,} 条输出、{audit["official_valid_query_checks"]:,} 次官方有效 SQL 检查、{audit["native_candidate_logprob_checks"]:,} 项原生候选概率核对、{audit["tokens_reencoded"]:,} 条重编码和 {audit["actual_shared_tensors_compared"]:,} 个实际共享初始化张量比较。',
  '', '原生候选概率核对的范围是每次 TREC 评测预先指定的两条检查样本及其全部 50 个候选，并非对每一道题重新运行独立原生参考；全量输出另行核对概率有限性、argmax、标签和汇总。',
  '', '完整协议见 `PROTOCOL.md`；最终逐文件封存和独立验证以 `SEAL.json` 及目录外验证日志为准。',
  '', '![开发集搜索](figures/development_search.png)','','![两类不确定性区间](figures/confirmation_intervals.png)']
 (ROOT/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')
 print('CSV, PDF/PNG figures and Chinese interpretation generated',flush=True)
if __name__=='__main__':main()
