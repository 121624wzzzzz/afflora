"""Present the fixed analysis; no model selection or additional significance tests."""
import collections,csv,hashlib,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();ROOT=args.root.resolve()
ARMS=['hidden','hidden_budget','hidden_both']
LABEL={'hidden':'普通 H','hidden_budget':'严格等参 H','hidden_both':'H+E+U'}
EN={'hidden':'H','hidden_budget':'Exact-budget H','hidden_both':'H+E+U'}
TASKS=['wikisql','trec50']
def read(p):return json.loads(p.read_text())
def ci(v):return f'[{v[0]:+.4f}, {v[1]:+.4f}]'
def conclusion(c):
 lo,hi=c['bonferroni16_ci95']
 if lo>0:return '本轮校正种子区间高于零'
 if hi<0:return '本轮校正种子区间低于零'
 return '本轮校正种子区间跨零，增量方向仍不确定'
def initial_loss_description():
 grouped=collections.defaultdict(list);inputs={}
 for job in read(ROOT/'STATE.json')['jobs']:
  s=job['spec']
  if s['stage'] not in ['search','confirmation']:continue
  assert job['state']=='passed'
  p=ROOT/'checkpoints'/s['name']
  for filename in ['INITIALIZATION.json','TRAINING.json','TRAIN_ORDER.json']:
   f=p/filename;inputs[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  init=read(p/'INITIALIZATION.json');h=read(p/'TRAINING.json')['history']
  grouped[(s['stage'],s['task'],s['seed'])].append({
   'run':s['name'],'arm':s['arm'],'lr':s['lr'],'first_batch_loss_before_update':h[0]['loss'],
   'recorded_shared_hidden_initialization_sha256':init['shared_hidden_initialization_sha256'],
   'recorded_full_initialization_sha256':init['initialization_sha256'],
   'recorded_frozen_before':init['frozen_before'],
   'order_sha256':inputs[str((p/'TRAIN_ORDER.json').relative_to(ROOT))]})
 groups=[]
 for (stage,task,seed),rs in sorted(grouped.items()):
  for field in ['recorded_shared_hidden_initialization_sha256','recorded_frozen_before','order_sha256']:
   assert len({r[field] for r in rs})==1,(stage,task,seed,field)
  for arm in ARMS:assert len({r['recorded_full_initialization_sha256'] for r in rs if r['arm']==arm})==1
  vs=[r['first_batch_loss_before_update'] for r in rs]
  groups.append({'stage':stage,'task':task,'seed':seed,'runs':rs,'min':min(vs),'max':max(vs),'range':max(vs)-min(vs)})
 report={'description':'Post-hoc descriptive first-batch loss ranges for all 102 formal fits, grouped by stage/task/seed; no new fits, selection changes or inferential tests.',
  'limitation':'Same recorded initialization/order hashes do not guarantee identical accelerated BF16 execution. Equality of aggregate losses does not prove bitwise equality of every intermediate or future update. Actual initial tensors and order are independently checked by the frozen final audit.',
  'inputs_sha256':inputs,'fits':sum(len(g['runs']) for g in groups),'groups':groups,
  'groups_with_nonzero_range':sum(g['range']>0 for g in groups),'max_range':max(g['range'] for g in groups)}
 assert report['fits']==102 and len(groups)==14
 (ROOT/'INITIAL_LOSS_REPRODUCIBILITY.json').write_text(json.dumps(report,indent=2)+'\n')
 assert report['groups_with_nonzero_range']==0 and report['max_range']==0
 return report
def main():
 assert read(ROOT/'FINAL_AUDIT.json')['status']=='passed' and not (ROOT/'SEAL.json').exists()
 a=read(ROOT/'ANALYSIS.json');sel=read(ROOT/'SELECTION.json');boot=read(ROOT/'CLUSTER_BOOTSTRAP.json');audit=read(ROOT/'FINAL_AUDIT.json')
 initial=initial_loss_description()
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
 for ax,field,title in [(axes[0],'bonferroni16_ci95','Paired training seeds (5 seeds)'),(axes[1],'bonferroni16_percentile_ci95','Data clusters (fixed five models)')]:
  cs=a['comparisons'] if ax is axes[0] else boot['comparisons']
  for i,c in enumerate(cs):
   m=c['mean_pp'];lo,hi=c[field]
   ax.hlines(i,lo,hi,color='#166c80',lw=2);ax.plot([lo,hi],[i,i],marker='|',color='#166c80',linestyle='none');ax.plot(m,i,'o',color='#166c80')
  ax.axvline(0,color='#777777',linestyle='--',lw=1);ax.set_yticks(range(len(cs)),[c['task'].upper()+': H+E+U − '+EN[c['b']] for c in cs],fontsize=8)
  ax.invert_yaxis();ax.set_title(title+'\nFamily-16 95% intervals',fontsize=10);ax.set_xlabel('Accuracy difference (percentage points)');ax.grid(axis='x',alpha=.18)
 fig.savefig(figures/'confirmation_intervals.pdf');fig.savefig(figures/'confirmation_intervals.png',dpi=180);plt.close(fig)
 model_identity=read(ROOT/'MODEL_IDENTITY_AUDIT.json');budget=read(ROOT/'BUDGET_ALLOCATION.json')
 lines=['# '+model_identity['repo']+' 叠加实验','',
  '两任务、三组、同一学习率搜索和五种子确认；具体结论由完整结果决定。所有原模型参数冻结，仅指定适配器训练。本轮固定FLA执行配置与PyTorch数值设置，每次拟合均与独立进程预存的首批16项损失精确比较，并检查重复反向的全部梯度相同。通过不等于全过程跨设备确定性；旧设置结果完整保留。',
  '', '本轮扩展的是Qwen家族内采用线性注意力与全注意力混合结构的新架构，不应计作新增独立模型家族。',
  '', f'H 为{budget["hidden_parameters"]:,}个参数；严格等参 H 与 H+E+U 均为{budget["hidden_parameters"]+budget["extra"]:,}个参数。原生输入输出权重共享={model_identity["tie_word_embeddings"]}。H 覆盖全部指定文本解码投影；E/U各rank16，E有偏置、U无偏置。视觉模块与MTP分支未参与纯文本任务。区间按四个修正设置的尺寸共16项主要比较校正。',
  '', '各任务各组均搜索6个学习率、2个开发种子，开发集选参后固定配置，再跑5个新训练种子。每次正式训练2048条样本、64次更新，使用最后checkpoint。此前模型的训练结果或预测未复用；共享公开确认集此前已被其他模型评估。',
  '', '|任务|方法|学习率|均值|种子标准差|','|---|---|---:|---:|---:|']
 for task in TASKS:
  lines.append(f'|{task}|Base|—|{a["base"][task]["primary"]:.4f}|—|')
  for arm in ARMS:
   v=[r['primary'] for r in a['results'][task][arm]]
   lines.append(f'|{task}|{LABEL[arm]}|{sel["selected"][task][arm]["lr"]:g}|{statistics.mean(v):.4f}|{statistics.stdev(v):.4f}|')
 lines+=['','WikiSQL为执行正确率，TREC50为候选两token完整概率分类正确率，均为百分比。','',
  '|任务|叠加相对对照|平均差（百分点）|五种子差值|校正种子95%区间|','|---|---|---:|---|---|']
 for c in a['comparisons']:
  ds=', '.join(f'{v:+.4f}' for v in c['paired_seed_deltas_pp'])
  lines.append(f'|{c["task"]}|{LABEL[c["b"]]}|{c["mean_pp"]:+.4f}|{ds}|{ci(c["bonferroni16_ci95"])}|')
 lines+=['']
 for c in a['comparisons']:lines.append(f'- {c["task"]} 相对{LABEL[c["b"]]}：{conclusion(c)}。')
 lines+=['','|任务|叠加相对对照|固定五个模型的数据重采样校正区间|','|---|---|---|']
 for c in boot['comparisons']:lines.append(f'|{c["task"]}|{LABEL[c["b"]]}|{ci(c["bonferroni16_percentile_ci95"])}|')
 if all(c['mean_pp']<0 for c in a['comparisons']) and all(c['bonferroni16_ci95'][0]<=0<=c['bonferroni16_ci95'][1] for c in a['comparisons']):
  lines[2]='本轮两个任务的叠加组五种子均值均低于普通和严格等参LoRA，未新增稳定叠加优势；四项校正种子区间均跨零，不能证明普遍无效或等效。所有原模型参数冻结，仅指定适配器训练。'
 lines+=['','以下次要指标仅作描述，不增加显著性检验或改变主指标：','',
  '|任务|方法|逻辑形式匹配% / macro F1|查询有效率% / 粗类别正确率%|','|---|---|---:|---:|']
 for task in TASKS:
  k1,k2=('lf_correct_pct','query_valid_pct') if task=='wikisql' else ('macro_f1_all50','coarse_accuracy')
  for arm in ARMS:
   rs=a['results'][task][arm]
   lines.append(f'|{task}|{LABEL[arm]}|{statistics.mean(r[k1] for r in rs):.4f}|{statistics.mean(r[k2] for r in rs):.4f}|')
 cases=read(ROOT/'PAIRED_CASES.json')['comparisons']
 for arm in ['hidden','hidden_budget']:
  rs=cases[f'wikisql hidden_both minus {arm}']
  both=statistics.mean(r['both_valid_net_pp'] for r in rs);other=statistics.mean(r['other_validity_net_pp'] for r in rs)
  lines+=['',f'WikiSQL相对{LABEL[arm]}：双方查询均有效样本贡献{both:+.4f}pp，其他查询有效性分区贡献{other:+.4f}pp；分母始终是完整确认集。这是描述性分区，不是格式与内容的因果分解。']
 lines+=['','种子区间与固定模型的数据重采样区间回答不同问题，不包含共同的训练、数据和选参不确定性；五种子t区间依赖小样本分布假设。跨零不等于证明零收益。严格等参和同等搜索次数也没有单独隔离位置、alpha、dropout、偏置和初始化的作用，不能推断所有任务或架构均会获益。',
  '',f'最终审计：{audit["jobs"]}项运行，{audit["responses"]:,}条输出，{audit["official_valid_query_checks"]:,}次官方有效SQL复算，{audit["native_candidate_logprob_checks"]:,}项原生候选概率核对，{audit["tokens_reencoded"]:,}条重编码，{audit["actual_shared_tensors_compared"]:,}个实际共享初始化张量比较。原生候选概率核对每次评测的预定短/长两例，全部输出另核对有限性、argmax、标签与汇总。',
  '', '完整设置见PROTOCOL.md；所有开发和逐种子结果见ALL_RESULTS.csv，次要指标和优化诊断见ANALYSIS.json；封存后以SEAL.json和目录外独立验证日志为准。',
  '', '![开发搜索](figures/development_search.png)','', '![确认区间](figures/confirmation_intervals.png)']
 diagnostic=ROOT/'DEV_SEED7600_TREC_DIAGNOSTICS.json'
 if diagnostic.exists():
  rows=read(diagnostic)['results'];highest=max(r['lr'] for r in rows)
  high={r['arm']:r for r in rows if r['lr']==highest}
  lines+=['',f'开发阶段补充描述：首个开发种子7600中，TREC50最高学习率{highest:g}的H / 等参H / H+E+U正确率分别为'+
          ' / '.join(f'{high[arm]["primary"]:.4f}%' for arm in ARMS)+'；最后8步训练损失均值分别为'+
          ' / '.join(f'{high[arm]["last8_mean_loss"]:.4f}' for arm in ARMS)+'。',
          '全部18项首种子TREC开发结果的训练与预测分布见DEV_SEED7600_TREC_DIAGNOSTICS.json。该事后描述不参与选参，也不能把梯度大小解释为E/U的因果作用；须与确认阶段结果区分。']
 second_diagnostic=ROOT/'DEV_SEED7601_TREC_DIAGNOSTICS.json'
 if second_diagnostic.exists():
  rows=read(second_diagnostic)['results'];highest=max(r['lr'] for r in rows);high={r['arm']:r for r in rows if r['lr']==highest}
  lines+=['',f'同样补充核对第二个开发种子7601的全部18项TREC拟合：最高学习率{highest:g}的H / 等参H / H+E+U正确率分别为'+
          ' / '.join(f'{high[arm]["primary"]:.4f}%' for arm in ARMS)+'；最后8步训练损失均值分别为'+
          ' / '.join(f'{high[arm]["last8_mean_loss"]:.4f}' for arm in ARMS)+'。完整记录见DEV_SEED7601_TREC_DIAGNOSTICS.json。两个开发种子的差异不能单独区分初始化、样本顺序、数值执行及其相互作用。']
 replay=ROOT/'INITIAL_FORWARD_REPLAY.json'
 if replay.exists():
  r=read(replay)
  lines+=['',f'额外数值复现限制：TREC50 seed7600的8e-4叠加运行首步更新前记录损失为{r["recorded_source_first_loss"]:.8f}；相同初始参数与前32条样本的GPU{r["gpu"]}只读重放（含两次保留零初始化E/U、一次移除E/U）分别为'+
          ' / '.join(f'{m["first_batch_loss"]:.8f}' for m in r['measurements'])+'。本次重放未复现原先的小差异，具体来源尚未定位，不能宣称加速BF16训练严格逐位确定，也不能据此确定后续退化的原因；原运行保留且未替换。详见INITIAL_FORWARD_REPLAY.json。']
 lines+=['',f'所有102项正式拟合另按阶段/任务/种子分为14组核对首批更新前损失：{initial["groups_with_nonzero_range"]}组的记录存在非零数值范围，最大范围{initial["max_range"]:.8f}。完整记录及输入哈希见INITIAL_LOSS_REPRODUCIBILITY.json；该统计为数值复现描述，不是新增效果检验。']
 (ROOT/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')
 print('CSV, figures and interpretation generated',flush=True)
if __name__=='__main__':main()
