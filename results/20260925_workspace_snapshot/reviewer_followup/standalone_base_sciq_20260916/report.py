"""Reporting only: no training, selection or scoring changes."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import *

LABELS={'qwen3_06b_base':'Qwen3-0.6B-Base','qwen25_15b_base':'Qwen2.5-1.5B (Base)'}
ARM_LABELS={'input':'仅输入 A-LoRA','output':'仅输出 A-LoRA','hidden_r8':'内部 LoRA r8（较大预算）'}

def main():
    r=read(HERE/'RESULTS.json');audit=read(HERE/'FINAL_AUDIT.json');assert audit['status']=='passed'
    lines=['# 从 Base 出发的单侧 A-LoRA 下游适配','',
        '本轮直接使用官方预训练 Base checkpoint。A-LoRA 两个实验臂每次只训练输入或输出边界的一处适配器；另设独立的常规内部 LoRA r8 对照。所有原始权重均冻结，不复用已经训练的 adapter。',
        '', '研究问题是科学选择题上的监督任务适配，不是完整通用指令后训练、RLHF 或跨任务能力保持。两模型分别用三个学习率做验证集选择，再用五个新种子确认。当前 SciQ 测试集在之前研究中已经查看过。',
        '', '## 主结果','',
        '998道预先固定的干净测试题，原始选项顺序，适配器为五种子均值。选项准确率比较四个标签的模型分数；严格生成要求实际 greedy 输出去掉首尾空白后恰好是正确字母。',
        '', '| Base 模型 | 方法 | 可训练参数 | 选项准确率 % | 严格生成准确率 % | 格式有效率 % | 长度上限率 % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for m,out in r['models'].items():
        entries=[('未适配 Base',0,out['base'])]+[(ARM_LABELS[a],out['arms'][a]['parameters'],out['arms'][a]) for a in ARMS]
        for name,params,v in entries:
            lines.append(f"| {LABELS[m]} | {name} | {params:,} | {v['primary']['accuracy']:.3f} | {v['generation']['strict_accuracy']:.3f} | {v['generation']['valid_answer_rate']:.3f} | {v['generation']['length_cap_rate']:.3f} |")
    lines += ['', '严格生成分同时受到内容与格式影响，不能把格式学习全部解释为知识增长；尤其不能把 Base 的低严格生成分解释为没有相应知识。选项准确率作为共同主指标，单独检验该固定输入协议下的答案选择。',
        '', '## 不确定性与逐种子确认','',
        '下表均为相对同一未适配 Base 的选项准确率差（百分点）。种子区间使用预定12个主对比的 Bonferroni 校正；题目区间是以五个拟合模型为条件的10,000次题目 bootstrap，名义95%。两者回答不同的不确定性。完整12个对比及逐种子结果见 RESULTS.json。',
        '', '| 模型 | 方法 | 平均差 pp | 校正种子区间 | 题目区间 | 两个主指标均通过种子标准 |','|---|---|---:|---|---|---|']
    for c in r['primary_contrasts']:
        if c['metric']!='candidate_accuracy':continue
        e=c['effect_pp'];lo,hi=e['family12_ci95'];ql,qh=c['question_bootstrap95_conditioned_on_fitted_seeds']
        passed=r['models'][c['model']]['arms'][c['arm']]['passes_both_task_metrics']
        lines.append(f"| {LABELS[c['model']]} | {ARM_LABELS[c['arm']]} | {e['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{ql:+.3f}, {qh:+.3f}] | {'是' if passed else '否'} |")
    lines += ['', '未通过正收益标准不等于严格无效，也不是通过非劣性检验。种子区间与题目区间不能互相替代。',
        '', '## 输入顺序敏感性','',
        '四种循环选项顺序平均，3992条干净候选评分；它是预定的支持分析，不替换 canonical 主结果。',
        '', '| 模型 | 未适配 Base | 输入 A-LoRA | 输出 A-LoRA | 内部 LoRA r8 |','|---|---:|---:|---:|---:|']
    for m,o in r['models'].items():
        vals=[o['base']['rotation_average']['accuracy']]+[o['arms'][a]['rotation_average']['accuracy'] for a in ARMS]
        lines.append('| '+LABELS[m]+' | '+' | '.join(f'{v:.3f}%' for v in vals)+' |')
    lines += ['', '## 更大预算对照','',
        '内部 LoRA 与 A-LoRA 从同一 Base 出发，使用相同数据、训练步数、学习率搜索次数和确认种子。它在模型内部的多处投影上训练更多参数，因此仅作为较大预算性能参照，不是同参数预算方法优劣检验。', '']
    for m,o in r['models'].items():
        h=o['arms']['hidden_r8'];diff=r['alora_minus_larger_hidden_pp'][m]
        lines.append(f"- {LABELS[m]}：内部 LoRA 参数 {h['parameters']:,}，选项准确率 {h['primary']['accuracy']:.3f}%。输入/输出 A-LoRA 与其相差 {diff['input']['candidate_accuracy']:+.3f} / {diff['output']['candidate_accuracy']:+.3f} pp。")
    lines += ['', '## Base 与已经后训练的 checkpoint','',
        '以下复用上一轮封存的 post-trained 起点结果。相同系列/规模、问题、输入 token 序列、监督目标、训练步数和学习率搜索预算；两轮确认种子不同。因此这是描述性阶段比较，不能据此确定机制或进行未经预定的配对显著性推断。',
        '', '| 系列 | Base 未适配 | Base + 输入 | Base + 输出 | Post-trained 未适配 | Post-trained + 输入 | Post-trained + 输出 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,o in r['models'].items():
        old=r['previous_posttrained_reference'][m]
        vals=[o['base']['primary']['accuracy'],o['arms']['input']['primary']['accuracy'],o['arms']['output']['primary']['accuracy'],old['base']['primary']['accuracy'],old['arms']['input']['primary']['accuracy'],old['arms']['output']['primary']['accuracy']]
        lines.append('| '+LABELS[m]+' | '+' | '.join(f'{v:.3f}%' for v in vals)+' |')
    lines += ['', '“从 Base 开始”与“冻结原始 Base 参数”不冲突：训练的适配器改变指定位置的有效计算。为了控制变量，本轮沿用固定 ChatML 任务序列及 Qwen3 空 thinking 前缀，让 Base 通过监督训练学习它。这个结果依赖该输入协议；它不是 Base 在最优 completion prompt 下的知识上限，也不能把大幅提升全部归为新增知识。',
        '', '## 冻结与复用审计','',
        f"全部 {audit['training_jobs']} 次训练（6烟测、18验证调参、30确认）通过原始参数逐位不变、每步冻结参数无梯度及优化器白名单检查。A-LoRA 实验没有隐藏 LoRA，内部 LoRA 对照没有边界适配器。所有 adapter 为 FP32 且从零残差开始，确认种子4002–4006。",
        '',f"重算 {audit['prediction_rows_recomputed']:,} 条候选预测并重新解码 {audit['generation_rows_redecoded']:,} 条生成。近并列重算翻转 {audit['near_tie_prediction_flips']} 条；候选与生成首 token 批次形状差异 {audit['generation_first_token_shape_disagreements']} 条。此前 {audit['original_sealed_files_rechecked']} 个封存文件重新验哈希，官方 Base 权重、配置、tokenizer 逐文件核验。",
        '', '模型来源见 MODEL_PROVENANCE.json，复用映射见 REUSE_AUDIT.json，选择见 SELECTION.json，完整结果见 RESULTS.json，核验见 FINAL_AUDIT.json。',
        '', '## 结论范围','',
        '- 只能将本轮解释为一个选择题任务上的 Base 监督适配验证。通用指令跟随、自由长文本、多任务迁移和原能力保持尚需独立证据。',
        '- 低参数数量不自动等于同预算竞争优势，也不意味着训练时间/显存按同比例减少；输入侧仍需穿过冻结 Transformer 反传。',
        '- 本轮没有同时叠加边界 A-LoRA 和内部 LoRA，不能直接回答 Base 起点上的叠加收益。',
        '- 输入含 bias、输出无 bias；输入与输出并非完全同参数预算。原始词表权重绑定不代表单侧适配后的有效计算仍然绑定。',
        '', '![Base 下游适配结果](figures/base_single_boundary.png)']
    (HERE/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.86,hspace=.4,wspace=.24)
    fig.suptitle('Task adaptation starting from pretrained Base checkpoints',fontsize=16,y=.98)
    fig.text(.5,.925,'SciQ | 998 canonical questions | 5 fresh seeds | all original weights frozen',ha='center',color='#555555')
    for row,(m,o) in enumerate(r['models'].items()):
        for col,(group,key,title) in enumerate([('primary','accuracy','Candidate accuracy'),('generation','strict_accuracy','Strict greedy answer accuracy')]):
            ax=axes[row,col];base=o['base'][group][key];vals=[base]+[o['arms'][a][group][key] for a in ARMS]
            bars=ax.bar(range(4),vals,width=.62,color=['#97a4ae','#297f9c','#885ba6','#bf7849'])
            for b,v in zip(bars,vals):
                inside=v>15;ax.text(b.get_x()+b.get_width()/2,v-8 if inside else v+2,f'{v:.2f}',ha='center',va='top' if inside else 'bottom',color='white' if inside else '#222',fontsize=10)
            metric='candidate_accuracy' if col==0 else 'strict_generation_accuracy'
            for i,a in enumerate(ARMS,1):
                c=next(c for c in r['primary_contrasts'] if c['model']==m and c['arm']==a and c['metric']==metric)
                ax.scatter(i+np.linspace(-.16,.16,5),np.array(c['effect_pp']['seeds'])+base,color='#222',s=13,zorder=3)
            ax.set(xticks=range(4),xticklabels=['Unadapted\nBase','Input\nA-LoRA','Output\nA-LoRA','Hidden LoRA\nr8'],ylim=(0,105),ylabel='Accuracy (%)',title=LABELS[m]+'\n'+title)
    fig.text(.08,.035,'Dots: individual seeds. Strict scores include format learning; low strict Base scores do not imply absent knowledge.\nHidden LoRA uses a much larger parameter budget. Fixed task serialization; no universal post-training or retention claim.',fontsize=9,color='#555555')
    (HERE/'figures').mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(HERE/f'figures/base_single_boundary.{suffix}',dpi=180)
    print(HERE/'FINAL_INTERPRETATION_ZH.md')
if __name__=='__main__':main()
