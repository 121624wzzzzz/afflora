import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import *

def main():
    r=read(HERE/'RESULTS.json');audit=read(HERE/'FINAL_AUDIT.json');assert audit['status']=='passed'
    labels={'qwen3_06b_chat':'Qwen3-0.6B','qwen25_15b_chat':'Qwen2.5-1.5B-Instruct'}
    lines=['# 只调整一处 A-LoRA 的下游验证','',
        '本轮分别训练输入端和输出端，每次只启用一处 rank16 A-LoRA，完全没有内部 LoRA。两模型各五个新种子，使用 SciQ 的已有干净划分；不是从未查看过的全新 benchmark。',
        '',f"冻结检查通过：{audit['training_jobs']} 次训练（含烟测、验证调参和确认）中，全部其他参数的训练前后字节摘要一致。每次反传检查基座梯度为 None，优化器只包含指定 A-LoRA 参数。独立重载和实际生成均经过检查。",'',
        '## 实际任务表现','',
        '选项准确率在四个候选分数中选最大值；严格生成要求 greedy 输出文本去除首尾空白后恰好为正确的大写答案字母。两者都使用原始选项顺序的998道干净测试题。适配器结果是五种子均值。',
        '', '| 模型 | 方法 | 可训练参数 | 选项准确率 % | 严格生成准确率 % | 输出格式有效率 % |', '|---|---|---:|---:|---:|---:|']
    for m,out in r['models'].items():
        b=out['base'];lines.append(f"| {labels[m]} | 未适配 | 0 | {b['primary']['accuracy']:.3f} | {b['generation']['strict_accuracy']:.3f} | {b['generation']['valid_answer_rate']:.3f} |")
        for arm,a in out['arms'].items():
            lines.append(f"| {labels[m]} | 只调{'输入' if arm=='input' else '输出'}端 | {a['parameters']:,} | {a['primary']['accuracy']:.3f} | {a['generation']['strict_accuracy']:.3f} | {a['generation']['valid_answer_rate']:.3f} |")
    lines += ['', '基座的严格生成分为0，需要按格式解释：原始输出通常是“字母＋选项文本”，并非没有答对任何问题。基座的选项准确率和未限制词表的首 token 正确率均明显高于0；严格生成的改善不能全部解释成知识或推理能力增长。预定的选项准确率主指标用于单独检查内容收益，两个主指标和选择规则保持不变。', '', '## 相对未适配模型的效应','',
        '单位为百分点，区间是五个训练种子的 Bonferroni 八对比校正95%区间。只有同一模型/单侧方法的选项准确率和严格生成准确率两条下界都大于0，才通过预先固定的正收益标准。未通过不等于严格无效，也不能解释为通过了非劣性检验。',
        '', '| 模型 | 单侧 | 指标 | 平均差 | 校正区间 | 正收益标准 |', '|---|---|---|---:|---|---|']
    for c in r['primary_contrasts']:
        e=c['effect_pp'];lo,hi=e['family8_ci95'];lines.append(f"| {labels[c['model']]} | {c['arm']} | {c['metric']} | {e['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'通过' if c['passes_positive_corrected_interval'] else '未通过'} |")
    lines += ['', '这些校正区间描述固定题集上的训练种子不确定性。对题目另做10,000次重采样、以五个拟合模型为条件的选项准确率差区间如下；它回答测试题抽样敏感性，是另一种不确定性，不能与种子区间混用：', '']
    for c in r['primary_contrasts']:
        if c['metric']=='candidate_accuracy':
            lo,hi=c['question_bootstrap95_conditioned_on_fitted_seeds'];lines.append(f"- {labels[c['model']]} / {c['arm']}：[{lo:+.3f}, {hi:+.3f}] pp。")
    lines += ['', 'Qwen3 两侧的题目区间也都在0以上。Qwen2.5 输入端虽然五个训练种子均有小幅提高、通过预定种子标准，题目区间仍包含0，因此尚不能宣称对新题分布已确认稳定收益。Qwen2.5 输出端的选项准确率两类区间均包含0。']
    lines += ['', '## 与常规内部 LoRA 的差距','',
        '以下内部 LoRA r8 数字来自经哈希核验的前一轮同任务、同候选评分协议五种子结果，是更大预算的参考，不能作为本轮新种子的配对检验或同预算比较。没有复用这些内部 adapter 来训练本轮 A-LoRA。','']
    for m,o in r['models'].items():
        h=r['larger_budget_hidden_r8_reference'][m]['accuracy'];hp=5046272 if m=='qwen3_06b_chat' else 9232384
        lines.append(f"- {labels[m]}：内部 LoRA r8 使用 {hp:,} 参数，选项准确率 {h:.3f}%；只调输入/输出端分别相差 {o['arms']['input']['primary']['accuracy']-h:+.3f} / {o['arms']['output']['primary']['accuracy']-h:+.3f} pp。")
    lines += ['', '低参数下能改善未适配模型，不等于达到常规内部 LoRA 的绝对表现。参数量减少也不代表输入侧训练时不需要穿过冻结 Transformer 反传，或训练速度/显存按同样比例节省。','',
        '## 解释边界','',
        '- “只调整这一处”指只有这处 adapter 参数被优化，其他原始参数未变。输入扰动仍会改变后续隐藏表示；冻结权重不保证所有模型行为保持不变。',
        '- 本轮直接检验科学选择题的准确率与严格答案生成。没有据此确认自由长文本、其他语言、其他任务或通用能力保持。',
        '- 输入含 bias、输出无 bias，参数数目略不同；两臂的直接对比不能被写成精确等预算优势。',
        '- 两模型原始 embedding/head 权重绑定，但本轮每个实验只改变其中一侧的有效计算；不宣称单侧合并后仍保留有效 tying。',
        '- 生成上限为16 token，保存了每条原始输出、EOS和长度上限信息；未从解释文本中抽取字母来提高评分。',
        '', '## 复现与核验','',
        'DESIGN.md 与 FROZEN_PROTOCOL.json 固定训练、评测和主检验；SELECTION.json 固定验证集选择，确认测试前不访问本轮测试分数。旧数据与模型文件重新核对哈希，全新训练 seeds3002–3006。',
        '',f"重算 {audit['prediction_rows_recomputed']:,} 条候选预测，重新解码 {audit['generation_rows_redecoded']:,} 条生成；近并列预测单条重算翻转 {audit['near_tie_prediction_flips']} 条，生成首步与候选评测批次形状的 argmax 差异 {audit['generation_first_token_shape_disagreements']} 条。",'',
        '详细指标、逐种子效应、题目 bootstrap 与轮换结果见 RESULTS.json；权重冻结和复用审计见 FINAL_AUDIT.json。旧实验封存文件再次全量核验。','',
        '![单侧任务结果](figures/single_boundary_task.png)']
    (HERE/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8));fig.subplots_adjust(left=.09,right=.97,bottom=.12,top=.87,hspace=.42,wspace=.27)
    fig.suptitle('Only one A-LoRA boundary is trained',fontsize=17,y=.98)
    fig.text(.5,.93,'SciQ | 998 canonical test questions | 5 fresh training seeds | all base weights unchanged',ha='center',color='#555555')
    for row,(m,out) in enumerate(r['models'].items()):
        for col,(group,key,title) in enumerate([('primary','accuracy','Candidate accuracy'),('generation','strict_accuracy','Strict greedy answer accuracy')]):
            ax=axes[row,col];base=out['base'][group][key];vals=[base,out['arms']['input'][group][key],out['arms']['output'][group][key]]
            bars=ax.bar(range(3),vals,color=['#96a3ad','#297f9c','#885ba6'],width=.58)
            for bar,value in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,value-8 if value>0 else 2,f'{value:.2f}',
                    ha='center',va='top' if value>0 else 'bottom',color='white' if value>0 else '#222222',fontsize=10)
            for idx,arm in enumerate(ARMS,1):
                metric='candidate_accuracy' if col==0 else 'strict_generation_accuracy'
                c=next(c for c in r['primary_contrasts'] if c['model']==m and c['arm']==arm and c['metric']==metric)
                ax.scatter(idx+np.linspace(-.15,.15,5),np.array(c['effect_pp']['seeds'])+base,color='#222222',s=14,zorder=3)
            if col==0:
                h=r['larger_budget_hidden_r8_reference'][m]['accuracy'];ax.axhline(h,ls='--',lw=1,color='#bc633e',label='Earlier hidden LoRA r8 (larger budget)')
                ax.legend(frameon=False,fontsize=8,loc='lower left')
            ax.set(xticks=range(3),xticklabels=['Frozen base','Input only','Output only'],ylim=(0,105),ylabel='Accuracy (%)',title=f'{labels[m]}\n{title}')
    fig.text(.09,.025,'Dots: individual training seeds. Strict scoring requires exactly one answer letter; base responses usually also include option text.\nThus a zero strict base score is not zero answer knowledge. No cross-task retention or equal-budget superiority is established.',fontsize=9,color='#555555')
    (HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/single_boundary_task.png',dpi=180);fig.savefig(HERE/'figures/single_boundary_task.pdf')
    print(HERE/'FINAL_INTERPRETATION_ZH.md')

if __name__=='__main__':main()
