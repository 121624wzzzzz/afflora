"""Chinese interpretation; presentation only, does not change the analysis."""
from common import *

def main():
    r=read(HERE/'RESULTS.json');audit=read(HERE/'FINAL_AUDIT.json');numeric=read(HERE/'NUMERIC_TIES_SUMMARY.json')
    assert audit['status']=='passed'
    labels={'qwen3_06b_chat':'Qwen3-0.6B','qwen25_15b_chat':'Qwen2.5-1.5B-Instruct'}
    arms={'none':'普通内部 LoRA','both':'LoRA + 双边 A-LoRA','interior':'LoRA + 相同内部仿射模块','hidden_budget':'等预算内部 LoRA'}
    lines=['# SciQ 架构对照：完整结果与下一步判断','',
        '本轮没有建立“双边 A-LoRA 叠加后稳定提高任务正确率”的证据，也没有建立边界位置优于所测内部位置的证据。六个预设主要比较的校正置信区间全部跨零；即使使用未经多重校正的普通95%区间，六个比较仍全部跨零。这不是等效性证明，也不是对所有 A-LoRA 变体的否定。','',
        '共完成16次对称学习率搜索、40次新种子确认训练，另有8次两步短跑验证。全部使用新初始化的适配器。两个模型、四个方案分别在验证集选择学习率，最终均选中2e-4；测试使用2002–2006五个新种子，调参种子2001不参与推断。','',
        'SciQ 的独立训练/验证/测试划分来自 [官方数据源](https://huggingface.co/datasets/allenai/sciq)，固定 revision 为 `2c94ad3e1aafab77146f384e23536f97a4849815`。仅提供题目及选项，不提供包含答案的 support 段落。训练前剔除19个与验证/测试问题重叠的训练样本，以及14个正确答案同时出现在干扰项中的训练样本，剩11,646条。主评测使用998道没有该类答案歧义的测试题；完整1,000题评分全部保留。详见 DATA_AUDIT.json 和 PROTOCOL_CLARIFICATIONS.md。','',
        '测试文本和标签在数据有效性检查中被读取；模型测试分数直到配置选择冻结后才开始计算。这里的 held-out 指未用于任务训练和模型选择，不表示原始测试数据从未被研究者查看，也不保证基座预训练没有接触公开基准。','',
        '| 模型 | 方案 | 固定顺序正确率 % | 四种顺序平均正确率 % | 候选答案 NLL ↓ |',
        '|---|---|---:|---:|---:|']
    for m,aa in r['aggregates'].items():
        for arm,x in aa.items():lines.append(f"| {labels[m]} | {arms[arm]} | {x['accuracy']:.4f} | {x['rotation_accuracy']:.4f} | {x['candidate_nll']:.6f} |")
    lines+=['','下表均为“双边 A-LoRA 减去对照”的正确率差，单位为百分点。区间按六个主要比较做 Bonferroni 校正，统计单位为五个配对训练种子。','',
        '| 模型 | 对照 | 平均差值 | 校正95%区间 |','|---|---|---:|---|']
    for c in r['comparisons']:
        x=c['primary_adjusted'];arm=c['contrast'].removeprefix('both-minus-')
        lines.append(f"| {labels[c['model']]} | {arms[arm]} | {x['mean']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] |")
    lines+=['',
        '所有主要效应的绝对均值都低于预先设定的0.5个百分点实用参考值。按题目配对重采样的支持性区间也全部跨零。种子区间固定这份测试集；题目重采样固定这五组已训练模型；二者都不是对所有任务的联合泛化保证。','',
        '对等参数内部 LoRA，Qwen3 的五个种子中双边 A-LoRA 仅1胜4负；Qwen2.5 为3胜1平1负。固定顺序的均值差分别为−0.0401和+0.0802个百分点。四种循环选项顺序平均后，两个模型的差值均为−0.0551个百分点；对应支持性区间仍跨零。因此，固定顺序里偶尔出现的正向差值没有表现为稳定的顺序稳健收益。','',
        '完整1,000题的名义标签评分给出同样判断：相比等预算内部 LoRA，均值差为Qwen3 −0.06、Qwen2.5 +0.08个百分点。结论不依赖于剔除那两道歧义题。候选答案 NLL 相比普通和等预算 LoRA 的均值均略差，区间也跨零，不能改用这个次要指标宣称成功。','',
        '新增预算分别为66,560和99,840参数，约为原内部 LoRA 参数的1.319%和1.081%。边界组、内部位置组及扩秩组总参数量精确一致。内部位置组使用同样的两个线性仿射模块、初始化、rank、scale和bias，位置固定在第6和第20号解码块之后；它是位置控制，不是带非线性的 Houlsby adapter 复现。扩秩对照使用明确的q/k分配规则，并非穷举所有最优内部分配。','',
        '原版 A-LoRA wrapper 与本轮执行路径的非零适配器输出，在 BF16 AMP 和 FP32 下均逐位一致。八组短跑的独立重载输出完全一致；主训练和推理流程也通过初始参数、样本顺序、预算、有限值、masked loss 等检查。',
        f"最终审计覆盖{audit['job_count']}组运行、{audit['prediction_rows_checked']:,}条预测和{audit['paired_groups_checked']}个配对初始化组，基座权重再次完整重算哈希。",'',
        f"额外数值诊断筛查160,000条测试查询，其中{numeric['flagged_rows']}条的前两名选项 logit 间隔不超过0.001。全部重新加载后逐条用FP32计算，选项翻转{numeric['prediction_flips']}次，最大形状相关 logit 差{numeric['maximum_observed_shape_logit_difference']:.8g}。这是结果出现后补充的诊断，不改变主要评分，也不是所有张量形状误差的数学上界。",'',
        '这轮真正补强的是结论的可检验性：任务正确率、额外容量与模块位置分开比较，并验证输出格式和选项顺序的影响。它没有补出期待的稳定正收益。以前特定SFT设置中的CE改善及CMRC平均参考答案概率改善仍是各自协议下的证据，但不能替代这里的准确率结果。','',
        '我不建议继续靠换基准或追加种子寻找正结果来强化“普遍叠加收益”。更值得验证的是：论文观察到的词表矩阵结构，是否对应模型实际使用的输出方向；以及输入侧、输出侧和双侧在什么条件下有不同作用。','',
        '下一步优先的机制实验（本轮未启动）：选用真实全参微调模型作为教师，冻结其内部表示；让不同参数化从同一基座输出头恢复教师的输出行为，在相同原始参数预算、对称验证集调参下比较 A-LoRA 与直接 Vocab-LoRA。训练/评测同时考察真实激活上的教师KL、金标准NLL与任务正确率，并把未加权矩阵重建误差仅作为描述量。这样可以检验“矩阵更容易被仿射近似”是否真能转化成有用的函数近似。教师来自真实微调，不能直接用 A-LoRA 自己生成的合成教师作为唯一证据。该实验只能解释输出参数化，不能单独证明联合训练时的叠加收益。','',
        '应用实验则应优先贴合论文的输出侧主张：在预先固定的任务和内部rank/训练数据量矩阵中，对输入侧、输出侧、双侧及等预算内部 LoRA 做配对消融，再用新的封存评测做确认。这一轮没有训练输出侧单独的 SciQ 方案，不能据此否定它；已有旧任务消融也不能包装成新的未见测试证据。','',
        '本轮方案、训练配置和推断规则见 DESIGN.md / FROZEN_PROTOCOL.json / SELECTION.json；全量统计见 RESULTS.json；逐题预测在 checkpoints/；图在 figures/sciq_placement.png 和 .pdf；审计见 FINAL_AUDIT.json。所有更早的已封存实验保持不变。']
    (HERE/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
