"""Render complete numeric tables without modifying pre-specified analysis."""
import numpy as np
from common import *

ARMS=['base','hidden','hidden_budget','hidden_both']
SIZE={'qwen25_15b_base':'1.5B（历史）','qwen25_3b_base':'3B','qwen25_7b_base':'7B'}
TASK={'cluener':'CLUENER / span F1','wikisql':'WikiSQL / 执行正确率%'}

def main():
    a=read(HERE/'TEST_ANALYSIS.json');audit=read(HERE/'RESULT_AUDIT.json')
    assert read(HERE/'EXPERIMENT_COMPLETE.json')['status']=='passed' and audit['status']=='passed'
    positives=[c for c in a['conditions'] if all(x['bonferroni8_ci'][0]>0 for x in c['comparisons'])]
    negative=[(c,x) for c in a['conditions'] for x in c['comparisons'] if x['mean']<0]
    lines=['# Qwen2.5 Base：3B / 7B 模型尺寸扩展','',
        f"已完成两个新尺寸、两项任务、三种适配方案各五种子，共60次正式训练、4组Base和4个短程检查。四个新任务/尺寸条件中，{len(positives)}组同时通过相对普通与预算LoRA的本轮预定校正区间标准；八项平均增量中，{len(negative)}项为负。下表保留全部结果。",'',
        '本轮是在此前1.5B表现较强的两项任务上作有针对性的尺寸扩展。历史1.5B锚点经过复核，但不混入新八项比较的统计族。不能据此宣布任意任务、任意尺寸都有效，也不能把实际型号差异单独归因为参数数量。','',
        '**结果解释：大模型上仍观察到平均叠加收益，幅度较1.5B减小；本轮没有新增同时通过两个对照校正区间的条件。** CLUENER的3B/7B与WikiSQL的3B，五种子均超过两个对照。7B WikiSQL仅前四种子同时超过对照，第五种子相对普通LoRA少对1题、与预算组持平。不能把均值为正写成所有种子或所有比较均已确认。','',
        '八个主要区间中，仅3B WikiSQL相对严格等参数LoRA的区间高于零：+1.1523 pp [0.2419, 2.0628]；它相对普通LoRA的区间仍跨零。其余七项也跨零。五种子的正向一致性是有用的描述性证据，但不替代预先固定的区间标准；跨零同样不证明真实增益为零。','',
        '在1.5B→3B→7B这三个型号上，叠加相对预算LoRA的均值差为：CLUENER +3.325→+1.675→+1.247 F1，WikiSQL +2.207→+1.152→+0.684 pp。三方案本身分数都随型号尺寸上升，但叠加的额外幅度缩小。这是当前任务与共同训练方案下的描述性趋势；并未检验尺寸间差值，也没有区分容量、权重绑定、网络形状与超参数匹配的作用。','',
        '开发集是辅助证据：7B WikiSQL叠加84.531%，普通LoRA84.766%，预算LoRA84.375%，相对普通均值为−0.234 pp。其余三个条件的开发集均值均超过两个对照，但所有开发集校正区间跨零。开发集没有用于挑选最终checkpoint、调整学习率或改变测试统计。','',
        '## 完整保留集结果','',
        '|任务 / 指标|Qwen2.5 Base|Base|普通LoRA|预算LoRA|LoRA+A-LoRA|','|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        cs=[c for c in a['historical_15b_anchors'] if c['task']==task]+[c for c in a['conditions'] if c['task']==task]
        for c in cs:
            lines.append('|'+TASK[task]+'|'+SIZE[c['model']]+'|'+'|'.join(f"{c['means'][arm]:.3f}" for arm in ARMS)+'|')
    lines += ['','适配器分数是五种子均值；Base是同一提示下的确定性零样本评测。CLUENER固定1343条，WikiSQL固定1024条。所有新模型使用同一批2048训练样本、相同任务提示、一轮64步、共同LR2e-4；各任务的microbatch与历史版本保持一致。零样本Base分数也受接口与格式遵循影响，不代表模型知识量。','',
        '**预算必须区分：3B为严格等参数；7B预算LoRA比叠加组多512个可训练参数。** 不能将7B写成严格等参。本轮不改变边界bias定义来凑预算。','',
        '|尺寸|普通LoRA可训练参数|预算LoRA可训练参数|叠加组可训练参数|边界模块新增参数|',
        '|---|---:|---:|---:|---:|',
        '|3B|14,966,784|15,099,904|15,099,904|133,120|',
        '|7B|20,185,088|20,418,560|20,418,048|232,960|','',
        '边界模块分别增加普通LoRA参数量的约0.89% / 1.15%。该参数表只说明容量预算；边界分支与内部LoRA的缩放、dropout和放置位置也不同，本轮比较的是完整固定方案，不能单独识别放置位置的因果效应。','',
        '## 预定主要比较','',
        '|任务 / 尺寸|对照|平均增量|配对差值SD|Bonferroni-8双侧95%种子t区间|五种子均正向|','|---|---|---:|---:|---|---|']
    for c in a['conditions']:
        for x in c['comparisons']:
            lo,hi=x['bonferroni8_ci'];control='普通LoRA' if x['control']=='hidden' else ('严格等参LoRA' if c['model']=='qwen25_3b_base' else '略高预算LoRA（+512）')
            lines.append(f"|{c['task']} / {SIZE[c['model']]}|{control}|{x['mean']:+.4f}|{x['sd']:.4f}|[{lo:+.4f}, {hi:+.4f}]|{'是' if x['all_seeds_positive'] else '否'}|")
    lines+=['','区间使用五个配对训练种子、df=4；同时校正本轮八个预定主要差值。只对固定测试题和当前设置有效。一个条件需两个区间下界都高于零才通过预定标准；跨零不代表已经证明无效。历史锚点的区间来自各自原实验，不重新并入本轮统计族。','',
            '![模型尺寸结果](figures/model_size_scores.png)','',
            '图中误差条为训练种子标准差，不能用误差条是否重叠代替配对显著性检验。','',
            '![新尺寸配对增量](figures/new_model_effects.png)','',
            '## 每个种子的分数','',
            '|任务 / 尺寸|种子|普通LoRA|预算LoRA|叠加|叠加−普通|叠加−预算|','|---|---|---:|---:|---:|---:|---:|']
    for c in a['conditions']:
        for i,seed in enumerate(c['seeds']):
            h,b,s=[c['values'][arm][i] for arm in ARMS[1:]]
            lines.append(f"|{c['task']} / {SIZE[c['model']]}|{seed}|{h:.3f}|{b:.3f}|{s:.3f}|{s-h:+.3f}|{s-b:+.3f}|")
    lines+=['','## 内容与格式诊断','',
            '以下是描述性辅助指标，未另作确认性检验。所有主评分、解析和推理预算在运行前冻结，未按新得分调整。','',
            '|CLUENER / 尺寸 / 方案|跨度F1|文本F1|整句完全正确%|schema有效%|严格JSON%|','|---|---:|---:|---:|---:|---:|']
    for c in a['conditions']:
        if c['task']!='cluener':continue
        for arm in ARMS:
            s=c['secondary_means'][arm]
            lines.append('|'+SIZE[c['model']]+' / '+arm+'|'+'|'.join(f'{s[k]:.3f}' for k in ['span_micro_f1','text_micro_f1','content_correct','schema_valid','strict_json'])+'|')
    lines+=['','|WikiSQL / 尺寸 / 方案|官方执行%|逻辑形式%|查询有效%|严格JSON%|独立占位符执行%|','|---|---:|---:|---:|---:|---:|']
    for c in a['conditions']:
        if c['task']!='wikisql':continue
        for arm in ARMS:
            s=c['secondary_means'][arm]
            lines.append('|'+SIZE[c['model']]+' / '+arm+'|'+'|'.join(f'{s[k]:.3f}' for k in ['primary','lf_correct_pct','query_valid_pct','strict_json_pct','distinct_parameter_execution_correct_pct'])+'|')
    d=read(HERE/'CONTENT_DIAGNOSTICS.json')
    lines+=['','|任务 / 尺寸 / 方案|平均生成token|原生EOS%|达到生成上限%|','|---|---:|---:|---:|']
    for c in d['conditions']:
        v={k:np.mean([s[k] for s in c['seeds']]) for k in ['mean_tokens','native_eos_pct','capped_pct']}
        lines.append(f"|{c['task']} / {SIZE[c['model']]} / {c['arm']}|{v['mean_tokens']:.2f}|{v['native_eos_pct']:.3f}|{v['capped_pct']:.3f}|")
    lines+=['','双方输出均有效的子集和逐实体类型的计数见 [CONTENT_DIAGNOSTICS.json](CONTENT_DIAGNOSTICS.json)。NER子集F1不可加回总体，也不能解释为因果中介。固定规则选择的胜例和败例见 [QUALITATIVE_PANEL.md](QUALITATIVE_PANEL.md) 与 [逐例复核说明](QUALITATIVE_REVIEW_NOTES.md)。','',
        'CLUENER相对预算组，3B五种子平均precision从63.571升至64.710、recall从65.475升至67.721；7B分别从68.183升至69.645、70.293升至71.315。两模型的五个种子在双方schema都有效的子集上均保留正F1差；严格JSON率反而略降。因此本次观察到的差异不只对应JSON语法遵循。','',
        'WikiSQL相对预算组，3B双方查询都有效的分层贡献平均+1.1523 pp，其余分层净贡献为零；7B对应+0.7031 pp与−0.0195 pp，合为+0.6836 pp。逻辑形式匹配均值也提高。以上分层由模型输出决定，是描述性核对，不是内容与格式的因果分解。7B普通LoRA的独立占位符诊断比官方执行均值高0.0195 pp，说明两种执行约定并非全量逐例等同；仍保留冻结的官方指标为主要结果。','',
        '## 正确性与复用核验','',
        f"全量核验：{audit['responses']:,}条新输出（含96条短程检查）、{audit['evaluations']}次评测、{audit['tokenizations']:,}条新模型分词记录、68份参数范围、20组新模型配对初始化/样本顺序，以及10组历史配对顺序。全部有效SQL预测中{audit['official_prediction_executions']:,}条再与官方执行器核对。",'',
        '60次正式训练均完成2048条、64步；原始权重冻结、优化器范围与FP32状态、checkpoint哈希和保存恢复逐次核验。历史1.5B的37,872条输出先重新解码评分，再作曲线锚点；新3B/7B不复用已训练adapter。官方24个模型/配置/分词器等文件、冻结数据与代码在封存前再哈希。','',
        '所有运行无按结果筛选、提前停止或失败重跑。四个短程检查只用于数值和资源验证，没有质量门槛。共享GPU被其他作业占用后，发生一次仅涉及资源准入的调度接续：原2–6号卡启动前至少48 GiB空闲，已完成检查点与输出先核验再复用，冻结的拟合、评分和统计源码不变。原调度器在无活动worker时主动结束，不计作拟合失败。详见 [RESOURCE_AMENDMENT.md](RESOURCE_AMENDMENT.md)；wall-clock不用于速度优越性比较。最终状态与完整文件清单见 [FINAL_AUDIT.json](FINAL_AUDIT.json)、[COMPLETION.json](COMPLETION.json)、[ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。','',
        '## 限制','',
        '任务因既有正结果而选择；本轮不是全新任务集合。3B tied、7B untied，模型深度和宽度也同时改变，不能声称识别了纯参数规模效应。共同学习率、少量训练数据、最终单检查点支持预算一致的比较，未替代各方法独立且等额的调参。公开数据可能被预训练接触，未被排除。','',
        '原有ANLI未见稳定收益、Banking77/E2E小幅但校正区间跨零的结果保持不变。本轮只新增尺寸证据，不产生standalone优越性、通用后训练提升、普遍任务优势或单独机制归因。更详细设置见 [PROTOCOL.md](PROTOCOL.md)、[METHOD_AND_LIMITS_ZH.md](METHOD_AND_LIMITS_ZH.md)；开发集结果见 [DEV_RESULTS.md](DEV_RESULTS.md)。','']
    (HERE/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines))
    (HERE/'README.md').write_text('# Qwen2.5 model-size extension: 3B / 7B\n\nCompleted: 60 fitted runs, 4 Base evaluations, 4 smoke runs on CLUENER and WikiSQL. Five paired seeds per task; fixed 2048-example training sets and common hyperparameters. Verified 1.5B results are historical anchors.\n\n3B: exact parameter match. 7B: budget LoRA has 512 MORE trainable parameters. Tasks selected after earlier positive results; size trends also mix tied/untied architectures.\n\nRead [full Chinese report](FINAL_INTERPRETATION_ZH.md), [frozen protocol](PROTOCOL.md), [methods and limits](METHOD_AND_LIMITS_ZH.md), [primary numbers](TEST_RESULTS.md), [diagnostics](CONTENT_DIAGNOSTICS.json), and [paired-effect plot](figures/new_model_effects.png).\n\nAudit: 90432 new responses, 37872 historical anchor responses, 13838 new token records, 68 scopes, 20 new paired initialization/order groups. No fitted adapter reused for new model runs.\n\nAfter sealing this directory is immutable. verify_seal.py is read-only; do not rerun writer scripts in a sealed archive.\n')

if __name__=='__main__':main()
