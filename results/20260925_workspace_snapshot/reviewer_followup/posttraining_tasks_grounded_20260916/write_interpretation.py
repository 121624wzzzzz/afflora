"""Render the reviewed interpretation from completed, audited analysis files."""
from common import *

TASK_LABELS = {'toolace': '工具调用', 'cluener': '实体抽取'}
MODEL_LABELS = {'qwen3_06b_base': 'Qwen3-0.6B Base', 'qwen25_15b_base': 'Qwen2.5-1.5B Base'}


def main():
    assert read(HERE / 'ANALYSIS_COMPLETE.json')['status'] == 'passed'
    assert read(HERE / 'HELDOUT_AUDIT.json')['status'] == 'passed'
    analysis = read(HERE / 'HELDOUT_ANALYSIS.json')
    tables = {(r['task'], r['model'], r['arm']): r for r in analysis['tables']}
    contrasts = analysis['primary_contrasts']
    aux = read(HERE / 'HELDOUT_CONTENT_DIAGNOSTIC.json')['table']
    aux_map = {(r['task'], r['model'], r['arm'], r['eval_split']): r for r in aux}
    decisive = [r for r in contrasts if r['task'] == 'cluener' and r['model'] == 'qwen25_15b_base']
    assert len(decisive) == 2 and all(r['lower'] > 0 for r in decisive)
    assert all(r['lower'] < 0 < r['upper'] for r in contrasts if r not in decisive)
    assert all(r['mean'] > 0 for r in contrasts)
    lines = [
        '# 新任务实验：最终解释与论文证据边界', '',
        '这轮找到了可支持的叠加收益：在 Qwen2.5-1.5B Base 的 CLUENER 实体抽取上，'
        '普通 LoRA 加输入/输出 A-LoRA，五种子保留集均值提高约 3.39 F1；'
        '相对严格等参数的内部 LoRA 高约 3.33 F1。两个主区间经八项比较校正后均高于零。'
        '它支持这个固定训练设置下的有效增量，不能外推为所有任务或最优调参后的普遍优势。', '',
        '其余三组任务/模型组合的叠加均值也为正，但五种子校正区间跨零。'
        '因此应表述为仍有不确定性，不能把它们写成稳定优势，也不能把跨零解释为已经证明无效。', '',
        '## 保留集主结果', '',
        '以下均为五个配对种子的均值。工具调用为 710 条保留样本上的完整调用集合正确率；'
        '实体抽取为 1,343 条 CLUENER 公开开发样本上的类别与跨度 micro F1，均按 0–100 计分。'
        '所有适配器只用 2,048 条训练数据、同一 LR 2e-4、单轮 64 步；评测前没有按保留集调参或选检查点。', '',
        '| 任务 | 模型 | 普通 LoRA | + A-LoRA | 等参数 LoRA | 相对普通 LoRA | 相对等参数 LoRA |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for task in TASKS:
        for model in MODELS:
            h, b, c = [tables[(task, model, arm)]['primary'] for arm in ['hidden', 'hidden_both', 'hidden_budget']]
            lines.append(f'| {TASK_LABELS[task]} | {MODEL_LABELS[model]} | {h:.2f} | {b:.2f} | {c:.2f} | {b-h:+.2f} | {b-c:+.2f} |')
    lines += ['', '| 任务 / 模型 | 对照 | 增量 | 五种子同时 95% t 区间（8 比较） | 条件题目/簇 95% 区间 |',
              '|---|---|---:|---|---|']
    for r in contrasts:
        label = '普通 LoRA' if r['control'] == 'hidden' else '等参数 LoRA'
        ci = r['paired_cluster_bootstrap_95']
        lines.append(f"| {TASK_LABELS[r['task']]} / {MODEL_LABELS[r['model']]} | {label} | {r['mean']:+.3f} | [{r['lower']:+.3f}, {r['upper']:+.3f}] | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines += ['', '种子 t 区间依赖其参数假设；条件题目/簇区间以五个已拟合模型为条件，不能替代训练种子变异。'
              '实体抽取两个模型的五个种子差值均为正，0.6B 的种子区间仍跨零；1.5B 的证据更充分。'
              '工具调用的保留集均值比开发集更正向，但种子方向不全一致。全部逐种子差值见 '
              '[HELDOUT_ANALYSIS.json](HELDOUT_ANALYSIS.json)。', '',
              '![保留集叠加效应与种子区间](figures/heldout_additive_effects.png)', '',
              '## 单独训练边界层', '',
              '下表的输入侧与输出侧均没有内部 LoRA 可训练参数，原模型权重冻结。'
              '输入侧参数量分别为 33,792 / 50,688，输出侧为 32,768 / 49,152。', '',
              '| 任务 | 模型 | 未适配 Base | 仅输入侧 | 仅输出侧 |', '|---|---|---:|---:|---:|']
    for task in TASKS:
        for model in MODELS:
            values = [tables[(task, model, arm)]['primary'] for arm in ['base', 'input', 'output']]
            lines.append(f'| {TASK_LABELS[task]} | {MODEL_LABELS[model]} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |')
    lines += ['', '输入侧在本轮四组任务上均能改善内容分数；输出侧的效果取决于任务，'
              '0.6B 工具调用低于未适配 Base。本轮未设置与单侧边界层同预算的内部 LoRA 对照，'
              '因此这部分支持低参数适配能力，不支持新的单侧同预算优越性主张。', '',
              '## 低 Base 分数和格式因素', '',
              '实体抽取零样本 Base 的低分包含提示与输出协议因素。两条固定训练示例能提高未适配 Base：', '',
              '| 模型 | 零样本 Base F1 | 两示例 Base F1 | 两示例 JSON 合法率 | 两示例截断率 |',
              '|---|---:|---:|---:|---:|']
    for model in MODELS:
        r = aux_map[('cluener', model, 'base', 'fewshot_test')]
        base = tables[('cluener', model, 'base')]['primary']
        lines.append(f"| {MODEL_LABELS[model]} | {base:.2f} | {r['primary']:.2f} | {r['json_valid']:.2f} | {r['capped_pct']:.2f} |")
    lines += ['', '两示例提示与正式零样本对照不同，只作提示敏感性诊断；不能据低 Base 分数声称模型完全没有实体知识。'
              '叠加效应应主要看已经完成任务适配的普通 LoRA 与等参数 LoRA。以下补充实体精确率、召回率和格式：', '',
              '| 模型 | 配置 | 实体精确率 | 实体召回率 | JSON 合法率 | 截断率 |', '|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        for arm, label in [('hidden', '普通 LoRA'), ('hidden_both', '+ A-LoRA'), ('hidden_budget', '等参数 LoRA')]:
            r = aux_map[('cluener', model, arm, 'test')]
            lines.append(f"| {MODEL_LABELS[model]} | {label} | {r['span_precision']:.2f} | {r['span_recall']:.2f} | {r['json_valid']:.2f} | {r['capped_pct']:.2f} |")
    lines += ['', '1.5B 的实体精确率从 58.04 提高到 61.43，召回率从 60.36 提高到 63.78；'
              '普通 LoRA 与叠加组的 JSON 合法率都已在 99.6% 左右。', '',
              '这些补充指标是事后描述，不替代预定主指标，也不单独识别因果机制。'
              '开发集普通 LoRA 的总 CE 降幅中，EOS 项约占 0.03%–0.34%；但答案 CE 仍包括 JSON 结构和复制 token，'
              '因此仍以实际生成的完整调用正确率与实体跨度 F1 判断效果。', '',
              '## 协议修订、审计与适用范围', '',
              'V1 的完整记录保留。发现无依据的工具日期标注及实体下标生成干扰后，在看到任何新任务 A-LoRA 分数前固定 V2。'
              '工具调用仅涵盖参数可在请求中找到字面依据的正调用，不评估拒绝调用、实际执行或完整 BFCL。'
              '实体生成改为类别/文本/出现序号，用不访问金标准的程序还原原始跨度。详见 '
              '[方法说明](METHOD_AND_LIMITS_ZH.md) 与 [V1 记录](../posttraining_tasks_20260916/README.md)。', '',
              '本轮完成 104 次训练（含 4 次冒烟），核验 110 个参数范围记录、46,168 条 token 化输入，'
              '开发集 22,000 条和保留集 109,442 条输出均重新解码、评分与核对输入/复用身份。'
              '共 131,442 条 V2 输出；V1 的 2,400 条另行保留。共同隐藏层初始化、样本顺序、'
              '原权重冻结、优化器白名单、保存重载及严格等参数关系均通过。', '',
              '叠加组与等参数对照总参数分别为 5,112,832 / 9,332,224。推理统一为 FP32、贪心、原生 EOS、'
              '512 token；改用 batch32 前的 256 条开发输出与 batch8 token 序列及停止标记全部相同。'
              '所有保留集标准答案都在长度上限内。共享 GPU 影响了运行时间，原始耗时不作方法吞吐率比较。', '',
              '论文可以据此给出一个具体、受控的有效案例：在 Qwen2.5-1.5B Base 的小数据实体抽取适配中，'
              '边界 A-LoRA 对内部 LoRA 提供了超过所设等参数扩容对照的额外收益。'
              '输入侧单独适配也有明确的任务分数改善。当前尚无独立学习率搜索、更多模型或跨语言机制消融；'
              '公共数据预训练暴露也不能排除。两种语言与数据来源同时变化，不能把任务差异归因于某个单一机制。', '',
              '机器可读结果与核验：[HELDOUT_ANALYSIS.json](HELDOUT_ANALYSIS.json)、'
              '[HELDOUT_AUDIT.json](HELDOUT_AUDIT.json)、[HELDOUT_CONTENT_DIAGNOSTIC.json](HELDOUT_CONTENT_DIAGNOSTIC.json)、'
              '[PARAMETER_AUDIT.json](PARAMETER_AUDIT.json)、[RESULT_AUDIT.json](RESULT_AUDIT.json)。'
              '最终完成状态及文件清单见 [COMPLETION.json](COMPLETION.json) 和 [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。']
    (HERE / 'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines) + '\n')
    print('Rendered FINAL_INTERPRETATION_ZH.md from completed audited results')


if __name__ == '__main__':
    main()
