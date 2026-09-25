"""Render the completed protocol diagnosis from audited results."""
from cross import HERE, read, write


def main():
    result = read(HERE / 'SUMMARY.json')
    assert result['status'] == 'completed_and_verified'
    thinking = result['qwen3_thinking']
    delta = result['descriptive_paired_contrasts']['qwen3_thinking_vs_chat_candidate']
    table = ['| 模型权重 | plain + Answer | plain 无 Answer | chat 无思考 | chat 无思考 + Answer |',
             '|---|---:|---:|---:|---:|']
    full_vocab = ['| 模型权重 | plain + Answer | plain 无 Answer | chat 无思考 | chat 无思考 + Answer |',
                  '|---|---:|---:|---:|---:|']
    generations = ['| 未适配模型及原输入 | 可抽取首字母准确率 | 恰好单个正确字母 | 原生成截断率 |',
                   '|---|---:|---:|---:|']
    for name, report in result['models'].items():
        conditions = report['conditions']
        table.append('| ' + name + ' | ' + ' | '.join(f"{conditions[c]['candidate_accuracy']:.2f}%" for c in ['plain_answer', 'plain_no_cue', 'chat', 'chat_answer']) + ' |')
        full_vocab.append('| ' + name + ' | ' + ' | '.join(f"{conditions[c]['unrestricted_accuracy']:.2f}%" for c in ['plain_answer', 'plain_no_cue', 'chat', 'chat_answer']) + ' |')
        gen = report['original_generation_reparsed']
        generations.append(f"| {name} | {gen['leading_letter_accuracy']:.2f}% | {gen['strict_accuracy']:.2f}% | {gen['length_cap_rate']:.2f}% |")
    notice = {
        'status': 'interpretation_narrowed',
        'source_artifacts_modified': False,
        'unsupported_interpretations': [
            'Qwen3 85.07 versus 62.53 proves a general knowledge regression caused by official post-training',
            'The 62.53 score resulted from our SFT reducing an 85.07 starting score',
            'Strict single-letter generation gains equal knowledge or general SFT gains',
            'A low fixed-position baseline establishes a lack of available answer-generation capability',
        ],
        'retained_scope': 'Previously measured adapter gains remain results under their explicitly fixed input/output and inference protocols; no new training was performed here.',
        'new_diagnosis': 'Strong prompt sensitivity and a different thinking-mode answer-generation operating point; not a pure causal estimate of reasoning or post-training.',
    }
    write(HERE / 'INTERPRETATION_NOTICE.json', notice)
    text = f'''# SciQ 的 Base / 后训练差距：协议交叉复测与 SFT 审计

当前发现的是实验定义和解释边界的问题，没有在本次复核中发现模型误加载、原始预测位置取错或 SFT 标签移位错误。Qwen3 的 62.53% 只描述固定无思考协议下的立即答案选择，不能代表这个后训练模型的总体答题能力。

尤其需要澄清：85.07% 和 62.53% 来自两个官方 checkpoint 各自未经我们 SFT 的结果，不是一次“我们的 SFT 把 85% 训到了 62%”的训练前后比较。

## 独立复核

- 重新验证原两轮封存研究共 {result['audit']['source_sealed_files_reverified']} 个文件，以及所用官方权重、配置和 tokenizer 哈希。
- 使用独立 AutoModelForCausalLM 完整 forward、左 padding 和显式 position IDs；未导入旧版建模/评分函数，未安装 adapter。
- 四模型原始条件的 4,000 条预测全部复现，预测翻转为零。最大标签 logit 差约 2.33e-4，处于预定数值容差内。
- 四模型各取固定 8 个训练例，用标准 Hugging Face shifted loss 核对原两位置目标。误差至多 2.38e-7；每题确实只监督答案字母和 EOS 两个 token。
- 完成并重算 16,000 条交叉提示预测，另生成并重新解码 1,000 条 thinking 输出。所有汇总仍只剔除之前固定的两道歧义题，n=998。

## 交叉提示结果

以下为四候选标签准确率；同一系列的两个 checkpoint 使用完全相同的序列化提示和 token IDs。四个提示在推理前固定并全部报告，未按测试分选取新主提示。

{chr(10).join(table)}

全词表最高分首 token 恰为正确字母的比例：

{chr(10).join(full_vocab)}

这个矩阵显示两种方向的 Base / 后训练高低都可以通过改变输入协议出现。特别是 Qwen3 Base 去掉 Answer 提示后，四候选分数从 85.07% 变为 26.05%，实际首 token 输出字母的比例也大幅下降；不能解释为删除几个提示字符就删除了科学知识。直接在不适合答案续写的位置比较四个低概率字母，测量含义会改变。

同一后训练 Qwen3 加入 Answer 前缀，四候选分数从 62.53% 提升到 69.74%。这表明原指标有明显提示敏感性；它本身尚不能解释全部差距。

chat 对 Base 是跨协议诊断，不是推荐的 Base 使用方式；plain 对后训练模型同样不代表其最佳或原生使用方式。相同文本控制和原生接口评测回答不同问题，不能混合为单一“模型谁更强”的排名。

## Qwen3 原生 thinking 生成

随后单独固定 THINKING_DESIGN.md，使用相同题目、同一个官方后训练 checkpoint，不做任何训练。启用原生 thinking，按官方建议使用温度 0.6、top_p 0.95、top_k 20 采样，每题一个输出，最多 2,048 新 token。只在生成的最后一个 </think> 之后提取答案，不把思考首 token 当作答案。

| 同一 Qwen3 后训练权重的使用方式 | 答案准确率 |
|---|---:|
| 原 chat 无思考，greedy 首答案字母 | 62.53% |
| chat 无思考，加 Answer 前缀，首答案字母 | 69.74% |
| thinking + 官方建议采样，完成思考后抽取最终答案 | {thinking['content_accuracy']:.2f}% |

thinking 条件答对 {thinking['correct_count']} / {thinking['n']}，可抽取最终答案率 {thinking['valid_final_answer_rate']:.2f}%，结束符终止率 {thinking['terminated_rate']:.2f}%，长度上限率 {thinking['length_cap_rate']:.2f}%，生成长度中位数 {thinking['median_generated_tokens']} token。严格“最终文本恰好一个正确字母”的分数另为 {thinking['strict_accuracy']:.2f}%，不作为内容能力的替代。

相对原无思考预测，这个操作条件的差为 {delta['left_minus_right_pp']:+.2f} 个百分点；修正 {delta['corrected']} 道、退步 {delta['regressed']} 道。名义题目 bootstrap 区间为 [{delta['nominal_question_bootstrap95_pp'][0]:.2f}, {delta['nominal_question_bootstrap95_pp'][1]:.2f}] pp，仅以本次预测为条件；没有多重比较校正或采样种子不确定性估计。

这里同时改变了思考模式、解码策略和推理预算，不能把差值全部归因于某一个因素。它足以说明 62.53% 不是该 checkpoint 唯一可达到的答题水平，但不是与 Base 同推理预算的优劣检验，更不是 A-LoRA 的效果。2,048 token 预算低于官方一般建议，截断样本均保留并单列，未挑选重跑。

## 内容和格式必须分开

对原先已保存的生成文本，只允许在开头提取带边界的 A/B/C/D，接着出现选项词或解释不再判成内容错误。没有扩展规则去适配个别测试答案，也不评价解释内容本身。

{chr(10).join(generations)}

例如 D. clone 可以选对答案，同时未满足“只输出 D”的格式要求。这两个指标都可有用，但不能把后者从零变高全部解释为知识或新能力提升。原生成仅允许 16 个 token，表中内容评分只是答案字母抽取，不是对完整自由回答的全面评价。

## 这轮 SFT 的适当定位

当前数据每题监督两个 token：答案字母与结束符。它是合理的受控选择题适配实验，但没有监督解释、推理轨迹、一般指令跟随或跨任务泛化。它可以用于研究固定接口下单层适配是否影响选项准确率，不能独立承担“完整后训练有效”的结论。

之前输入 A-LoRA 在 Base 上 +1.58 / +1.90 pp 的固定协议内容收益仍是已经测得的结果；这些数字没有被此次诊断推翻。Qwen3 后训练起点上更大的收益也应限定为固定无思考协议下的适配收益，不能直接说成从没有到有学会了科学知识。输出侧、叠加收益、同预算优势和通用能力仍需各自证据。

后续应先固定任务定义与推理预算：内容评分容许明确的答案表达，格式评分独立列出，缺失答案和截断单独报告。Qwen3 的思考/无思考应成为明确实验条件。原生接口结果与相同输入控制分别报告，训练前后保持同一协议。再加入与论文主张匹配的生成式任务和同预算对照，避免通过选择较差提示制造提升空间。

官方使用依据：[Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/README.md)。原研究文件没有修改；本记录收窄解释范围，不覆盖已有原始结果。
'''
    (HERE / 'INTERPRETATION_ZH.md').write_text(text)
    print('Report and interpretation notice written.')


if __name__ == '__main__':
    main()
