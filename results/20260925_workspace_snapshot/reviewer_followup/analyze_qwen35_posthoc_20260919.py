"""Read-only, exploratory diagnosis of a sealed study; no new fits or tests."""
import collections
import csv
import hashlib
import json
import math
import statistics as st
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARENT = Path(__file__).resolve().parent
SOURCE = PARENT / 'qwen35_transfer_20260919'
OUT = PARENT / 'qwen35_posthoc_analysis_20260919'
ARMS = ['hidden', 'hidden_budget', 'hidden_both']
LABELS = {'hidden': '普通H', 'hidden_budget': '等参H', 'hidden_both': 'H+E+U'}
EXPECTED = '35af1951cb22c7c414927a847b3aa7848cc2ce0be052633d6b6b052543f114d9'


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest_bytes = (SOURCE / 'SEAL_MANIFEST.json').read_bytes()
    assert sha_bytes(manifest_bytes) == EXPECTED
    manifest = json.loads(manifest_bytes)['files']
    inputs = {}

    def read(rel, jsonl=False):
        data = (SOURCE / rel).read_bytes()
        assert sha_bytes(data) == manifest[rel]['sha256'], rel
        assert len(data) == manifest[rel]['bytes'], rel
        inputs[rel] = sha_bytes(data)
        return [json.loads(x) for x in data.splitlines()] if jsonl else json.loads(data)

    analysis = read('ANALYSIS.json')
    selection = read('SELECTION.json')
    first = read('INITIAL_LOSS_REPRODUCIBILITY.json')
    partitions = read('PAIRED_CASES.json')['comparisons']
    optimization, grid, trajectories, probability = [], [], {}, []
    for task in ['wikisql', 'trec50']:
        for arm in ARMS:
            for r in analysis['results'][task][arm]:
                h = read(f'checkpoints/{r["run"]}/TRAINING.json')['history']
                assert len(h) == 64
                trajectories[r['run']] = h
                row = {'task': task, 'arm': arm, 'seed': r['seed'],
                       'lr': selection['selected'][task][arm]['lr'],
                       'primary': r['primary'],
                       'first16_online_loss': st.mean(x['loss'] for x in h[:16]),
                       'tail8_online_loss': st.mean(x['loss'] for x in h[-8:]),
                       'max_gradient_norm': max(x['grad_norm'] for x in h),
                       'clipped_steps': sum(x['grad_norm'] > 1 for x in h),
                       'mean_clip_factor': st.mean(x['joint_clip_factor'] for x in h),
                       'hidden_update_path': sum(x['group_update_norms']['hidden'] for x in h)}
                for group in ['hidden', 'input', 'output']:
                    row[f'{group}_gradient_energy_share'] = st.mean(
                        x['group_grad_norms'][group] ** 2 /
                        sum(v * v for v in x['group_grad_norms'].values()) for x in h)
                optimization.append(row)
                if task == 'trec50':
                    rs = read(f'evaluations/{r["run"]}/confirm/responses.jsonl', True)
                    nlls, raw, confidences, histogram = [], [], [], collections.Counter()
                    for x in rs:
                        values = x['label_logprobs']
                        peak = max(values)
                        z = peak + math.log(sum(math.exp(v - peak) for v in values))
                        gold = int(x['gold_code'])
                        nlls.append(z - values[gold])
                        raw.append(-values[gold])
                        confidences.append(math.exp(peak - z))
                        histogram[x['predicted_code']] += 1
                    probability.append({'arm': arm, 'seed': r['seed'],
                        'conditional_label_nll': st.mean(nlls), 'raw_gold_code_nll': st.mean(raw),
                        'mean_confidence': st.mean(confidences),
                        'distinct_predicted_labels': len(histogram),
                        'largest_class_count': max(histogram.values())})
        for i in range(6):
            row = {'task': task, 'lr': selection['scores'][task]['hidden'][i]['lr']}
            for arm in ARMS:
                row[arm] = selection['scores'][task][arm][i]['mean']
                tails = []
                for seed in [7600, 7601]:
                    h = read(f'checkpoints/search_{task}_{arm}_c{i}_s{seed}/TRAINING.json')['history']
                    tails.append(st.mean(x['loss'] for x in h[-8:]))
                row[arm + '_tail8_loss'] = st.mean(tails)
            row['delta_H'] = row['hidden_both'] - row['hidden']
            row['delta_budget'] = row['hidden_both'] - row['hidden_budget']
            grid.append(row)

    summaries = []
    for task in ['wikisql', 'trec50']:
        for arm in ARMS:
            rows = [r for r in optimization if r['task'] == task and r['arm'] == arm]
            keys = [k for k in rows[0] if k not in ['task', 'arm', 'seed']]
            summaries.append({'task': task, 'arm': arm, **{k: st.mean(r[k] for r in rows) for k in keys}})
    first_pairs = []
    for group in first['groups']:
        if group['stage'] != 'confirmation':
            continue
        values = {r['arm']: r['first_batch_loss_before_update'] for r in group['runs']}
        first_pairs.append({'task': group['task'], 'seed': group['seed'],
                            'HEU_minus_H': values['hidden_both'] - values['hidden'],
                            'HEU_minus_budget': values['hidden_both'] - values['hidden_budget']})

    OUT.mkdir(exist_ok=True)
    for name, rows in [('development_grid', grid), ('optimization_runs', optimization), ('trec_probability', probability)]:
        with (OUT / (name + '.csv')).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report = {'at': datetime.now().astimezone().isoformat(), 'source_manifest_sha256': EXPECTED,
              'scope': 'Post-hoc descriptive analysis only; no additional fitting, model selection or significance tests.',
              'inputs_checked_against_manifest': inputs, 'script_sha256': sha_bytes(Path(__file__).read_bytes()),
              'development_grid': grid, 'selected_optimization': summaries,
              'optimization_per_run': optimization, 'trec_probability': probability,
              'confirmation_initial_loss_differences': first_pairs, 'paired_cases': partitions,
              'limitations': ['Online tail training losses use changing parameters and different learning rates; not a fixed-checkpoint generalization gap.',
                             'Gradient energy and parameter update norms depend on parameterization; neither measures gradient conflict or functional impact.',
                             'The new conditional label NLL is an exploratory diagnostic, not the prespecified primary metric.',
                             'An equal first-batch scalar does not prove later bitwise reproducibility.',
                             'Single-side benefit, kernel mismatch and optimization mechanisms remain untested causal hypotheses.']}
    (OUT / 'ANALYSIS.json').write_text(json.dumps(report, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    colors = ['#526176', '#d58421', '#166c80']
    for ax, task in zip(axes, ['wikisql', 'trec50']):
        for arm, color in zip(ARMS, colors):
            hs = [trajectories[r['run']] for r in analysis['results'][task][arm]]
            curves = [[st.mean(x['loss'] for x in h[k:k+4]) for k in range(0, 64, 4)] for h in hs]
            xs = list(range(4, 65, 4))
            for curve in curves:
                ax.plot(xs, curve, color=color, alpha=.15, lw=.8)
            lr = selection['selected'][task][arm]['lr']
            label = {'hidden':'H', 'hidden_budget':'Exact-budget H', 'hidden_both':'H+E+U'}[arm]
            ax.plot(xs, [st.mean(v) for v in zip(*curves)], color=color, label=f'{label}, LR={lr:g}', lw=2)
        ax.set_title(task.upper()); ax.set_yscale('log'); ax.grid(alpha=.2)
        ax.set_xlabel('Optimizer step (four-step block means)'); ax.set_ylabel('Online training token loss')
        ax.legend(fontsize=8)
    fig.suptitle('Selected configurations: five seeds, mean in bold; this is not a fixed-checkpoint loss evaluation', fontsize=10)
    fig.savefig(OUT / 'selected_training_curves.png', dpi=180)
    fig.savefig(OUT / 'selected_training_curves.pdf'); plt.close(fig)

    lines = ['# Qwen3.5 叠加负增量：事后分析', '',
             '本次只读取已封存的实验数据；所有使用文件均对照原封存清单核验。没有新训练、换种子、重选配置或额外显著性检验。', '',
             '已经确认：共同学习率下的开发优势，在各自调参后明显缩小。尚待验证：E/U与H联合训练的优化设置、侧别与数值路径是否造成确认阶段的负增量。目前没有单一因果定论，不能仅因新混合架构为负就指认GatedDeltaNet为原因。', '',
             '## 先限定问题', '',
             '本轮只测试H+E+U，不包括H+E、H+U或无H的独立边界适配。论文摘要本身强调侧别随已有适配容量变化，并不主张无条件双侧叠加；该摘要的侧别主张主要来自历史loss实验，不等于新任务已有单侧正证据。因此本轮结果是当前双侧配置的负证据，不能直接外推整个A-LoRA无效，也不能据此宣布U-only会有效。', '',
             '## 完整共同学习率网格（两个开发种子平均）', '',
             '|任务|LR|普通H|等参H|H+E+U|叠加−普通|叠加−等参|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(grid, key=lambda x: (x['task'], x['lr'])):
        lines.append(f'|{r["task"]}|{r["lr"]:g}|{r["hidden"]:.4f}|{r["hidden_budget"]:.4f}|{r["hidden_both"]:.4f}|{r["delta_H"]:+.4f}|{r["delta_budget"]:+.4f}|')
    lines += ['', '固定2e-4时，WikiSQL/TREC叠加相对等参的开发增量为+0.4883/+2.3438pp。各自选参后，WikiSQL开发叠加低于普通H0.8301pp、仅高于等参0.2441pp；TREC叠加与等参持平。五种子确认后，两任务相对等参为−0.9863/−0.4800pp。因此当前变化并不只发生于确认集；调优对照本身已使开发优势大幅缩小。开发和确认分数来自不同数据，不能将其差值当作训练前后收益。旧型号的公平调参尚未全做，不能用此轮结果反推旧型号的所有正增量都仅由学习率造成。', '',
              '## 优化诊断（五个确认种子均值）', '',
              '|任务|方法|LR|前16步在线loss|末8步在线loss|被裁剪步数/64|H参数更新范数之和|', '|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f'|{r["task"]}|{LABELS[r["arm"]]}|{r["lr"]:g}|{r["first16_online_loss"]:.5f}|{r["tail8_online_loss"]:.5f}|{r["clipped_steps"]:.1f}|{r["hidden_update_path"]:.3f}|')
    lines += ['', 'WikiSQL叠加选中8e-4，而普通/等参选中4e-4；不仅边界层，H本身的学习率也翻倍。TREC叠加选中3e-4、对照4e-4。当前比较是完整配置的公平有限搜索，不能解释成“在完全相同的已训练H上加一个小插件”。各方案从配对的共同H初始化开始联合训练，边界零初始化在代数上保留初始函数，但不能保证优化终点或离散任务正确率不下降。', '',
              'WikiSQL确认训练前16步的平均loss，普通/等参约0.181/0.181，叠加约0.363；叠加前期更不稳定，随后才接近对照。这里的学习率同时不同，不能单独归因于E或U。最高候选LR被两个开发种子选中，不等于已证明它在更多新种子上稳定；但也不能事后据确认分数改选。', '',
              '输入侧在WikiSQL/TREC的平均梯度平方范数占比分别约66.05%/71.59%。这与更频繁的联合裁剪一起提示边界优化强度值得检查，但参数化会改变范数，且AdamW按历史一、二阶矩缩放更新，不能把梯度占比解释成“抢走多少学习能力”，也不能从裁剪次数直接推出H更新更小。这里记录的参数更新范数之和也不是函数变化量。', '',
              '末段在线训练loss并未呈现叠加更低的模式，暂不支持直接套用“训练更好、测试更差”的典型过拟合解释。但它不是固定checkpoint在同一完整训练集上的评测，仍不足以排除过拟合。训练/评测分别使用BF16加速与FP32原生路径；是否存在方法相关的数值路径差距也未由本轮日志单独隔离。', '',
              '![训练轨迹](selected_training_curves.png)', '',
              '## 下降发生在哪些题上', '',
              'WikiSQL相对等参平均每2048题少对20.2题；双方查询都有效的分区贡献−0.8594pp，占总净差−0.9863pp的大部分，其他分区−0.1270pp。逻辑形式匹配也从80.4297%降至79.4629%。这使“仅输出格式或停止问题”的解释缺乏支持；分区不是因果分解。TREC相对等参平均每500题少对2.4题，是小增量而非确认阶段崩塌。', '',
              '|TREC方法|候选归一化gold NLL（越低越好）|预测类别数范围|', '|---|---:|---|']
    for arm in ARMS:
        rows = [r for r in probability if r['arm'] == arm]
        lines.append(f'|{LABELS[arm]}|{st.mean(r["conditional_label_nll"] for r in rows):.5f}|{min(r["distinct_predicted_labels"] for r in rows)}–{max(r["distinct_predicted_labels"] for r in rows)}|')
    lines += ['', '该NLL将50个已保存候选分数归一化后计算，只是新的事后概率诊断；不更换主指标、不追加显著性结论。叠加组预测43–44个类别，不是单类别塌缩；NLL均值也较差，下降不只体现在argmax计数。', '',
              '## 数值复现限制仍未解决', '',
              '102次正式拟合的14个阶段/任务/种子组中7组首批损失不完全相同，最大范围0.00858930。确认阶段TREC五个种子，H/等参/叠加首批损失逐个相同；WikiSQL的H和叠加在4/5种子相同，另一个种子反而是叠加略正向。因此不能把所有负增量归咎于已观察到的首批差异；相同首批标量也不证明后续逐位确定。此前三次零更新重放未复现最大异常，来源仍需定位。相同种子并不自动消除所有非确定性；参考[PyTorch 2.6复现文档](https://docs.pytorch.org/docs/2.6/notes/randomness.html)。这份文档不是本轮差异具体来源的证据。', '',
              '## 下一步应能区分原因', '',
              '1. 先做固定同一设备、相同初始权重、输入及RNG状态的逐层重复前向/反向，比较原生与加速路径；检查首个分歧点。另对固定最终checkpoint的小样本比较训练精度与评测精度，估计路径差距。该诊断不替换原成绩。',
              '2. 在新的预先限定开发实验中比较H、H+E、H+U、H+E+U，并给每个单/双侧方案设置对应参数预算对照。固定H学习率，独立调整边界学习率，避免当前共同LR同时改变两部分。必须保留所有方案和负结果，不能据本轮确认分数反复挑参数。',
              '3. 若还需隔离“额外适配能力”和“联合优化干扰”，从同一已训练H checkpoint出发、冻结H后再训练E/U，同时设置同额外训练数据和步骤的内部适配控制。这是不同训练问题，应独立报告，不替代本轮联合训练结果。', '',
              '下一步优先查数值路径、分侧和拆分学习率，比继续扩型号更能区分现有假设。论文可保留参数化与独立低预算适配的证据，对双侧叠加仅提出条件性主张。', '',
              '原始证据：[封存报告](../qwen35_transfer_20260919/FINAL_INTERPRETATION_ZH.md)；[论文中文摘要](../../../emnlp/iclr2027/submissions/paper-2/zh/main.tex)。本目录ANALYSIS.json记录所有实际使用文件的哈希，CSV包含全网格、逐次优化与概率诊断。']
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    assert sha_bytes((SOURCE / 'SEAL_MANIFEST.json').read_bytes()) == EXPECTED
    print(json.dumps({'output': str(OUT), 'verified_source_files': len(inputs), 'formal_training_runs_read': len(trajectories) + 72,
                      'new_fits': 0, 'new_inferential_tests': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
