"""Descriptive review of completed, audited budget comparisons; no new fits."""
import hashlib
import json
import statistics as st
from datetime import datetime
from pathlib import Path

P = Path(__file__).resolve().parent
OUT = P / 'qwen35_budget_review_20260920'
ARMS = ['hidden', 'hidden_budget', 'hidden_both']
LABELS = ['Hidden LoRA', 'Equal-budget LoRA', 'LoRA + aLoRA']
inputs = {}


def read(path):
    raw = path.read_bytes()
    inputs[str(path.relative_to(P))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def main():
    data = {}
    training = {}
    states = {}
    for size in ['08b', '2b', '4b', '9b']:
        root = P / f'qwen35_fixed_{size}_20260920'
        state = read(root / 'STATE.json')
        states[size] = dict(at=state['at'], phase=state['phase'], counts=state['counts'])
        for job in state['jobs']:
            spec = job['spec']
            if job['state'] != 'passed' or spec['stage'] != 'search':
                continue
            audit = read(root / 'audits' / (spec['name'] + '.json'))
            path = root / 'evaluations' / spec['name'] / 'dev' / 'SUMMARY.json'
            result = read(path)
            assert audit['status'] == 'passed'
            assert inputs[str(path.relative_to(P))] == audit['summary_sha256']['dev']
            key = (size, spec['task'], spec['candidate'], spec['seed'], spec['arm'])
            data[key] = result['primary']
            if size == '2b' and spec['task'] == 'wikisql':
                tr = read(root / 'checkpoints' / spec['name'] / 'TRAINING.json')
                assert tr['frozen_before'] == tr['frozen_after'] and tr['optimizer_whitelist_verified']
                h = tr['history']
                training[key] = dict(score=result['primary'], history=h,
                                     evaluation={k: result[k] for k in ['lf_correct_pct', 'query_valid_pct', 'strict_json_pct', 'capped_pct']},
                                     first16=st.mean(x['loss'] for x in h[:16]),
                                     tail8=st.mean(x['loss'] for x in h[-8:]),
                                     clipped_steps=sum(x['joint_clip_factor'] < 1 for x in h))

    groups = []
    for size in ['08b', '2b', '4b', '9b']:
        for task in ['wikisql', 'trec50']:
            for c in range(6):
                candidate = f'c{c}'
                if not all((size, task, candidate, seed, arm) in data
                           for seed in [7600, 7601] for arm in ARMS):
                    continue
                scores = {arm: [data[size, task, candidate, seed, arm] for seed in [7600, 7601]] for arm in ARMS}
                means = {arm: st.mean(v) for arm, v in scores.items()}
                groups.append(dict(size=size, task=task, candidate=candidate, scores=scores, means=means,
                                   budget_minus_hidden=means['hidden_budget']-means['hidden'],
                                   alora_minus_budget=means['hidden_both']-means['hidden_budget'],
                                   alora_minus_hidden=means['hidden_both']-means['hidden']))
    conditional = {}
    for label, predicate in [('up', lambda x: x > 0), ('tie', lambda x: x == 0), ('down', lambda x: x < 0)]:
        rows = [g for g in groups if predicate(g['budget_minus_hidden'])]
        conditional[label] = dict(n=len(rows), alora_above_budget=sum(g['alora_minus_budget'] > 0 for g in rows),
                                  alora_above_both=sum(g['alora_minus_budget'] > 0 and g['alora_minus_hidden'] > 0 for g in rows))

    root = P / 'qwen35_fixed_2b_20260920'
    identities = []
    for seed in [7600, 7601]:
        first = []
        shared = []
        for arm in ARMS:
            names = [f'search_wikisql_{arm}_{c}_s{seed}' for c in ['c2', 'c5']]
            inits = [read(root / 'checkpoints' / n / 'INITIALIZATION.json') for n in names]
            orders = [read(root / 'checkpoints' / n / 'TRAIN_ORDER.json') for n in names]
            assert inits[0]['initialization_sha256'] == inits[1]['initialization_sha256']
            assert orders[0] == orders[1]
            shared.append(inits[0]['shared_hidden_initialization_sha256'])
            first.extend(training['2b', 'wikisql', c, seed, arm]['history'][0]['loss'] for c in ['c2', 'c5'])
        assert len(set(first)) == len(set(shared)) == 1
        identities.append(dict(seed=seed, first_loss=first[0], common_hidden_initialization=True,
                               same_initialization_and_order_between_lr_4e4_and_8e4=True))
    plan = read(root / 'BUDGET_ALLOCATION.json')
    at = datetime.now().astimezone().isoformat(timespec='seconds')
    OUT.mkdir(exist_ok=True)
    artifact = dict(at=at, scope='Posthoc development-only snapshot; no causal or significance claims; no protocol changes.',
                    states=states, groups=groups, conditional=conditional, identity_checks=identities,
                    budget_plan={k: plan[k] for k in ['extra', 'hidden_parameters', 'rank_pattern', 'extra_rank_counts', 'rule']},
                    training={'|'.join(map(str, k)): v for k, v in training.items()}, inputs=inputs)
    (OUT / 'ANALYSIS.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, seed in zip(axes, [7600, 7601]):
        for arm, label in zip(ARMS, LABELS):
            h = training['2b', 'wikisql', 'c5', seed, arm]['history']
            ax.plot([x['step'] for x in h], [x['loss'] for x in h], label=label, linewidth=1.3)
        ax.set_title(f'Seed {seed}, LR=8e-4')
        ax.set_xlabel('Optimizer step'); ax.set_yscale('log'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Online minibatch loss (log scale)'); axes[1].legend(fontsize=8)
    fig.suptitle('Qwen3.5-2B / WikiSQL: configuration and seed sensitivity')
    fig.tight_layout(); fig.savefig(OUT / 'training_instability.png', dpi=160); fig.savefig(OUT / 'training_instability.pdf'); plt.close(fig)

    lines = ['# 等参扩容与 aLoRA 增量的阶段复核', '', f'快照：{at}', '',
             f'只纳入三个方法均完成、相同学习率、两个开发种子均齐的 {len(groups)} 组；不混入单种子或独立选参后的比较。不同学习率复用了数据和种子，不能当独立重复实验。', '',
             '|等参相对普通|组合数|aLoRA 高于等参|aLoRA 高于两个对照|', '|---|---:|---:|---:|']
    for key, label in [('up', '提高'), ('tie', '持平'), ('down', '下降')]:
        x = conditional[key]; lines.append(f'|{label}|{x["n"]}|{x["alora_above_budget"]}|{x["alora_above_both"]}|')
    lines += ['', '这是已完成开发组合的描述，不能证明“等参上涨导致aLoRA增益”。两个差值共享等参分数，也不适合直接用简单相关系数解释机制。', '',
              '## 2B WikiSQL：相同学习率的两种子均值', '',
              '|学习率|普通 LoRA|等参 LoRA|叠加 aLoRA|等参−普通|aLoRA−等参|', '|---|---:|---:|---:|---:|---:|']
    lrs = [2e-4, 1e-4, 4e-4, 5e-5, 3e-4, 8e-4]
    for g in groups:
        if g['size'] != '2b' or g['task'] != 'wikisql': continue
        v=g['means'];lr=lrs[int(g['candidate'][1:])]
        lines.append(f'|{lr:g}|{v["hidden"]:.4f}|{v["hidden_budget"]:.4f}|{v["hidden_both"]:.4f}|{g["budget_minus_hidden"]:+.4f}|{g["alora_minus_budget"]:+.4f}|')
    lines += ['', '## 8e-4 的逐种子异常', '', '|种子|普通 LoRA|等参 LoRA|叠加 aLoRA|', '|---|---:|---:|---:|']
    for seed in [7600, 7601]:
        vals=[data['2b','wikisql','c5',seed,arm] for arm in ARMS]
        lines.append(f'|{seed}|{vals[0]:.4f}|{vals[1]:.4f}|{vals[2]:.4f}|')
    lines += ['', '等参的较大下降集中在7600；叠加组的严重下降集中在7601，不能推定同一个原因。叠加7601第4步loss为4.7717、梯度范数464.55，第5步loss为7.3363，64步均发生全局梯度裁剪。4e-4的同种子叠加结果为81.4453%。',
              '同种子在4e-4/8e-4之间初始化与训练顺序相同，三方法共享hidden初始化及首步loss一致，基座冻结与优化器白名单检查通过。支持高学习率配置敏感性；没有直接定位到实现bug，也不证明完整训练轨迹跨设备逐位确定。梯度量级依赖参数化，裁剪次数不构成裁剪致因的证明。',
              '各方法独立调参时，叠加与等参选择4e-4、普通选择8e-4；不应将未选中的8e-4失稳直接当成所选4e-4无增益的原因。', '',
              '8e-4种子7600的等参下降并非更差的格式指标：普通/等参查询有效率99.5117/99.7070%，严格JSON率均99.8047%，但逻辑形式匹配77.7344/74.8047%、执行正确率82.3242/79.1016%。种子7601的叠加查询有效率仍为94.8242%、严格JSON率98.1445%、无长度截断，执行正确率仅31.3477%；不能把这次大幅下降仅归于格式。', '',
              '等参组末8步在线loss在4e-4为0.068363、普通为0.068079；8e-4分别0.064502/0.063696，差距较小，尚不能判定普遍优化失败。在线loss不等于固定checkpoint在同一训练集上的损失，无法单凭这些曲线区分过拟合与优化不足。', '',
              '## 参数分配检查', '',
              '2B普通8,409,600参数，等参/叠加8,542,720参数，新增133,120。等参方案将14个融合QKV投影和3个完整注意力Q投影由rank8扩为rank9，保留共有rank8初始化和缩放2；额外参数确实更新。公式和预算检查通过，但这不能证明该分配是强基线或最优分配。',
              '下一步诊断应分别验证：等参扩容位置敏感性，以及叠加组在相同hidden学习率下的边界学习率/预热敏感性；另用固定checkpoint共同训练子集评分区分训练拟合与开发变化。应另设诊断目录、预先固定方案，不修改或替换当前冻结结果，不强制对照分数单调。此复核没有启动这些新实验。', '',
              '![训练曲线](training_instability.png)', '', '完整逐组合、逐步训练数据及来源哈希见 ANALYSIS.json。']
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(at=at, groups=len(groups), conditional=conditional, report=str(OUT/'REPORT_ZH.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()
