"""Report audited stage results without changing any frozen experiment decision."""
import hashlib
import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

PARENT = Path(__file__).resolve().parent
OUT = PARENT / 'qwen35_fixed_multiscale_20260920' / 'stage_review'
ARMS = ['hidden', 'hidden_budget', 'hidden_both']
LRS = [2e-4, 1e-4, 4e-4, 5e-5, 3e-4, 8e-4]
SIZES = [('08b', '0.8B'), ('2b', '2B'), ('4b', '4B'), ('9b', '9B')]
TASKS = ['wikisql', 'trec50']
STAGE_LABELS = {'dev_seed7600': '开发：第1种子，单独选参',
                'dev_seed7601': '开发：第2种子，单独选参',
                'dev_two_seeds': '开发：两种子完整搜索',
                'confirm_1_seeds': '确认：1个配对种子',
                'confirm_3_seeds': '确认：3个配对种子',
                'confirm_5_seeds': '确认：5个配对种子'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def summarize(arms):
    comparisons = {}
    for control in ARMS[:2]:
        diffs = [x - y for x, y in zip(arms['hidden_both']['scores'], arms[control]['scores'])]
        comparisons[control] = dict(mean=statistics.mean(diffs), paired_differences=diffs,
                                    positive=sum(x > 0 for x in diffs),
                                    tied=sum(x == 0 for x in diffs), negative=sum(x < 0 for x in diffs))
    return dict(arms=arms, comparisons=comparisons)


def main():
    at = datetime.now().astimezone().isoformat(timespec='seconds')
    records = []
    status = {}
    for key, size in SIZES:
        root = PARENT / f'qwen35_fixed_{key}_20260920'
        state = read(root / 'STATE.json')
        status[size] = dict(phase=state['phase'], counts=state['counts'])
        data = {}
        for job in state['jobs']:
            spec = job['spec']
            if job['state'] != 'passed' or spec['stage'] not in ['search', 'confirmation'] or spec['smoke']:
                continue
            audit_path = root / 'audits' / (spec['name'] + '.json')
            audit = read(audit_path)
            assert audit['status'] == 'passed'
            split = 'dev' if spec['stage'] == 'search' else 'confirm'
            path = root / 'evaluations' / spec['name'] / split / 'SUMMARY.json'
            digest = sha(path)
            assert digest == audit['summary_sha256'][split], path
            result = read(path)
            expected_n = {'dev': {'wikisql': 1024, 'trec50': 256},
                          'confirm': {'wikisql': 2048, 'trec50': 500}}[split][spec['task']]
            assert result['n'] == expected_n and math.isfinite(result['primary'])
            assert 0 <= result['primary'] <= 100
            idx = (spec['stage'], spec['task'], spec['arm'], spec['candidate'], spec['seed'])
            assert idx not in data
            data[idx] = dict(score=result['primary'], lr=spec['lr'],
                             inputs={str(path.relative_to(PARENT)): digest,
                                     str(audit_path.relative_to(PARENT)): sha(audit_path)})

        selection_path = root / 'SELECTION.json'
        selection = read(selection_path) if selection_path.exists() else None
        if selection:
            for path, digest in selection['inputs'].items():
                assert sha(root / path) == digest, path

        for task in TASKS:
            for seeds, label in [([7600], 'dev_seed7600'), ([7601], 'dev_seed7601'),
                                 ([7600, 7601], 'dev_two_seeds')]:
                indices = [("search", task, arm, f'c{c}', seed)
                           for arm in ARMS for c in range(6) for seed in seeds]
                if not all(idx in data for idx in indices):
                    continue
                arms = {}
                grid = {}
                for arm in ARMS:
                    rows = [dict(candidate=f'c{c}', lr=LRS[c],
                                 scores=[data['search', task, arm, f'c{c}', seed]['score'] for seed in seeds])
                            for c in range(6)]
                    for row in rows:
                        row['mean'] = statistics.mean(row['scores'])
                    grid[arm] = rows
                    arms[arm] = max(rows, key=lambda row: row['mean'])
                    if len(seeds) == 2 and selection:
                        assert arms[arm]['candidate'] == selection['selected'][task][arm]['id']
                inputs = {path: digest for idx in indices for path, digest in data[idx]['inputs'].items()}
                if len(seeds) == 2 and selection:
                    inputs[str(selection_path.relative_to(PARENT))] = sha(selection_path)
                records.append(dict(id=f'{key}_{task}_{label}', size=size, task=task,
                                    stage=label, seeds=seeds, selected_on_reported_data=True,
                                    official_selection_available=bool(len(seeds) == 2 and selection),
                                    grid=grid, inputs=inputs, **summarize(arms)))

            if not selection:
                continue
            for count in [1, 3, 5]:
                seeds = list(range(7700, 7700 + count))
                indices = [('confirmation', task, arm, selection['selected'][task][arm]['id'], seed)
                           for arm in ARMS for seed in seeds]
                if not all(idx in data for idx in indices):
                    continue
                arms = {}
                for arm in ARMS:
                    chosen = selection['selected'][task][arm]
                    scores = [data['confirmation', task, arm, chosen['id'], seed]['score'] for seed in seeds]
                    arms[arm] = dict(candidate=chosen['id'], lr=chosen['lr'], scores=scores,
                                     mean=statistics.mean(scores))
                inputs = {path: digest for idx in indices for path, digest in data[idx]['inputs'].items()}
                inputs[str(selection_path.relative_to(PARENT))] = sha(selection_path)
                records.append(dict(id=f'{key}_{task}_confirm_{count}_seeds', size=size, task=task,
                                    stage=f'confirm_{count}_seeds', seeds=seeds,
                                    selected_on_reported_data=False, inputs=inputs, **summarize(arms)))

    OUT.mkdir(exist_ok=True)
    (OUT / 'milestones').mkdir(exist_ok=True)
    new = []
    for record in records:
        target = OUT / 'milestones' / (record['id'] + '.json')
        if target.exists():
            old = read(target)
            # A later official selection can add provenance to an earlier complete task grid.
            for name in ['size', 'task', 'stage', 'seeds', 'arms', 'comparisons']:
                assert old[name] == record[name], (record['id'], name)
        else:
            write(target, json.dumps(dict(observed_at=at, **record), ensure_ascii=False, indent=2) + '\n')
            new.append({k: record[k] for k in ['id', 'size', 'task', 'stage', 'arms', 'comparisons']})
    snapshot = dict(at=at, status=status, records=records, reporter_sha256=sha(Path(__file__)))
    write(OUT / 'LATEST.json', json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Qwen3.5 阶段结果', '', f'更新时间：{at}', '',
             '只读取已通过单项核验的结果，并再次核对评分文件哈希。开发结果是在同一开发数据上选参后的描述，不能代替确认集结果。',
             '单开发种子仅作阶段诊断；正式选择使用两个开发种子。确认阶段仅汇总完整配对的前1、3、5种子，不进行中途显著性检验、提前停止或改参；最终结论等待原定五种子及完整审计。', '',
             '分数为正确率（%），增量为百分点；开发阶段每行的三种方法分别按该行所含种子选取最优学习率。单种子行与双种子行可能选择不同配置，不可混算。', '',
             '|尺寸|任务|阶段|普通 LoRA|严格等参 LoRA|叠加 aLoRA|相对普通|相对等参|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for record in records:
        a = record['arms']; c = record['comparisons']
        lines.append(f"|{record['size']}|{record['task']}|{STAGE_LABELS[record['stage']]}|{a['hidden']['mean']:.4f}|{a['hidden_budget']['mean']:.4f}|{a['hidden_both']['mean']:.4f}|{c['hidden']['mean']:+.4f}|{c['hidden_budget']['mean']:+.4f}|")
    lines += ['', '完整学习率网格、逐种子分数及输入哈希见 LATEST.json。此报告不改变冻结的训练、选择和统计协议。', '']
    write(OUT / 'LATEST_ZH.md', '\n'.join(lines))
    print(json.dumps(dict(at=at, milestone_count=len(records), new_milestones=new), ensure_ascii=False))


if __name__ == '__main__':
    main()
