"""Post-hoc descriptive checks of all TREC development fits; no new fits or selection."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'gemma_transfer_20260918'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (ROOT / 'SEAL.json').exists()
    selection = read(ROOT / 'SELECTION.json')
    assert selection['status'] == 'frozen'
    results = []
    for arm in ['hidden', 'hidden_budget', 'hidden_both']:
        for candidate in selection['scores']['trec50'][arm]:
            for seed in [7400, 7401]:
                name = f'search_trec50_{arm}_{candidate["id"]}_s{seed}'
                run = ROOT / 'evaluations' / name / 'dev'
                audit = ROOT / 'audits' / f'{name}.json'
                training = ROOT / 'checkpoints' / name / 'TRAINING.json'
                assert read(audit)['status'] == 'passed'
                assert read(audit)['summary_sha256']['dev'] == sha(run / 'SUMMARY.json')
                summary = read(run / 'SUMMARY.json')
                assert summary['responses_sha256'] == sha(run / 'responses.jsonl')
                rs = [json.loads(x) for x in (run / 'responses.jsonl').read_text().splitlines()]
                h = read(training)['history']
                assert len(rs) == 256 and len(h) == 64
                accuracy = 100 * sum(r['content_correct'] for r in rs) / len(rs)
                assert accuracy == summary['primary']
                histogram = collections.Counter(r['predicted_code'] for r in rs)
                maximum = max(h, key=lambda r: r['grad_norm'])
                total_energy = sum(v * v for v in maximum['group_grad_norms'].values())
                sources = [audit, training, run / 'SUMMARY.json', run / 'responses.jsonl', run / 'CANDIDATE_AUDIT.json']
                results.append({
                    'run': name, 'arm': arm, 'lr': candidate['lr'], 'seed': seed,
                    'accuracy_pct': accuracy, 'distinct_predicted_codes': len(histogram),
                    'predicted_code_histogram': dict(sorted(histogram.items())),
                    'first_loss': h[0]['loss'],
                    'tail8_training_loss': statistics.mean(x['loss'] for x in h[-8:]),
                    'block8_training_losses': [statistics.mean(x['loss'] for x in h[i:i+8]) for i in range(0, 64, 8)],
                    'maximum_gradient_norm': maximum['grad_norm'], 'maximum_gradient_step': maximum['step'],
                    'gradient_energy_share_at_maximum': {k: v*v/total_energy for k, v in maximum['group_grad_norms'].items()},
                    'clipped_steps': sum(x['grad_norm'] > 1 for x in h),
                    'native_candidate_check': read(run / 'CANDIDATE_AUDIT.json'),
                    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in sources},
                })
    collapsed = next(r for r in results if r['run'] == 'search_trec50_hidden_both_c5_s7400')
    assert collapsed['predicted_code_histogram'] == {'45': 256}
    document = {
        'status': 'passed', 'scope': 'All 36 TREC development fits only; no confirmation results inspected.',
        'analysis_type': 'Post-hoc descriptive diagnostic prompted by one low development score; no new fits, candidate choices, or significance tests.',
        'selection_sha256': sha(ROOT / 'SELECTION.json'), 'results': results,
        'limitations': [
            'Native reference checks use the two prespecified examples, not a second native evaluation of all 256 questions.',
            'Losses, gradient norms and prediction collapse are associations; they do not establish a causal role for clipping, E bias, U, or soft-capping.',
            'Two development seeds cannot estimate a reliable collapse probability; selected configurations still require the five-seed confirmation.',
        ],
    }
    (ROOT / 'DEVELOPMENT_OPTIMIZATION_DIAGNOSTICS.json').write_text(json.dumps(document, indent=2) + '\n')
    lines = ['# 开发阶段的优化敏感性：事后描述性核对', '',
             '本记录在开发选择冻结后生成，检查全部 36 次 TREC 开发拟合；未查看确认集结果，也未增加训练、修改候选或重做选择。', '',
             '叠加组在 LR=8e-4、seed=7400 时，256 条开发问题全部预测为代码 45，正确率 2.734375%；另一种子为 83.593750%。同样高学习率下两个内部 LoRA 对照没有出现这次单类别塌缩。', '',
             '该次运行的末八步训练 loss 为 1.257604，最大梯度范数为 248.067505；同种子、选中的 LR=4e-4 叠加组分别为 0.185943 与 33.897469。这些现象与优化不稳定一致，但不足以将原因归于某个模块、裁剪或 soft-capping。', '',
             '参数范围、冻结基座、保存重载与评分审计均已通过；预先指定的两条原生候选概率检查误差为零、类别选择一致。该检查覆盖两条样本，不能描述成全量原生重复评测。', '',
             '两个开发种子不足以估计可靠的塌缩概率。三种方法的同一六学习率网格完整保留，所有 TREC 方法按开发均值选中 4e-4；五种子确认仍按原方案执行。', '',
             '| 方法 | LR | 种子 | 开发准确率% | 预测类别数 | 末八步 loss | 最大梯度范数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["arm"]} | {r["lr"]:g} | {r["seed"]} | {r["accuracy_pct"]:.6f} | {r["distinct_predicted_codes"]} | {r["tail8_training_loss"]:.6f} | {r["maximum_gradient_norm"]:.6f} |')
    lines += ['', '完整分布、梯度峰值步及文件指纹见 `DEVELOPMENT_OPTIMIZATION_DIAGNOSTICS.json`。']
    (ROOT / 'DEVELOPMENT_INSTABILITY_ZH.md').write_text('\n'.join(lines) + '\n')
    (ROOT / 'provenance' / (Path(__file__).name + '.txt')).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': 'passed', 'development_runs': len(results), 'single_code_runs': [r['run'] for r in results if r['distinct_predicted_codes'] == 1]}), flush=True)


if __name__ == '__main__':
    main()
