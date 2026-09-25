"""Post-hoc descriptive audit of all TREC development fits for one seed.

No selection, extra fitting, inferential tests, or changes to the frozen study.
"""
import argparse
import collections
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'qwen35_transfer_20260919'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[7600, 7601], default=7600)
    seed = parser.parse_args().seed
    assert not (ROOT / 'SEAL.json').exists()
    inputs = {}
    results = []
    for arm in ['hidden', 'hidden_budget', 'hidden_both']:
        for candidate in range(6):
            name = f'search_trec50_{arm}_c{candidate}_s{seed}'
            paths = {
                'training': ROOT / 'checkpoints' / name / 'TRAINING.json',
                'summary': ROOT / 'evaluations' / name / 'dev/SUMMARY.json',
                'responses': ROOT / 'evaluations' / name / 'dev/responses.jsonl',
                'candidate_audit': ROOT / 'evaluations' / name / 'dev/CANDIDATE_AUDIT.json',
                'audit': ROOT / 'audits' / f'{name}.json',
            }
            assert read(paths['audit'])['status'] == 'passed'
            for path in paths.values():
                inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            history = read(paths['training'])['history']
            summary = read(paths['summary'])
            responses = [json.loads(line) for line in paths['responses'].read_text().splitlines()]
            counts = collections.Counter(row['predicted_code'] for row in responses)
            shares = {
                group: statistics.mean(
                    row['group_grad_norms'][group] ** 2 /
                    sum(value ** 2 for value in row['group_grad_norms'].values())
                    for row in history
                )
                for group in ['hidden', 'input', 'output']
            }
            results.append({
                'run': name,
                'arm': arm,
                'candidate': candidate,
                'lr': summary['spec']['lr'],
                'primary': summary['primary'],
                'macro_f1_all50': summary['macro_f1_all50'],
                'coarse_accuracy': summary['coarse_accuracy'],
                'first8_mean_loss': statistics.mean(row['loss'] for row in history[:8]),
                'last8_mean_loss': statistics.mean(row['loss'] for row in history[-8:]),
                'max_loss': max(row['loss'] for row in history),
                'max_loss_step': max(history, key=lambda row: row['loss'])['step'],
                'median_grad_norm': statistics.median(row['grad_norm'] for row in history),
                'max_grad_norm': max(row['grad_norm'] for row in history),
                'clipped_steps': sum(row['grad_norm'] > 1 for row in history),
                'mean_group_gradient_energy_share': shares,
                'distinct_predicted_codes': len(counts),
                'predicted_code_counts': dict(sorted(counts.items())),
                'native_candidate_audit': read(paths['candidate_audit']),
            })
    report = {
        'at': datetime.now().astimezone().isoformat(),
        'scope': f'Post-hoc descriptive analysis of all 18 seed-{seed} TREC50 development fits; not confirmation evidence.',
        'limitations': [
            'Training-loss windows, gradient summaries and label histograms are descriptive, not causal evidence.',
            'Gradient energy shares depend on parameterization and do not isolate effects of E, U, global clipping or learning rate.',
            'Native candidate audits cover the two predeclared examples per fit; all responses have separate finite-score, argmax, label and aggregate audits.',
            'No frozen protocol, grid, selection rule or confirmation configuration is changed by this report.',
        ],
        'inputs_sha256': inputs,
        'results': results,
    }
    destination = ROOT / f'DEV_SEED{seed}_TREC_DIAGNOSTICS.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'audited_fits': len(results)}))


if __name__ == '__main__':
    main()
