"""Descriptive token-budget and development-loss diagnostics; no model selection."""
import numpy as np
from common import *


def main():
    lengths = []
    losses = []
    for task in TASKS:
        for model in MODELS:
            for split in ['pilot_train', 'pilot_dev', 'test']:
                records = read(HERE / f'tokens/{task}_{model}_{split}.json')
                prompt_lengths = [len(r['prompt_ids']) for r in records]
                target_lengths = [len(r['target_ids']) for r in records]
                lengths.append({'task': task, 'model': model, 'split': split, 'n': len(records),
                    'prompt_p50_p95_max': np.quantile(prompt_lengths, [.5, .95, 1]).tolist(),
                    'target_including_eos_p50_p95_max': np.quantile(target_lengths, [.5, .95, 1]).tolist(),
                    'targets_exceeding_512_including_eos': sum(n > 512 for n in target_lengths),
                    'eos_fraction_of_supervised_tokens': len(records) / sum(target_lengths)})
            dev = read(HERE / f'tokens/{task}_{model}_pilot_dev.json')
            eos_n = len(dev)
            answer_n = sum(len(r['target_ids']) - 1 for r in dev)
            base = read(HERE / 'evaluations' / f'pilot_{task}_{model}_base' / 'baseline/SUMMARY.json')
            for arm in ['hidden', 'input', 'output', 'hidden_both', 'hidden_budget']:
                values = []
                for seed in range(6100, 6105):
                    name = (f'pilot_{task}_{model}_hidden' if arm == 'hidden' and seed == 6100
                            else f'adapter_{task}_{model}_{arm}_s{seed}')
                    values.append(read(HERE / 'evaluations' / name / 'final/SUMMARY.json'))
                answer = float(np.mean([v['answer_ce'] for v in values]))
                eos = float(np.mean([v['eos_ce'] for v in values]))
                answer_drop = (base['answer_ce'] - answer) * answer_n / (answer_n + eos_n)
                eos_drop = (base['eos_ce'] - eos) * eos_n / (answer_n + eos_n)
                total_drop = answer_drop + eos_drop
                losses.append({'task': task, 'model': model, 'arm': arm,
                    'base_answer_ce': base['answer_ce'], 'mean_answer_ce': answer,
                    'base_eos_ce': base['eos_ce'], 'mean_eos_ce': eos,
                    'answer_contribution_to_total_ce_drop': answer_drop,
                    'eos_contribution_to_total_ce_drop': eos_drop,
                    'eos_share_of_total_ce_drop': eos_drop / total_drop if total_drop > 0 else None})
    write(HERE / 'TOKEN_LENGTH_AND_LOSS_DIAGNOSTIC.json', {'at': now(),
        'scope': 'post-hoc descriptive development diagnostic; held-out gold used only for token-budget checks',
        'warning': 'Answer CE includes JSON syntax and copied tokens; its improvement is not a content-accuracy claim.',
        'lengths': lengths, 'development_loss_decomposition': losses})
    for row in losses:
        if row['arm'] == 'hidden':
            print(row['task'], row['model'], 'EOS share of CE drop', row['eos_share_of_total_ce_drop'])


if __name__ == '__main__':
    main()
