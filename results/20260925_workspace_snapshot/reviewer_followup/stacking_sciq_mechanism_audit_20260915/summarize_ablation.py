"""Descriptive paired analysis; no post hoc configuration selection."""
import math
import numpy as np
from support import *
from analyze_predictions import interval, arrays

MODES = ['full', 'input_only', 'output_only', 'both_off']

def center(x):
    return x - x.mean(axis=-1, keepdims=True)

def compensation(full, off, control):
    boundary = center(full - off)
    hidden_difference = center(off - control)
    total = center(full - control)
    assert np.max(np.abs(total - boundary - hidden_difference)) < 1e-12
    aa = np.mean(boundary ** 2)
    bb = np.mean(hidden_difference ** 2)
    cross = np.mean(boundary * hidden_difference)
    return {
        'boundary_rms': float(np.sqrt(aa)),
        'hidden_difference_rms': float(np.sqrt(bb)),
        'total_rms': float(np.sqrt(np.mean(total ** 2))),
        'cosine_boundary_hidden_difference': float(cross / np.sqrt(aa * bb)),
        'total_energy_over_sum_component_energies': float(np.mean(total ** 2) / (aa + bb)),
    }

def main():
    rr = [read(p) for p in sorted((HERE / 'ablations').glob('*/RESULTS.json'))]
    assert len(rr) == 10
    old = {(s['model'], s['arm'], s['seed']): arrays(load_predictions(s)) for s in original_specs()}
    out = {'at': now(), 'scope': 'Post hoc descriptive, five paired training seeds; rotations are repeated observations.', 'models': {}}
    for model in read(HERE / 'models.json'):
        results = [r for r in rr if r['spec']['model'] == model]
        assert [r['spec']['seed'] for r in results] == list(range(2002, 2007))
        tokens = read(HERE / 'tokens' / f'{model}_test_rotations.json')
        clean = np.array([not x['ambiguous_gold'] for x in tokens])
        m = {'modes': {}, 'paired_effects': {}, 'compensation': {}, 'factorial_logit_interaction': {}, 'geometry': {}}
        for mode in MODES:
            m['modes'][mode] = {split: {metric: interval([r['metrics'][mode][split][metric] for r in results])
                for metric in results[0]['metrics'][mode][split] if metric != 'n'} for split in ['canonical', 'rotations']}
        m['unadapted_base'] = results[0]['metrics']['unadapted_base']
        for name, left, right in [('full_minus_both_off', 'full', 'both_off'),
                ('input_effect_output_on', 'full', 'output_only'),
                ('input_effect_output_off', 'input_only', 'both_off'),
                ('output_effect_input_on', 'full', 'input_only'),
                ('output_effect_input_off', 'output_only', 'both_off')]:
            m['paired_effects'][name] = {split: {metric: interval([
                r['metrics'][left][split][metric] - r['metrics'][right][split][metric] for r in results])
                for metric in ['accuracy', 'candidate_nll', 'label_mass_penalty']} for split in ['canonical', 'rotations']}
        all_comp = {control: [] for control in ['none', 'hidden_budget']}
        interactions = []
        for r in results:
            seed = r['spec']['seed']
            path = HERE / 'ablations' / r['spec']['name'] / 'scores.npz'
            assert sha(path) == r['scores_sha256']
            with np.load(path) as npz:
                a = {mode: npz[mode][clean].astype(float).reshape(998, 4, 4) for mode in MODES}
                assert np.array_equal(a['full'], old[model, 'both', seed]['logits'])
                for control in all_comp:
                    all_comp[control].append(compensation(a['full'], a['both_off'], old[model, control, seed]['logits']))
                interaction = center(a['full'] - a['input_only'] - a['output_only'] + a['both_off'])
                interactions.append(float(np.sqrt(np.mean(interaction ** 2))))
        for control, records in all_comp.items():
            m['compensation'][control] = {key: interval([r[key] for r in records]) for key in records[0]}
        m['factorial_logit_interaction']['rms'] = interval(interactions)
        m['geometry']['relative_label_matrix'] = {key: results[0]['geometry'][key] for key in [
            'relative_label_matrix_shape', 'rank', 'singular_values', 'condition_number', 'right_inverse_max_error', 'arbitrary_rank3_relative_correction_max_error']}
        m['geometry']['learned_relative_update_frobenius_ratio'] = interval([r['geometry']['learned_relative_update_frobenius_ratio'] for r in results])
        for mode in ['input', 'output']:
            m['geometry'][mode] = {key: interval([r['geometry'][mode + '_map'][key] for r in results])
                for key in ['linear_residual_frobenius', 'linear_residual_spectral', 'bias_norm']}
        out['models'][model] = m
    write(HERE / 'ABLATION_ANALYSIS.json', out)
    for model, m in out['models'].items():
        print(model, 'base', m['unadapted_base']['rotations']['accuracy'])
        for key, val in m['paired_effects'].items():
            print(key, 'rot accuracy', val['rotations']['accuracy'], 'rot NLL', val['rotations']['candidate_nll'])
        print('compensation', m['compensation'])
        print('interaction', m['factorial_logit_interaction'])

if __name__ == '__main__':
    main()
