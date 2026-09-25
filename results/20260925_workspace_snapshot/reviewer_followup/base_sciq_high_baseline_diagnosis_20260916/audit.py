"""Recompute saved diagnostic scores and verify the sealed source, without inference."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'standalone_base_native_sciq_20260916'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    preflight = read(HERE / 'PREFLIGHT.json')
    assert sha(SOURCE / 'ARTIFACT_MANIFEST.json') == preflight['source_manifest_sha256']
    assert sha(HERE / 'diagnose.py') == preflight['diagnostic_code_sha256']
    assert sha(HERE / 'DESIGN.md') == preflight['design_sha256']
    manifest = read(SOURCE / 'ARTIFACT_MANIFEST.json')
    for item in manifest['files']:
        path = SOURCE / item['path']
        assert path.stat().st_size == item['bytes']
        assert sha(path) == item['sha256'], path
    data = rows(SOURCE / 'data/test.jsonl')
    assert len(data) == 1000
    tokenization = read(SOURCE / 'DATA_AUDIT.json')['tokenization']
    reports, permutations = {}, []
    for model in ['qwen3_06b_base', 'qwen25_15b_base']:
        report = read(HERE / f'{model}_RESULTS.json')
        permutation = read(HERE / f'{model}_question_permutation.json')
        assert sorted(permutation) == list(range(1000))
        assert all(i != j for i, j in enumerate(permutation))
        permutations.append(permutation)
        labels = tokenization[model]['label_ids']
        for condition, result in report['conditions'].items():
            path = HERE / f'{model}_{condition}.jsonl'
            assert sha(path) == result['prediction_sha256']
            predictions = rows(path)
            assert len(predictions) == len(data)
            for p, source in zip(predictions, data):
                assert all(p[k] == source[k] for k in ['id', 'gold', 'ambiguous_gold'])
                assert len(p['label_logits']) == 4
                assert all(math.isfinite(x) for x in p['label_logits'])
                assert p['prediction'] == max(range(4), key=lambda i: p['label_logits'][i])
                assert p['correct'] == (p['prediction'] == p['gold'])
                assert p['unrestricted_correct'] == (p['unrestricted_token'] == labels[p['gold']])
                assert p['valid_label'] == (p['unrestricted_token'] in labels)
            clean = [p for p in predictions if not p['ambiguous_gold']]
            assert len(clean) == result['n'] == 998
            for key, row_key in [('candidate_accuracy', 'correct'),
                                 ('unrestricted_first_token_accuracy', 'unrestricted_correct'),
                                 ('valid_label_rate', 'valid_label')]:
                recomputed = 100 * sum(p[row_key] for p in clean) / len(clean)
                assert abs(recomputed - result[key]) < 1e-10
        reports[model] = report['conditions']
    assert permutations[0] == permutations[1]
    audit = {
        'at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'status': 'passed',
        'source_sealed_files_reverified': len(manifest['files']),
        'source_manifest_sha256': sha(SOURCE / 'ARTIFACT_MANIFEST.json'),
        'diagnostic_rows_recomputed': 6000,
        'same_valid_derangement_for_both_models': True,
        'unchanged_diagnostic_code_and_design': True,
        'original_prediction_flips': {m: r['original']['independent_reference_prediction_flips'] for m, r in reports.items()},
        'conditions': reports,
    }
    (HERE / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'conditions'}, indent=2))


if __name__ == '__main__':
    main()
