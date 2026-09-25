"""Validate all diagnostic artifacts and summarize without replacing old results."""
import math
import statistics
from collections import Counter
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer

from cross import BASE, CHAT, HERE, read, rows, sha, write


def answer(text):
    text = text.lstrip()
    if not text or text[0] not in 'ABCD':
        return None
    if len(text) > 1 and not (text[1].isspace() or text[1] in '.,:;)]'):
        return None
    return 'ABCD'.index(text[0])


def paired(left, right):
    assert [r['id'] for r in left] == [r['id'] for r in right]
    l = np.array([r['correct'] for r in left if not r['ambiguous_gold']], dtype=float)
    r = np.array([r['correct'] for r in right if not r['ambiguous_gold']], dtype=float)
    d = l - r
    rng = np.random.default_rng(20260916)
    boot = np.concatenate([d[rng.integers(0, len(d), (1000, len(d)))].mean(1) * 100 for _ in range(10)])
    return {'left_minus_right_pp': float(100 * d.mean()),
            'nominal_question_bootstrap95_pp': np.quantile(boot, [.025, .975]).tolist(),
            'corrected': int((d == 1).sum()), 'regressed': int((d == -1).sum()),
            'inference': 'descriptive, conditional on these predictions; no multiplicity correction or sampling-seed uncertainty'}


def main():
    preflight = read(HERE / 'PREFLIGHT.json')
    assert sha(HERE / 'DESIGN.md') == preflight['design_sha256']
    assert sha(HERE / 'cross.py') == preflight['code_sha256']
    tp = read(HERE / 'THINKING_PREFLIGHT.json')
    for filename, key in [('THINKING_DESIGN.md', 'design_sha256'), ('thinking.py', 'code_sha256'), ('cross.py', 'cross_code_sha256')]:
        assert sha(HERE / filename) == tp[key]
    count = 0
    for source in [BASE, CHAT]:
        assert sha(source / 'ARTIFACT_MANIFEST.json') == preflight[source.name]['manifest_sha256']
        manifest = read(source / 'ARTIFACT_MANIFEST.json')
        for item in manifest['files']:
            assert sha(source / item['path']) == item['sha256']
            count += 1
    data = rows(BASE / 'data/test.jsonl')
    models = {}
    for name in ['qwen3_06b_base', 'qwen3_06b_chat', 'qwen25_15b_base', 'qwen25_15b_chat']:
        report = read(HERE / f'{name}_RESULTS.json')
        for condition, metrics in report['conditions'].items():
            path = HERE / f'{name}_{condition}.jsonl'
            assert sha(path) == metrics['sha256']
            pp = rows(path)
            assert len(pp) == len(data)
            for p, d in zip(pp, data):
                assert all(p[k] == d[k] for k in ['id', 'gold', 'ambiguous_gold'])
                assert len(p['label_logits']) == 4 and all(math.isfinite(x) for x in p['label_logits'])
                assert p['prediction'] == max(range(4), key=lambda i: p['label_logits'][i])
                assert p['correct'] == (p['prediction'] == d['gold'])
                assert p['unrestricted_correct'] == (p['unrestricted_token'] == d['gold'] + 32)
                assert p['valid_label'] == (p['unrestricted_token'] in range(32, 36))
            clean = [p for p in pp if not p['ambiguous_gold']]
            assert len(clean) == metrics['n'] == 998
            for key, field in [('candidate_accuracy', 'correct'), ('unrestricted_accuracy', 'unrestricted_correct'), ('valid_label_rate', 'valid_label')]:
                assert abs(100 * sum(r[field] for r in clean) / len(clean) - metrics[key]) < 1e-10
        target = read(HERE / f'{name}_TARGET_AUDIT.json')
        assert target['standard_shift_check_passed'] and target['abs_loss_error'] < 1e-5
        models[name] = report | {'target_audit': target}
    cfg = read(CHAT / 'models.json')['qwen3_06b_chat']
    tok = AutoTokenizer.from_pretrained(cfg['path'], local_files_only=True)
    generation = []
    for shard in range(4):
        path = HERE / f'thinking_shard{shard}.jsonl'
        done = read(HERE / f'thinking_shard{shard}_DONE.json')
        assert sha(path) == done['sha256']
        pp = rows(path)
        assert len(pp) == done['rows'] == 250
        for p in pp:
            ids = p['generated_ids']
            stop = next((i for i, token in enumerate(ids) if token in done['stop_ids']), None)
            assert stop is None or stop == len(ids) - 1
            content = ids if stop is None else ids[:stop]
            assert tok.decode(content, skip_special_tokens=False) == p['text']
            closes = [i for i, token in enumerate(content) if token == 151668]
            final = tok.decode(content[closes[-1] + 1:], skip_special_tokens=False) if closes else ''
            assert final == p['final_text']
            pred = answer(final) if closes else None
            assert p['thinking_closed'] == bool(closes)
            assert p['prediction'] == pred
            assert p['correct'] == (pred == p['gold'])
            assert p['strict_correct'] == (bool(closes) and final.strip() == 'ABCD'[p['gold']])
            assert p['valid_answer'] == (pred is not None)
            assert p['terminated'] == (stop is not None)
            assert p['hit_length_cap'] == (stop is None and len(ids) >= 2048)
            assert p['generated_tokens'] == len(ids)
        generation += pp
    assert len(generation) == 1000
    assert len({r['id'] for r in generation}) == 1000
    for p, d in zip(generation, data):
        assert all(p[k] == d[k] for k in ['id', 'gold', 'ambiguous_gold'])
    clean = [r for r in generation if not r['ambiguous_gold']]
    summary = {'n': len(clean), 'content_accuracy': 100 * sum(r['correct'] for r in clean) / len(clean),
               'strict_accuracy': 100 * sum(r['strict_correct'] for r in clean) / len(clean),
               'valid_final_answer_rate': 100 * sum(r['valid_answer'] for r in clean) / len(clean),
               'thinking_closed_rate': 100 * sum(r['thinking_closed'] for r in clean) / len(clean),
               'terminated_rate': 100 * sum(r['terminated'] for r in clean) / len(clean),
               'length_cap_rate': 100 * sum(r['hit_length_cap'] for r in clean) / len(clean),
               'median_generated_tokens': statistics.median(r['generated_tokens'] for r in clean),
               'max_generated_tokens': max(r['generated_tokens'] for r in clean),
               'correct_count': sum(r['correct'] for r in clean),
               'capped_ids': [r['id'] for r in clean if r['hit_length_cap']],
               'invalid_final_ids': [r['id'] for r in clean if not r['valid_answer']]}
    contrasts = {
        'qwen3_thinking_vs_chat_candidate': paired(generation, rows(HERE / 'qwen3_06b_chat_chat.jsonl')),
        'qwen3_chat_answer_vs_chat': paired(rows(HERE / 'qwen3_06b_chat_chat_answer.jsonl'), rows(HERE / 'qwen3_06b_chat_chat.jsonl')),
        'qwen25_chat_vs_base_shared_plain': paired(rows(HERE / 'qwen25_15b_chat_plain_answer.jsonl'), rows(HERE / 'qwen25_15b_base_plain_answer.jsonl')),
    }
    report = {'at': datetime.now().astimezone().isoformat(timespec='seconds'), 'status': 'completed_and_verified',
              'scope': 'post hoc diagnosis only; no new fitted models', 'models': models,
              'qwen3_thinking': summary, 'descriptive_paired_contrasts': contrasts,
              'audit': {'source_sealed_files_reverified': count, 'cross_prediction_rows_recomputed': 16000,
                        'thinking_generations_redecoded': 1000, 'standard_sft_shift_checks': 4,
                        'original_reference_prediction_flips': 0}}
    write(HERE / 'SUMMARY.json', report)
    print(__import__('json').dumps({'thinking': summary, 'contrasts': contrasts, 'audit': report['audit']}, indent=2))


if __name__ == '__main__':
    main()
