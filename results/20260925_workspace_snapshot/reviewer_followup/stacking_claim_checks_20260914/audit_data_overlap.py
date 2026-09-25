"""Read-only exact-overlap diagnostic; never change the frozen splits."""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'chat' / 'data'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize(value):
    return re.sub(r'\s+', ' ', value).strip()


def main():
    rows = {s: [json.loads(line) for line in (DATA / f'{s}.jsonl').read_text().splitlines()]
            for s in ['train', 'dev', 'test', 'ifeval']}
    signatures = {}
    for split in ['train', 'dev', 'test']:
        signatures[split] = {
            'record_id': [r['record_id'] for r in rows[split]],
            'conversation_exact': [digest(r['conversations']) for r in rows[split]],
            'conversation_whitespace_normalized': [digest([
                {'role': c['role'], 'content': normalize(c['content'])}
                for c in r['conversations']]) for r in rows[split]],
            'user_prompt_whitespace_normalized': [digest(normalize(c['content']))
                for r in rows[split] for c in r['conversations'] if c['role'] == 'user'],
            'first_user_prompt_whitespace_normalized': [digest(normalize(next(
                c['content'] for c in r['conversations'] if c['role'] == 'user')))
                for r in rows[split]],
            'assistant_prediction_context_whitespace_normalized': [digest([
                {'role': t['role'], 'content': normalize(t['content'])}
                for t in r['conversations'][:i]])
                for r in rows[split] for i, c in enumerate(r['conversations']) if c['role'] == 'assistant'],
            'assistant_context_and_target_whitespace_normalized': [digest([
                {'role': t['role'], 'content': normalize(t['content'])}
                for t in r['conversations'][:i+1]])
                for r in rows[split] for i, c in enumerate(r['conversations']) if c['role'] == 'assistant'],
        }
    result = {
        'status': 'complete', 'checked_at': datetime.now().astimezone().isoformat(),
        'scope': 'Exact and whitespace-normalized duplicates only. This does not establish absence of semantic near-duplicates or pretraining contamination. No rows are removed.',
        'split_rows': {s: len(r) for s, r in rows.items()},
        'within_split': {}, 'between_splits': {}, 'ifeval_vs_sft_user_prompts': {},
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve()] + [DATA / f'{s}.jsonl' for s in rows]},
    }
    for split, items in signatures.items():
        result['within_split'][split] = {
            key: {'total': len(values), 'unique': len(set(values)),
                  'duplicate_occurrences_beyond_first': len(values) - len(set(values))}
            for key, values in items.items()}
    for a, b in combinations(signatures, 2):
        result['between_splits'][f'{a}:{b}'] = {}
        for key in signatures[a]:
            ca, cb = Counter(signatures[a][key]), Counter(signatures[b][key])
            shared = set(ca) & set(cb)
            result['between_splits'][f'{a}:{b}'][key] = {
                'shared_unique': len(shared), f'{a}_occurrences': sum(ca[k] for k in shared),
                f'{b}_occurrences': sum(cb[k] for k in shared)}
    for split in signatures:
        prompts = set(signatures[split]['user_prompt_whitespace_normalized'])
        matched = [r['key'] for r in rows['ifeval'] if digest(normalize(r['prompt'])) in prompts]
        result['ifeval_vs_sft_user_prompts'][split] = {'matching_ifeval_keys': matched, 'count': len(matched)}
    result['all_cross_split_conversations_disjoint'] = all(
        metrics[key]['shared_unique'] == 0 for metrics in result['between_splits'].values()
        for key in ['record_id', 'conversation_exact', 'conversation_whitespace_normalized'])
    (HERE / 'DATA_OVERLAP_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
