"""Post-hoc descriptive breakdown; all groups retained, primary results unchanged."""
from collections import defaultdict
from statistics import mean
from common import *


def main():
    assert read(HERE / 'HELDOUT_AUDIT.json')['status'] == 'passed'
    groups = defaultdict(list)
    sources = {}
    for path in sorted((HERE / 'heldout').glob('*/*/SUMMARY.json')):
        s = read(path)
        spec = s['spec']
        groups[(spec['task'], spec['model'], spec['arm'], s['eval_split'])].append(s)
        sources[str(path.relative_to(HERE))] = sha(path)
    table = []
    for (task, model, arm, split), summaries in sorted(groups.items()):
        record = {'task': task, 'model': model, 'arm': arm, 'eval_split': split,
                  'configurations': len(summaries)}
        for metric in ['primary', 'function_correct', 'json_valid', 'schema_valid',
                       'native_eos_pct', 'capped_pct', 'mean_generated_tokens']:
            if metric in summaries[0]:
                record[metric] = mean(s[metric] for s in summaries)
        if task == 'cluener':
            counts = [s['span_counts'] for s in summaries]
            record['span_precision'] = mean(100 * c['tp'] / c['pred'] if c['pred'] else 0 for c in counts)
            record['span_recall'] = mean(100 * c['tp'] / c['gold'] if c['gold'] else 0 for c in counts)
            record['mean_span_counts'] = {k: mean(c[k] for c in counts) for k in ['tp', 'pred', 'gold']}
        table.append(record)
    assert len(sources) == 106 and len(table) == 26
    write(HERE / 'HELDOUT_CONTENT_DIAGNOSTIC.json', {'at': now(),
        'scope': 'post-hoc descriptive seed-mean precision/recall, function-name accuracy and output behavior',
        'warning': 'These auxiliary metrics do not replace the frozen main contrasts or identify a causal mechanism.',
        'source_summary_sha256': sources, 'table': table})
    for r in table:
        if r['arm'] in ['hidden', 'hidden_both', 'hidden_budget']:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    main()
