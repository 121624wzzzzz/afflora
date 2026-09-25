"""Correct a inherited report sentence; never change scores or frozen sources."""
from common import HERE, now, read, sha, write


def main():
    old = 'ToolACE 主分为完整调用集合正确率，包含无调用情形；不是 BFCL 官方得分。'
    new = ('ToolACE 主分为完整调用集合正确率；本轮仅含参数值可从请求中直接找到的正调用样本，'
           '不含无调用/拒绝调用情形，也不是 BFCL 官方得分。')
    log = HERE / 'REPORTING_CORRECTIONS.json'
    records = read(log)['files'] if log.exists() else []
    for name in ['RESULTS_ZH.md', 'HELDOUT_RESULTS_ZH.md']:
        path = HERE / name
        if not path.exists():
            continue
        text = path.read_text()
        previous = [r for r in records if r['file'] == name]
        if previous:
            assert len(previous) == 1 and sha(path) == previous[0]['after_sha256'], name
            assert old not in text and text.count(new) == 1, name
            continue
        assert text.count(old) == 1, name
        before = sha(path)
        path.write_text(text.replace(old, new))
        records.append({'file': name, 'before_sha256': before, 'after_sha256': sha(path)})
    assert records, 'No completed report to correct'
    write(log, {
        'at': now(), 'scope': 'report wording only; all metrics and frozen analysis sources unchanged',
        'reason': 'The report template inherited an all-call-types description from V1; V2 excludes no-call rows.',
        'old': old, 'new': new, 'files': records,
    })


if __name__ == '__main__':
    main()
