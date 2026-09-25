"""Seal the completed, reviewed experiment; run only after all writers exit."""
from pathlib import Path
from common import HERE, now, read, sha, write


def main():
    assert not (HERE / 'ARTIFACT_MANIFEST.json').exists(), 'Already sealed'
    for phase in ['PILOT', 'ADAPTER_PILOT', 'FEWSHOT', 'HELDOUT_PROBE', 'HELDOUT', 'ANALYSIS']:
        assert read(HERE / f'{phase}_COMPLETE.json')['status'] == 'passed', phase
    for phase in ['ADAPTER_PILOT', 'HELDOUT_PROBE', 'HELDOUT']:
        s = read(HERE / f'{phase}_STATE.json')
        assert not s['active'] and not s['pending'] and not s['failed'], phase
    verified = {}
    for name in ['CODE_FROZEN.json', 'DATA_FROZEN.json', 'FEWSHOT_FROZEN.json',
                 'ADAPTER_PROTOCOL_FROZEN.json', 'HELDOUT_FROZEN.json']:
        m = read(HERE / name)
        files = m.get('files', {}) | m.get('data_files', {}) | m.get('code_files', {})
        for rel, expected in files.items():
            assert sha(HERE / rel) == expected, (name, rel)
        verified[name] = len(files)
    fewshot_source = sha(HERE / 'fewshot_diagnostic.py')
    assert fewshot_source == read(HERE / 'FEWSHOT_FROZEN.json')['code_sha256']
    model_files = 0
    for config in read(HERE / 'models.json').values():
        for path, expected in config['files'].items():
            assert sha(Path(path)) == expected, path
            model_files += 1
    assert model_files == 14
    dev = read(HERE / 'RESULT_AUDIT.json')
    held = read(HERE / 'HELDOUT_AUDIT.json')
    parameters = read(HERE / 'PARAMETER_AUDIT.json')
    assert dev['status'] == held['status'] == parameters['status'] == 'passed'
    assert dev['responses_verified'] == 22000 and len(dev['training_runs']) == 104
    assert held['total_responses'] == 109442 and held['configurations'] == 106
    assert len(parameters['runs']) == 110 and parameters['token_records_checked'] == 46168
    for name in ['ANALYSIS.json', 'HELDOUT_ANALYSIS.json']:
        assert read(HERE / name)['all_pairing_and_budget_checks'] == 'passed'
    for name in ['README.md', 'FINAL_INTERPRETATION_ZH.md', 'RESULTS_ZH.md',
                 'HELDOUT_RESULTS_ZH.md', 'REPORTING_CORRECTIONS.json',
                 'VISUAL_REVIEW.json', 'HELDOUT_CONTENT_DIAGNOSTIC.json',
                 'figures/additive_effects.png', 'figures/heldout_additive_effects.png']:
        assert (HERE / name).is_file(), name
    visual = read(HERE / 'VISUAL_REVIEW.json')
    assert visual['status'] == 'passed'
    for name, expected in visual['files'].items():
        assert sha(HERE / name) == expected, name
    for name in ['RESULTS_ZH.md', 'HELDOUT_RESULTS_ZH.md']:
        assert '包含无调用情形' not in (HERE / name).read_text()
    workspace = HERE.parents[2]
    docs = ['lora/README.md', 'lora/docs/RESULTS_SO_FAR.md',
            'emnlp/iclr2027/submissions/paper-2/EXPERIMENT_EVIDENCE_STATUS.md']
    write(HERE / 'FINAL_AUDIT.json', {'at': now(), 'status': 'passed',
        'frozen_manifest_file_counts': verified, 'development_responses': 22000,
        'fewshot_diagnostic_source_sha256': fewshot_source,
        'model_files_verified_at_sealing': model_files,
        'heldout_responses': 109442, 'training_runs': 104, 'checkpoint_scopes': 110,
        'workspace_evidence_doc_sha256_at_sealing': {p: sha(workspace / p) for p in docs}})
    write(HERE / 'COMPLETION.json', {'at': now(), 'status': 'passed',
        'scope': 'five-seed, two-task, two-Base fixed-small-training-budget study with held-out evaluation',
        'training_runs_including_smoke': 104, 'development_responses': 22000,
        'heldout_responses': 109442, 'total_responses': 131442,
        'manifest': 'ARTIFACT_MANIFEST.json', 'interpretation': 'FINAL_INTERPRETATION_ZH.md'})
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert not any(p.name.endswith('.tmp') for p in files)
    entries = {str(p.relative_to(HERE)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}
    write(HERE / 'ARTIFACT_MANIFEST.json', {'at': now(), 'status': 'sealed',
        'exclusions': ['ARTIFACT_MANIFEST.json itself', '__pycache__'], 'files': entries})
    # Read-only verification after the seal; no report or audit writer follows.
    manifest = read(HERE / 'ARTIFACT_MANIFEST.json')
    for rel, record in manifest['files'].items():
        p = HERE / rel
        assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], rel
    print('Sealed', len(entries), 'files; manifest SHA256', sha(HERE / 'ARTIFACT_MANIFEST.json'))


if __name__ == '__main__':
    main()
