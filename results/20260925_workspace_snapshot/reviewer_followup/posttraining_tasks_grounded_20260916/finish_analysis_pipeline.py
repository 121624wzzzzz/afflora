"""Run the already specified audits only after each complete experiment phase."""
import os
import subprocess
import time
from common import HERE, PYTHON, now, read, sha, write


def await_phase(filename):
    while not (HERE / filename).exists():
        time.sleep(10)
    assert read(HERE / filename)['status'] == 'passed', filename


def run(filename):
    write(HERE / 'ANALYSIS_STATE.json', {'at': now(), 'running': filename})
    print(now(), filename, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    subprocess.run([PYTHON, '-u', str(HERE / filename)], env=env, check=True)


def main():
    assert not (HERE / 'ANALYSIS_COMPLETE.json').exists()
    await_phase('ADAPTER_PILOT_COMPLETE.json')
    for rel, expected in read(HERE / 'FEWSHOT_FROZEN.json')['files'].items():
        assert sha(HERE / rel) == expected, rel
    run('audit_parameters.py')
    scope = read(HERE / 'PARAMETER_AUDIT.json')
    assert len(scope['runs']) == 110
    assert scope['token_records_checked'] == 46168
    run('audit_results.py')
    audit = read(HERE / 'RESULT_AUDIT.json')
    assert audit['responses_verified'] == 22000
    assert len(audit['training_runs']) == 104
    run('analyze.py')
    run('plot_results.py')
    write(HERE / 'DEVELOPMENT_ANALYSIS_COMPLETE.json', {'at': now(), 'status': 'passed'})
    await_phase('HELDOUT_COMPLETE.json')
    run('audit_heldout.py')
    run('analyze_heldout.py')
    run('plot_heldout.py')
    run('correct_report_scope.py')
    write(HERE / 'ANALYSIS_COMPLETE.json', {'at': now(), 'status': 'passed',
        'development_responses': 22000, 'heldout_responses': 109442,
        'remaining': 'human-readable interpretation, visual inspection, evidence documentation and final sealing'})
    write(HERE / 'ANALYSIS_STATE.json', {'at': now(), 'status': 'passed'})


if __name__ == '__main__':
    main()
