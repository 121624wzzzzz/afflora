"""Complete the original audit with one documented derived-report exclusion.

The original manifest and experiment source remain unchanged. RESULTS.md was
mistakenly frozen after the initial smoke summary and is rewritten by the frozen
summarizer. Every other frozen entry, package version and model file is checked.
"""
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CHAT=HERE/'chat'
HISTORY=HERE/'report_manifest_audit'
sys.path.insert(0,str(CHAT))
import run
import support
import summarize


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    lock=(CHAT/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=CHAT/'manifest.json';m=json.loads(manifest_path.read_text())
    assert manifest_path.read_bytes()==(HISTORY/'discovery_manifest.json').read_bytes()
    excluded=CHAT/'RESULTS.md'
    failure_line='AssertionError: '+str(excluded)
    assert failure_line in (CHAT/'scheduler.log').read_text(), 'Preserve and verify the original scheduler failure before finalizing.'
    original=HISTORY/'reconstruction_smoke/RESULTS.md'
    assert sha(original)==m['sha256'][str(excluded)]
    for path,expected in m['sha256'].items():
        if path==str(excluded):continue
        assert sha(path)==expected,path
    for name,version in m['versions'].items():
        assert importlib.metadata.version(name)==version,name
    model_checks={}
    for model in m['models'].values():
        for path,expected in model['files'].items():
            actual=sha(path);assert actual==expected,path
            model_checks[path]=actual
    state=json.loads((CHAT/'state.json').read_text())
    assert state['matrix']==m['matrix']==run.jobs()
    assert state['references']==m['references']==run.references()
    support.STATE=state;run.STATE=state
    cases=run.references()+run.jobs()
    for job in cases:
        assert state['jobs'][job['name']]['status']=='complete',job['name']
        if not job.get('reference'):
            run.validate_train(job)
            marker=json.loads((Path(job['checkpoint'])/'TRAIN_COMPLETE.json').read_text())
            assert marker['checkpoint_hashes']==run.checkpoint_hashes(Path(job['checkpoint']))
        for split in ['dev','test']:run.validate_ce(job,split)
        run.validate_generation(job)
    summary=summarize.summarize()
    assert summary['completed']==24 and not summary['audit_errors']
    # Preserve the original scheduler's final summary and failure log before
    # recording the separately executed, corrected audit.
    for name in ['state.json','summary.json','RESULTS.md','scheduler.log']:
        destination=HISTORY/('after_original_scheduler_'+name)
        if not destination.exists():destination.write_bytes((CHAT/name).read_bytes())
    exception={
        'path':str(excluded), 'kind':'derived_report_mistakenly_frozen',
        'expected_initial_sha256':m['sha256'][str(excluded)],
        'reconstructed_original':str(original),
        'explanation':'The frozen summarizer writes this file after each completed cell; training and evaluation never read it. Only this derived output is excluded from the immutable-input check. The original manifest, source, data, model files, checkpoints and scores are unchanged.',
        'documentation':str(HERE/'REPORT_MANIFEST_NOTE.md'),
    }
    state['phase']='complete'
    state['final_audit_implementation']=str(Path(__file__).resolve())
    support.write_json(CHAT/'state.json',state)
    summarize.summarize()
    exception['final_report_sha256']=sha(excluded)
    audit={'status':'passed','checked_at':support.now(),
           'new_training_cells':24,'reference_cells':2,'ce_reports':52,'ifeval_reports':26,
           'audit_errors':[], 'original_scheduler_audit':'failed_on_derived_RESULTS_md_hash',
           'documented_manifest_exclusions':[exception],
           'frozen_input_files_checked':len(m['sha256'])-1,
           'model_file_sha256_rechecked':model_checks,
           'manifest_sha256':sha(manifest_path),
           'audit_implementation':str(Path(__file__).resolve()),
           'audit_implementation_sha256':sha(Path(__file__).resolve())}
    support.write_json(CHAT/'FINAL_AUDIT.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='model_file_sha256_rechecked'},indent=2))


if __name__=='__main__':main()
