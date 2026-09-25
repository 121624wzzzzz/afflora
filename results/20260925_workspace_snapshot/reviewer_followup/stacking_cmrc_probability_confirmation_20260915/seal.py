"""Seal the complete experiment and preserve every observed result and protocol revision."""
from common import HERE,read,sha,write,now

def main():
    assert read(HERE/'main_state.json')['phase']=='complete'
    audit=read(HERE/'FINAL_AUDIT.json');assert audit['status']=='passed'
    assert read(HERE/'RESULTS.json')['complete']
    assert (HERE/'RUN_HANDOFF.md').read_text().startswith('# COMPLETE')
    for name in ['FINAL_INTERPRETATION_ZH.md','figures/new_seed_confirmation.png','figures/new_seed_confirmation.pdf',
        'GENERATION_PROBABILITY_LINK.json','GENERATION_PROBABILITY_LINK.md']:
        assert (HERE/name).is_file(),name
    for p,h in audit['source_sha256'].items():assert sha(p)==h,p
    plan=read(HERE/'SUPPLEMENTARY_PLAN.json')
    for p,h in plan['script_sha256'].items():assert sha(p)==h,p
    assert plan['main_manifest_sha256']==sha(HERE/'manifest.json')
    for name in ['GENERATION_PROBABILITY_LINK.json','GENERATION_CAP_DIAGNOSTIC.json','INTERIM_QWEN3_PRIMARY.json','INTERIM_QWEN3_SUPPORT.json','INTERIM_ALL_PROBABILITY.json']:
        for p,h in read(HERE/name)['sources_sha256'].items():assert sha(p)==h,p
    results=read(HERE/'RESULTS.json');link=read(HERE/'GENERATION_PROBABILITY_LINK.json')
    for c in link['contrasts']:
        group=c['categories'];assert sum(v['seed_question_pairs'] for v in group.values())==5*3219
        target=lambda metric:next(x['seed_nominal95'] for x in results['contrasts'] if x['model']==c['model'] and x['control']==c['control'] and x['split']=='public' and x['metric']==metric)
        assert abs(sum(v['contribution_to_overall_probability_delta'] for v in group.values())-target('answer_set_probability')['mean'])<1e-12
        assert abs((group['corrected']['seed_question_pairs']-group['regressed']['seed_question_pairs'])/(5*3219)-target('em')['mean'])<1e-12
        for seed in c['per_seed']:
            assert abs(seed['em_delta']-target('em')['per_seed'][str(seed['seed'])])<1e-12
            assert abs(seed['probability_delta']-target('answer_set_probability')['per_seed'][str(seed['seed'])])<1e-12
    artifact=HERE/'ARTIFACT_MANIFEST.json' 
    if artifact.exists():
        for p,h in read(artifact)['sha256'].items():assert sha(p)==h,p
        print('EXISTING SEAL VERIFIED');return
    paths=[p for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.lock','.tmp']]
    hashes={str(p):sha(p) for p in paths}
    write(artifact,{'status':'complete','sealed_at':now(),'file_count':len(hashes),'sha256':hashes,
        'scope':'Entire new confirmation folder except this self-referential manifest, bytecode caches, lock files and temporary files. Includes checkpoints, raw outputs, logs, protocol failure history, reports and figures.',
        'upstream_inputs':'All upstream hashes independently rechecked via FINAL_AUDIT.json; old experiment artifacts were not modified.'})
    for p,h in hashes.items():assert sha(p)==h,p
    print('SEALED',len(hashes),'files',flush=True)

if __name__=='__main__':main()
