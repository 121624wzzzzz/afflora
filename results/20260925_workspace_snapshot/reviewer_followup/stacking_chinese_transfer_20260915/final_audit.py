"""Read-only verification of frozen inputs and recomputation from raw outputs."""
import importlib.metadata
import math
import sys
sys.dont_write_bytecode=True
from common import HERE, CHAT, checkpoint_hashes, now, read, rows, sha, write
from evaluate import task_summary, official

def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
    else:assert a==b,(a,b)

def main():
    assert read(HERE/'main_state.json')['phase']=='complete'
    manifest=read(HERE/'manifest.json');reuse=read(HERE/'REUSE_AUDIT.json')
    verified={**manifest['local_sha256'],**reuse['verified_sha256']}
    for p,h in verified.items():assert sha(p)==h,p
    for n,v in manifest['versions'].items():assert importlib.metadata.version(n)==v
    prior_sources=read(CHAT/'ANALYSIS_SOURCES.json');ce_sources={}
    for j in manifest['endpoints']:
        path=CHAT/f"reports/{j['name']}.test.json";assert sha(path)==prior_sources[str(path)];ce_sources[str(path)]=sha(path)
    cmrc=rows(HERE/'data/cmrc_eval.jsonl');c3=rows(HERE/'data/c3_eval.jsonl')
    raw_cmrc=read(HERE/'vendor/cmrc2018/squad-style-data/cmrc2018_dev.json')
    output_sources={};replay_errors=[]
    for j in manifest['endpoints']:
        folder=HERE/'outputs'/j['name'];complete=read(folder/'COMPLETE.json')
        assert complete['status']=='passed' and not complete['identity']['smoke']
        assert complete['identity']['manifest_sha256']==sha(HERE/'manifest.json')
        assert checkpoint_hashes(j['checkpoint'])==j['checkpoint_hashes']
        for p,h in complete['files'].items():assert sha(p)==h,p;output_sources[p]=h
        replay=read(folder/'REPLAY_AUDIT.json');assert replay['status']=='passed' and replay['examples']==8
        replay_errors.append(replay['max_abs_ce_difference'])
        for task,data in [('cmrc',cmrc),('c3',c3)]:
            saved=read(folder/f'{task}_scores.json');prob=rows(folder/f'{task}_likelihood.jsonl')
            expected=[(r['id'],i) for r in data for i in range(len(r['answers'] if task=='cmrc' else r['choices']))]
            assert [(r['row_id'],r['index']) for r in prob]==expected
            assert all(math.isfinite(x['nll']) and x['nll']>=0 and x['tokens']>0 and 0<=x['top1_correct']<=x['tokens'] for x in prob)
            generated=rows(folder/'cmrc_generation.jsonl') if task=='cmrc' else None
            recomputed=task_summary(task,data,prob,generated)
            for k in recomputed:close(recomputed[k],saved[k])
            if generated:
                assert [x['id'] for x in generated]==[x['id'] for x in cmrc]
                for x in generated:
                    assert 1<=x['generated_tokens']<=256 and x['generated_tokens']==len(x['token_ids'])
                    assert (x['terminal_token'] is None)==x['hit_token_cap']
                    if x['hit_token_cap']:assert x['generated_tokens']==256
                    else:assert x['token_ids'][-1]==x['terminal_token'] and x['terminal_token'] in [151643,151645]
                official_f1,official_em,total,skipped=official.evaluate(raw_cmrc,{x['id']:x['response'] for x in generated})
                assert total==3219 and skipped==0
                close(official_f1/100,saved['metrics']['f1']);close(official_em/100,saved['metrics']['em'])
            print('verified',j['name'],task,flush=True)
    results=read(HERE/'RESULTS.json');assert results['complete'] and results['scored_endpoints_tasks']==52
    for p,h in results['sources_sha256'].items():assert sha(p)==h,p
    generation_replay=read(HERE/'GENERATION_REPLAY_AUDIT.json');assert generation_replay['status']=='passed'
    assert len(generation_replay['checks'])==6
    for name,check in generation_replay['checks'].items():
        assert check['status']=='passed' and check['all_token_ids_text_and_scores_identical']
        assert sha(HERE/'outputs'/name/'cmrc_generation.jsonl')==check['original_output_sha256']
        assert sha(HERE/'generation_replay'/name/'cmrc_generation.jsonl')==check['replay_output_sha256']
    sources={**verified,**output_sources,**ce_sources}
    sources.update({str(HERE/n):sha(HERE/n) for n in ['RESULTS.json','RESULTS.md','paired_effects.csv','analyze.py','final_audit.py','manifest.json','PRIOR_CE_SOURCE_AUDIT.json','GENERATION_REPLAY_AUDIT.json','replay_generation.py','run_replays_when_free.py']})
    write(HERE/'FINAL_AUDIT.json',{'status':'passed','checked_at':now(),'endpoints':26,'task_reports':52,
        'reuse_replay_examples':26*8,'max_replay_ce_error':max(replay_errors),'prior_CE_source_hashes_checked':len(ce_sources),
        'exact_generation_replay_checkpoints':6,'exact_generation_replay_examples':sum(c['examples'] for c in generation_replay['checks'].values()),
        'all_base_weight_sha256_rechecked':True,'prior_manifest_exclusions':reuse['documented_prior_exclusions'],
        'input_and_output_sources_sha256':sources})
    print('FINAL AUDIT PASSED',flush=True)

if __name__=='__main__':main()
