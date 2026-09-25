import math
from pathlib import Path
from shared import D,B,read,rows,sha,write,now,jobs
from analyze import summarize

def main():
    assert read(D/'main_state.json')['phase']=='complete'
    source={**read(D/'manifest.json')['code_sha256'],**read(D/'PREPARATION.json')['source_sha256']}
    for p,h in source.items():assert sha(p)==h,p
    result=read(D/'RESULTS.json');assert result['complete'] and result['completed']==24
    rr={(r['name'],r['split']):r for r in result['records']}
    n_public=n_dev=0;max_public=max_dev=max_fp64=0.;tie=0
    data={'internal_dev':rows(B/'data/dev.jsonl'),'public_dev':rows(B/'data/cmrc_eval.jsonl')}
    for j in jobs():
        out=D/'outputs'/j['name'];marker=read(out/'COMPLETE.json');assert marker['status']=='passed'
        cp=Path(j['checkpoint']);assert marker['checkpoint_hashes']==read(cp/'TRAIN_COMPLETE.json')['checkpoint_hashes']
        for f,h in marker['checkpoint_hashes'].items():assert sha(cp/f)==h
        for p,h in marker['files'].items():assert sha(p)==h,p;source[p]=h
        source[str(out/'COMPLETE.json')]=sha(out/'COMPLETE.json')
        for split in data:
            raw=rows(out/f'{split}.jsonl')
            old=rows(B/'outputs'/j['name']/'cmrc_likelihood.jsonl') if split=='public_dev' else read(B/'outputs'/j['name']/'internal_dev_ce.json')['per_example']
            assert len(raw)==len(old)==(9657 if split=='public_dev' else 1028)
            assert len({(r['row_id'],r['index']) for r in raw})==len(raw)
            for r,o in zip(raw,old):
                assert abs(r['nll']-r['first_nll']-r['rest_nll'])<1e-12
                assert r['top1_correct']==r['first_top1']+r['rest_top1_correct']
                assert 0<=r['top1_correct']<=r['top5_correct']<=r['tokens']
                assert r['eos_nll']>=0 and r['first_nll']>=0 and r['rest_nll']>=0
                assert r['fp64_check_max']<5e-6;max_fp64=max(max_fp64,r['fp64_check_max'])
                assert r['teacher_exact']==(r['top1_correct']==r['tokens'] and r['eos_top1'])
                tie+=r['argmax_vs_topk1_different']
                if split=='public_dev':
                    assert (r['row_id'],r['index'],r['tokens'],r['top1_correct'])==(o['row_id'],o['index'],o['tokens'],o['top1_correct'])
                    err=abs(r['nll']-o['nll']);max_public=max(max_public,err);assert err<1e-8;n_public+=1
                else:
                    assert r['row_id']==o['record_id'] and r['tokens']+1==o['token_count']
                    err=abs(r['nll']+r['eos_nll']-o['nll_sum']);max_dev=max(max_dev,err);assert err<1e-4;n_dev+=1
                assert abs(r['old_nll_absolute_error']-err)<1e-12
            metrics,_=summarize(raw,split,data[split])
            for k,v in metrics.items():assert math.isclose(v,rr[j['name'],split][k],rel_tol=0,abs_tol=1e-12),(j['name'],split,k)
        print('verified',j['name'],flush=True)
    for p,h in result['source_sha256'].items():assert sha(p)==h
    for n in ['manifest.json','RESULTS.json','RESULTS.md','contrasts.csv','analyze.py','final_audit.py']:source[str(D/n)]=sha(D/n)
    write(D/'FINAL_AUDIT.json',{'status':'passed','checked_at':now(),'checkpoints':24,'public_reference_replays':n_public,
        'internal_example_replays':n_dev,'public_max_nll_error':max_public,'internal_max_nll_error':max_dev,'fp64_max_error':max_fp64,
        'argmax_vs_topk1_different_total':tie,'upstream_artifacts_unchanged':True,'source_sha256':source})
    print('FINAL AUDIT PASSED',flush=True)

if __name__=='__main__':main()
