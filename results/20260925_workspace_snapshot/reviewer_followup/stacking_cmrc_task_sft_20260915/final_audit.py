import importlib.metadata
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common import HERE, checkpoint_hashes, now, read, rows, sha, write
from experiment import MODELS
from run import validate_train
from evaluate import task_summary,official,torch
from safetensors import safe_open

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
    torch.set_num_threads(4)
    assert read(HERE/'main_state.json')['phase']=='complete'
    m=read(HERE/'manifest.json');sources={**m['local_sha256'],**m['external_sha256']}
    for p,h in sources.items():assert sha(p)==h,p
    for n,v in m['versions'].items():assert importlib.metadata.version(n)==v
    data=rows(HERE/'data/cmrc_eval.jsonl');dev=rows(HERE/'data/dev.jsonl')
    transfer=HERE.parent/'stacking_chinese_transfer_20260915';truth=read(transfer/'vendor/cmrc2018/squad-style-data/cmrc2018_dev.json')
    init_groups={};total_supervised=None
    for j in m['matrix']:
        cp=Path(j['checkpoint']);validate_train(j)
        init=read(cp/'initialization_audit.json');init_groups.setdefault((j['model'],j['seed']),{})[j['arm']]=init
        assert init['hidden_rank']==8 and init['base_tied']
        if j['arm']=='hidden_budget':
            cfg=read(cp/'adapter_config.json');assert cfg['rank_pattern']==init['budget_control']['rank_pattern']
            assert all(v==9 for v in cfg['rank_pattern'].values()) and all(v==18 for v in cfg['alpha_pattern'].values())
        for f in ['adapter_model.safetensors','affine_vocab_adapter.safetensors']:
            if not (cp/f).exists():continue
            with safe_open(str(cp/f),framework='pt',device='cpu') as store:
                for key in store.keys():
                    value=store.get_tensor(key);assert value.dtype==torch.float32 and torch.isfinite(value).all(),(j['name'],key)
        for p in cp.iterdir():
            if p.is_file() and p.suffix in ['.json','.safetensors']:sources[str(p)]=sha(p)
        folder=HERE/'outputs'/j['name'];complete=read(folder/'COMPLETE.json')
        assert complete['status']=='passed' and not complete['identity']['job']['smoke']
        assert complete['identity']['manifest_sha256']==sha(HERE/'manifest.json')
        assert complete['identity']['checkpoint_hashes']==checkpoint_hashes(cp)
        for p,h in complete['files'].items():assert sha(p)==h,p;sources[p]=h
        internal=read(folder/'internal_dev_ce.json');per=internal['per_example']
        assert internal['num_examples']==1028 and [r['record_id'] for r in per]==[r['record_id'] for r in dev]
        assert all(x['token_count']>0 and math.isfinite(x['nll_sum']) for x in per)
        close(sum(x['nll_sum'] for x in per)/sum(x['token_count'] for x in per),internal['avg_ce'])
        if total_supervised is None:total_supervised=internal['supervised_tokens']
        else:assert total_supervised==internal['supervised_tokens']
        prob=rows(folder/'cmrc_likelihood.jsonl');gen=rows(folder/'cmrc_generation.jsonl');saved=read(folder/'cmrc_scores.json')
        expected=[(r['id'],i) for r in data for i in range(len(r['answers']))]
        assert [(r['row_id'],r['index']) for r in prob]==expected
        assert [r['id'] for r in gen]==[r['id'] for r in data]
        assert all(x['tokens']>0 and math.isfinite(x['nll']) and x['nll']>=0 and 0<=x['top1_correct']<=x['tokens'] for x in prob)
        for r in gen:
            assert 1<=r['generated_tokens']<=256 and len(r['token_ids'])==r['generated_tokens']
            if r['hit_token_cap']:assert r['terminal_token'] is None and r['generated_tokens']==256
            else:assert r['terminal_token'] in [151643,151645] and r['token_ids'][-1]==r['terminal_token']
        recomputed=task_summary('cmrc',data,prob,gen)
        for k in recomputed:close(recomputed[k],saved[k])
        f1,em,n,skipped=official.evaluate(truth,{r['id']:r['response'] for r in gen})
        assert n==3219 and skipped==0;close(f1/100,saved['metrics']['f1']);close(em/100,saved['metrics']['em'])
        print('verified',j['name'],flush=True)
    for group in init_groups.values():
        assert set(group)=={'none','output','both','hidden_budget'}
        assert len({x['shared_hidden_init_sha256'] for x in group.values()})==1
        assert group['both']['affine_components']['input']==group['both']['affine_components']['output']==group['output']['affine_components']['output']
        assert group['both']['total_trainable']==group['hidden_budget']['total_trainable']
    result=read(HERE/'RESULTS.json');assert result['complete'] and result['trained_endpoints_completed']==24
    for p,h in result['sources_sha256'].items():assert sha(p)==h,p;sources[p]=h
    reference=read(HERE/'REFERENCE_REUSE_AUDIT.json');assert reference['status']=='passed'
    for p,h in reference['source_sha256'].items():assert sha(p)==h,p;sources[p]=h
    for n in ['RESULTS.json','RESULTS.md','paired_effects.csv','analyze.py','final_audit.py','REFERENCE_REUSE_AUDIT.json','manifest.json']:sources[str(HERE/n)]=sha(HERE/n)
    write(HERE/'FINAL_AUDIT.json',{'status':'passed','checked_at':now(),'fresh_training_runs':24,'training_steps_per_run':570,
        'paired_initialization_groups':6,'cmrc_reports':24,'internal_dev_reports':24,'reused_unadapted_references':2,
        'all_trainable_tensors_finite_fp32':True,'all_base_and_input_sha256_rechecked':True,
        'source_sha256':sources})
    print('FINAL AUDIT PASSED',flush=True)

if __name__=='__main__':main()
