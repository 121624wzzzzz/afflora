"""Independent reconstruction from per-reference losses and official generation outputs."""
import importlib.metadata
import math
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common import HERE,CHAT,checkpoint_hashes,now,read,rows,sha,write
from experiment import MODELS,SEEDS,ARMS,jobs
from run import validate_train
sys.path.insert(0,str(CHAT/'ifeval_deps'));os.environ['NLTK_DATA']=str(CHAT/'nltk_data')
import cmrc_official_py3 as official
import numpy as np
from scipy.stats import t
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

def close(a,b,tol=2e-12):
    assert math.isclose(a,b,rel_tol=0,abs_tol=tol),(a,b)

def check_prob(data,cache):
    assert len(data)==len(cache)
    for row,truth in zip(data,cache):
        assert row['id']==truth['id'] and row['cluster']==truth['cluster']
        assert len(row['unique_answers'])==len(truth['unique_answers'])
        mapped={};total_prob=[]
        for value,answer in zip(row['unique_answers'],truth['unique_answers']):
            assert value['answer']==answer['answer'] and value['reference_indices']==answer['reference_indices']
            assert value['tokens']==len(answer['answer_ids'])>0
            for k in ['content_nll','eos_nll','first_nll','rest_nll']:assert math.isfinite(value[k]) and value[k]>=0
            close(value['first_nll']+value['rest_nll'],value['content_nll'])
            assert value['content_top1']==value['first_top1']+value['rest_top1']
            assert 0<=value['content_top1']<=value['tokens']
            p=math.exp(-(value['content_nll']+value['eos_nll']));close(p,value['probability']);total_prob.append(p)
            for index in answer['reference_indices']:
                assert index not in mapped and truth['references'][index]==value['answer'];mapped[index]=value
        assert sorted(mapped)==list(range(len(truth['references'])))
        mass=math.fsum(total_prob);assert 0<mass<=1+1e-6
        close(row['answer_set_probability'],mass);close(row['answer_set_nll'],-math.log(mass),1e-11)
        values=list(mapped.values());assert row['reference_count']==len(values)
        for k,target in [('content_tokens','tokens'),('content_nll','content_nll'),('eos_nll','eos_nll'),('content_top1','content_top1'),
            ('first_nll','first_nll'),('rest_nll','rest_nll'),('first_top1','first_top1'),('rest_top1','rest_top1')]:close(row[k],math.fsum(v[target] for v in values))
        close(row['content_macro_ce'],math.fsum(v['content_nll']/v['tokens'] for v in values)/len(values))

def independent_summary(prob,gen=None):
    n=len(prob);s=lambda key:math.fsum(x[key] for x in prob)
    tokens=s('content_tokens');refs=s('reference_count')
    m={k:s(k)/n for k in ['answer_set_probability','answer_set_nll','content_macro_ce']}
    m.update(content_micro_ce=s('content_nll')/tokens,total_micro_ce=(s('content_nll')+s('eos_nll'))/(tokens+refs),
        eos_ce=s('eos_nll')/refs,content_top1=s('content_top1')/tokens,first_ce=s('first_nll')/refs,
        first_top1=s('first_top1')/refs,rest_ce=s('rest_nll')/(tokens-refs),rest_top1=s('rest_top1')/(tokens-refs))
    if gen:m.update({k:math.fsum(x[k] for x in gen)/n for k in ['em','f1','avg','generated_tokens','hit_token_cap']})
    return m

def main():
    torch.set_num_threads(4);assert read(HERE/'main_state.json')['phase']=='complete'
    manifest=read(HERE/'manifest.json');sources={**manifest['local_sha256'],**manifest['external_sha256']}
    for p,h in sources.items():assert sha(p)==h,p
    for name,version in manifest['versions'].items():assert importlib.metadata.version(name)==version
    truth=read(HERE.parent/'stacking_chinese_transfer_20260915/vendor/cmrc2018/squad-style-data/cmrc2018_dev.json')
    cache={m:{split:rows(HERE/'token_cache'/m/(split+'.jsonl')) for split in ['internal_dev','public_dev']} for m in MODELS}
    tokenizers={m:AutoTokenizer.from_pretrained(p,local_files_only=True) for m,p in MODELS.items()}
    for v in cache.values():
        assert len(v['internal_dev'])==1028 and len(v['public_dev'])==3219
        assert sum(len(x['unique_answers']) for x in v['public_dev'])==4188
        assert sum(len(x['references']) for x in v['public_dev'])==9657
    groups={};endpoints={};max_nll=0.;max_prob=0.;max_fp64=0.;max_full=0.;caps=0
    for j in jobs():
        cp=Path(j['checkpoint']);validate_train(j)
        init=read(cp/'initialization_audit.json');groups.setdefault((j['model'],j['seed']),{})[j['arm']]=init
        assert init['hidden_rank']==8 and init['base_tied']
        if j['arm']=='hidden_budget':
            config=read(cp/'adapter_config.json');assert config['rank_pattern']==init['budget_control']['rank_pattern']
            assert all(v==9 for v in config['rank_pattern'].values()) and all(v==18 for v in config['alpha_pattern'].values())
        for f in ['adapter_model.safetensors','affine_vocab_adapter.safetensors']:
            if not (cp/f).exists():continue
            with safe_open(str(cp/f),framework='pt',device='cpu') as store:
                for key in store.keys():
                    value=store.get_tensor(key);assert value.dtype==torch.float32 and torch.isfinite(value).all(),(j['name'],key)
        for p in cp.iterdir():
            if p.is_file() and p.suffix in ['.json','.safetensors']:sources[str(p)]=sha(p)
        folder=HERE/'outputs'/j['name'];complete=read(folder/'COMPLETE.json')
        assert complete['status']=='passed' and complete['job']==j and complete['manifest_sha256']==sha(HERE/'manifest.json')
        assert complete['checkpoint_hashes']==checkpoint_hashes(cp)
        for p,h in complete['files'].items():assert sha(p)==h,p;sources[p]=h
        sources[str(folder/'COMPLETE.json')]=sha(folder/'COMPLETE.json')
        numeric=read(folder/'NUMERICAL_AUDIT.json');assert numeric['status']=='passed' and len(numeric['checks'])==8
        for r in numeric['checks']:
            assert r['canonical_replay_exact']
            assert r['token_loss_max_diff']<=5e-4 and r['probability_max_diff']<=1e-4 and r['fp64_max_diff']<5e-6
            assert r['full_logits_loss_max_diff']<=2e-4 and r['full_logits_probability_diff']<=1e-5
            max_nll=max(max_nll,r['token_loss_max_diff']);max_prob=max(max_prob,r['probability_max_diff'])
            max_fp64=max(max_fp64,r['fp64_max_diff']);max_full=max(max_full,r['full_logits_loss_max_diff'])
        dev=rows(folder/'internal_probability.jsonl');prob=rows(folder/'public_probability.jsonl');gen=rows(folder/'generation.jsonl')
        check_prob(dev,cache[j['model']]['internal_dev']);check_prob(prob,cache[j['model']]['public_dev'])
        assert [r['id'] for r in gen]==[r['id'] for r in prob]
        tokenizer=tokenizers[j['model']]
        for r,reference in zip(gen,cache[j['model']]['public_dev']):
            assert r['cluster']==reference['cluster'] and len(r['token_ids'])==r['generated_tokens'] and 1<=r['generated_tokens']<=256
            assert tokenizer.decode(r['token_ids'],skip_special_tokens=True)==r['response']
            if r['hit_token_cap']:assert r['terminal_token'] is None and r['generated_tokens']==256;caps+=1
            else:assert r['terminal_token'] in [151643,151645] and r['token_ids'][-1]==r['terminal_token']
            close(r['em'],official.calc_em_score(reference['references'],r['response']))
            close(r['f1'],official.calc_f1_score(reference['references'],r['response']));close(r['avg'],(r['em']+r['f1'])/2)
        saved=read(folder/'scores.json');assert saved['job']==j and saved['precision']=='FP32 eager TF32 disabled'
        endpoint={}
        for split,data,generation in [('public',prob,gen),('internal',dev,None)]:
            rebuilt=independent_summary(data,generation);assert set(rebuilt)==set(saved[split]['metrics'])
            for key,value in rebuilt.items():close(value,saved[split]['metrics'][key])
            endpoint[split]=rebuilt
        endpoints[(j['model'],j['seed'],j['arm'])]=endpoint
        f1,em,n,skipped=official.evaluate(truth,{r['id']:r['response'] for r in gen})
        assert n==3219 and skipped==0;close(f1/100,endpoint['public']['f1']);close(em/100,endpoint['public']['em'])
        print('verified',j['name'],flush=True)
    assert len(groups)==10
    for group in groups.values():
        assert set(group)==set(ARMS) and len({x['shared_hidden_init_sha256'] for x in group.values()})==1
        assert group['both']['affine_components']['input']==group['both']['affine_components']['output']
        assert group['both']['total_trainable']==group['hidden_budget']['total_trainable']
    result=read(HERE/'RESULTS.json');assert result['complete'] and result['fresh_training_runs']==30 and result['seeds']==SEEDS
    for r in result['contrasts']:
        delta=np.array([endpoints[(r['model'],s,'both')][r['split']][r['metric']]-endpoints[(r['model'],s,r['control'])][r['split']][r['metric']] for s in SEEDS])
        for name,family in [('seed_nominal95',1)]+([('seed_bonferroni_family4',4)] if r['primary'] else []):
            v=r[name];mean=float(delta.mean());error=float(delta.std(ddof=1)/math.sqrt(5)*t.ppf(1-.05/(2*family),4))
            close(mean,v['mean']);close(mean-error,v['ci'][0]);close(mean+error,v['ci'][1])
            for seed,value in zip(SEEDS,delta):close(float(value),v['per_seed'][str(seed)])
        if r['primary']:assert r['confirmed']==(r['seed_bonferroni_family4']['ci'][0]>0)
    assert result['joint_two_model_confirmation']==all(x['confirmed'] for x in result['primary_contrasts'])
    for p,h in result['sources_sha256'].items():assert sha(p)==h,p;sources[p]=h
    for n in ['RESULTS.json','RESULTS.md','paired_effects.csv','analyze.py','final_audit.py','manifest.json']:sources[str(HERE/n)]=sha(HERE/n)
    write(HERE/'FINAL_AUDIT.json',{'status':'passed','checked_at':now(),'fresh_training_runs':30,'steps_per_run':570,
        'paired_initialization_groups':10,'public_question_evaluations':30*3219,'unique_public_reference_evaluations':30*4188,
        'internal_question_evaluations':30*1028,'numerical_canonical_exact_replays':240,
        'max_shape_token_nll_difference':max_nll,'max_shape_probability_difference':max_prob,
        'max_fp32_ce_vs_fp64_difference':max_fp64,'max_full_logits_token_nll_difference':max_full,
        'generation_cap_hits':caps,'all_trainable_tensors_finite_fp32':True,'all_frozen_source_hashes_rechecked':True,
        'reference_mass_independently_recomputed':True,'official_generation_scores_recomputed':True,
        'all_seed_effects_and_intervals_independently_recomputed':True,'source_sha256':sources})
    print('FINAL AUDIT PASSED',flush=True)

if __name__=='__main__':main()
