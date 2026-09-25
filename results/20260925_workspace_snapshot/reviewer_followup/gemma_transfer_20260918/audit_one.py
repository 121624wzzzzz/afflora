"""Independent post-run audit, invoked only after a worker exits successfully."""
import argparse,sys
import math
from transformers import AutoTokenizer
from safetensors import safe_open
from common import *
from scoring import score,aggregate
from prepare import official_engine
from design import validate_spec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args()
    root=HERE/'checkpoints'/args.name;spec=read(root/'spec.json');validate_spec(spec);assert read(root/'COMPLETE.json')['status']=='passed'
    init=read(root/'INITIALIZATION.json');model=spec['model'];arm=spec['arm'];task=spec['task'];budget=read(HERE/'BUDGET_PLAN.json')[model]
    assert init['trainable_parameters']==budget[arm]
    assert init['tied_weights']==read(HERE/'models.json')[model]['tie_word_embeddings']
    assert init['attention_implementation']=='eager' and init['attention_softcap']==50. and init['final_softcap']==30.
    assert init['input_boundary_location']=='raw word embedding, before native sqrt(hidden_size) scaling'
    names=init['trainable_names'];assert all(n.startswith('boundary_') or '.lora_A.' in n or '.lora_B.' in n for n in names)
    assert any('.lora_' in n for n in names)==arm.startswith('hidden')
    for side,allowed in [('input',['input','both','hidden_input','hidden_both']),('output',['output','both','hidden_output','hidden_both'])]:
        assert any(n.startswith('boundary_'+side) for n in names)==(arm in allowed)
    assert init['preflight']['zero_residual_max_abs_error']==0
    extra_rank_checks=0
    if arm!='base':
        tr=read(root/'TRAINING.json');n=64 if spec['smoke'] else 2048;steps=2 if spec['smoke'] else 64
        assert tr['steps']==steps and tr['examples']==n and tr['frozen_before']==tr['frozen_after']
        assert tr['reload_loss_error']==0 and tr['optimizer_whitelist_verified'] and tr['optimizer_fp32']
        assert tr['changed_groups']==tr['present_groups']
        assert len(tr['history'])==steps
        for i,r in enumerate(tr['history']):
            warmup=math.ceil(.03*steps)
            factor=(i+1)/warmup if i<warmup else .5*(1+math.cos(math.pi*(i-warmup)/max(1,steps-warmup)))
            expected={k:spec['lr']*factor*(1. if k=='hidden' else spec['boundary_lr_ratio']) for k,v in tr['present_groups'].items() if v}
            assert r['group_lrs'].keys()==expected.keys()
            assert all(math.isclose(r['group_lrs'][k],v,rel_tol=1e-12,abs_tol=1e-15) for k,v in expected.items())
            assert math.isclose(sum(v*v for v in r['group_grad_norms'].values())**.5,r['grad_norm'],rel_tol=1e-5)
            assert r['joint_clip_factor']==min(1.,1./(r['grad_norm']+1e-6))
            assert all(math.isfinite(v) and v>=0 for v in r['group_update_norms'].values())
        assert sha(root/'adapter.safetensors')==tr['adapter_sha256']
        for file in ['initial_adapter.safetensors','adapter.safetensors']:
            with safe_open(str(root/file),framework='pt',device='cpu') as f:
                assert set(f.keys())==set(names)
                assert sum(f.get_tensor(k).numel() for k in f.keys())==budget[arm]
                assert all(str(f.get_tensor(k).dtype)=='torch.float32' for k in f.keys())
        import torch
        from safetensors.torch import load_file
        from modeling import digest
        initial=load_file(str(root/'initial_adapter.safetensors'));final=load_file(str(root/'adapter.safetensors'))
        assert digest(initial)==init['initialization_sha256'] and digest(final)==tr['adapter_tensor_sha256']
        if arm=='hidden_budget':
            control=init['budget_control'];assert control['actual_extra_parameters']==control['target_extra_parameters']==232960
            assert len(control['rank_pattern'])==33
            for module,rank in control['rank_pattern'].items():
                assert rank==9
                ak=module+'.lora_A.default.weight';bk=module+'.lora_B.default.weight'
                assert initial[ak].shape[0]==final[ak].shape[0]==9 and initial[bk].shape[1]==final[bk].shape[1]==9
                assert torch.count_nonzero(initial[bk][:,8:])==0
                assert torch.count_nonzero(final[bk][:,8:])>0 and not torch.equal(initial[ak][8:],final[ak][8:]),module
                extra_rank_checks+=1
        data=rows(HERE/f'data/{task}_train.jsonl');order=torch.randperm(len(data),generator=torch.Generator().manual_seed(spec['seed'])).tolist()[:n]
        assert read(root/'TRAIN_ORDER.json')=={'indices':order,'ids':[data[i]['id'] for i in order]}
    else:
        tr=read(root/'BASE_EVALUATION.json');assert tr['status']=='passed' and tr['frozen_before']==tr['frozen_after']==init['frozen_before']
    tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[model]['path'],local_files_only=True)
    checked=official=0;summaries={}
    expected_tags=['smoke_dev'] if spec['smoke'] else spec['eval_splits']
    evalroot=HERE/'evaluations'/args.name
    assert (set(p.name for p in evalroot.iterdir()) if evalroot.exists() else set())==set(expected_tags)
    for tag in expected_tags:
        split='dev' if tag=='smoke_dev' else tag;p=HERE/'evaluations'/args.name/tag;s=read(p/'SUMMARY.json');rs=rows(p/'responses.jsonl');data=rows(HERE/f'data/{task}_{split}.jsonl');tokens=read(HERE/f'tokens/{task}_{model}_{split}.json')
        if spec['smoke']:data=data[:TASK_SETTINGS[task]['test_batch']];tokens=tokens[:len(data)]
        assert s['responses_sha256']==sha(p/'responses.jsonl')
        assert s['input_ids_sha256']==htext(canonical([r['prompt_ids'] for r in tokens]))
        assert [r['id'] for r in rs]==[r['id'] for r in data]
        if task=='wikisql':
            source_splits={g['source_split'] for g in data};assert len(source_splits)==1
            engine=official_engine(next(iter(source_splits)));from lib.query import Query
            from scoring_sql import parse
        else:
            check=read(p/'CANDIDATE_AUDIT.json');assert check['argmax_equal'] and check['max_abs_joint_logprob_error']<2e-4
            assert check['scoring_path']=='five native complete-prefix forwards; no cache'
        for r,g in zip(rs,data):
            if task=='trec50':
                assert len(r['label_logprobs'])==50 and all(math.isfinite(v) for v in r['label_logprobs'])
                pred=f'{max(range(50),key=lambda j:r["label_logprobs"][j]):02d}'
                assert r['predicted_code']==pred and r['gold_code']==g['target'] and r['content_correct']==(pred==g['target'])
                checked+=1;continue
            assert tok.decode(r['token_ids'],skip_special_tokens=False)==r['text'] and TOKEN_EOS_ID not in r['token_ids']
            assert r['length']==len(r['token_ids']) and r['capped']==(not r['native_eos'] and r['length']>=TASK_SETTINGS[task]['max_new_tokens'])
            for k,v in score(r['text'],g).items():assert r[k]==v,(args.name,g['id'],k)
            if task=='wikisql' and r['query_valid']:
                q,_=parse(r['text'])
                try:ref=engine.execute_query(g['table_id'],Query.from_dict(q),lower=True)==g['gold_execution']
                except Exception:ref=False
                assert ref==r['content_correct'] and r['lf_correct']==(Query.from_dict(q)==Query.from_dict(g['target']));official+=1
            checked+=1
        got=aggregate(rs,task);assert got=={k:s[k] for k in got};summaries[tag]=sha(p/'SUMMARY.json')
    write(HERE/'audits'/f'{args.name}.json',{'at':now(),'status':'passed','name':args.name,'responses':checked,'official_sql_executions':official,'parameter_scope':budget[arm],'extra_rank_modules_updated':extra_rank_checks,'summary_sha256':summaries})
    print(canonical({'name':args.name,'status':'passed','responses':checked}),flush=True)
if __name__=='__main__':main()
