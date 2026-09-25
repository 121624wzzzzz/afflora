"""Full response/token/paired-training audit, plus independent official SQL execution."""
from transformers import AutoTokenizer
from common import *
from scoring import score,aggregate
from scoring_sql import parse
from prepare import official_engine

def main():
    for manifest in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/manifest)['files'].items():assert sha(HERE/rel)==h,rel
    toks={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in read(HERE/'models.json').items()}
    checked=evals=encoded=official=0
    for task in TASKS:
        for split in ['train','dev','test']:
            data=rows(HERE/f'data/{task}_{split}.jsonl')
            for m,tok in toks.items():
                ts=read(HERE/f'tokens/{task}_{m}_{split}.json');assert len(data)==len(ts)
                for r,z in zip(data,ts):
                    assert r['id']==z['id'] and tok.encode(prompt(r),add_special_tokens=False)==z['prompt_ids']
                    assert tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]==z['target_ids'];encoded+=1
    engines={s:official_engine(s) for s in ['dev','test']};from lib.query import Query
    for p in sorted((HERE/'evaluations').glob('*/*/SUMMARY.json')):
        s=read(p);spec=s['spec'];task=spec['task'];split=spec['eval_split'];rs=rows(p.parent/'responses.jsonl');data=rows(HERE/f'data/{task}_{split}.jsonl')
        if spec.get('smoke'):data=data[:TASK_SETTINGS[task]['test_batch']]
        assert [r['id'] for r in rs]==[r['id'] for r in data];assert sha(p.parent/'responses.jsonl')==s['responses_sha256']
        for r,gold in zip(rs,data):
            assert toks[spec['model']].decode(r['token_ids'],skip_special_tokens=False)==r['text']
            assert 151643 not in r['token_ids'] and r['length']==len(r['token_ids'])
            assert r['capped']==(not r['native_eos'] and r['length']>=TASK_SETTINGS[task]['max_new_tokens'])
            for k,v in score(r['text'],gold).items():assert r[k]==v,(p,r['id'],k)
            if task=='wikisql' and r['query_valid']:
                q,_=parse(r['text'])
                try:ref=engines[split].execute_query(gold['table_id'],Query.from_dict(q),lower=True)==gold['gold_execution']
                except Exception:ref=False
                assert ref==r['content_correct'];assert r['lf_correct']==(Query.from_dict(q)==Query.from_dict(gold['target']));official+=1
            checked+=1
        got=aggregate(rs,task);assert got=={k:s[k] for k in got},p
        evals+=1
    for task in TASKS:
        for m in MODELS:
            for seed in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+5):
                paths=[HERE/'checkpoints'/f'{task}_{m}_{a}_s{seed}' for a in ['hidden','hidden_budget','hidden_both']];audits=[read(p/'INITIALIZATION.json') for p in paths]
                assert len({a['shared_hidden_initialization_sha256'] for a in audits})==1
                assert audits[1]['trainable_parameters']-audits[2]['trainable_parameters']==read(HERE/'BUDGET_PLAN.json')[m]['budget_excess']
                assert len({sha(p/'TRAIN_ORDER.json') for p in paths})==1
                # Same shuffled example IDs as the verified1.5B anchor at the same task/seed.
                anchor=next(e for e in read(HERE/'ANCHOR_INDEX.json') if e['task']==task and e['arm']=='hidden' and e['seed']==seed)
                assert read(paths[0]/'TRAIN_ORDER.json')==read(HERE/anchor['directory']/'TRAIN_ORDER.json')
                for p in paths:
                    tr=read(p/'TRAINING.json');assert tr['steps']==64 and tr['examples']==2048 and tr['frozen_before']==tr['frozen_after']
                    assert tr['reload_loss_error']==0 and tr['optimizer_whitelist_verified'] and tr['optimizer_fp32']
                    assert sha(p/'adapter.safetensors')==tr['adapter_sha256']
    for task in TASKS:
        for seed in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+5):
            entries=[e for e in read(HERE/'ANCHOR_INDEX.json') if e['task']==task and e['arm']!='base' and e['seed']==seed];assert len(entries)==3
            assert len({read(HERE/e['directory']/'INITIALIZATION.json')['shared_hidden_initialization_sha256'] for e in entries})==1
            assert len({sha(HERE/e['directory']/'TRAIN_ORDER.json') for e in entries})==1
    assert checked==90432 and evals==132 and encoded==13838,(checked,evals,encoded)
    write(HERE/'RESULT_AUDIT.json',{'at':now(),'status':'passed','evaluations':evals,'responses':checked,'tokenizations':encoded,'paired_initializations_and_orders':20,'historical_anchor_pair_groups':10,'official_prediction_executions':official})
    print('audited',checked,'responses;',encoded,'tokens;',evals,'evaluations;',official,'official SQL prediction executions')
if __name__=='__main__':main()
