"""Re-score every saved response, verify public corpus metrics and paired training provenance."""
import math
from transformers import AutoTokenizer
from common import *
from scoring import *

def main():
    for manifest in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/manifest)['files'].items():assert sha(HERE/rel)==h,rel
    toks={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in read(HERE/'models.json').items()}
    checked=0;evals=0;encoded=0
    for task in TASKS:
        for split in ['train','dev','test']:
            data=rows(HERE/f'data/{task}_{split}.jsonl')
            for m,tok in toks.items():
                ts=read(HERE/f'tokens/{task}_{m}_{split}.json');assert len(data)==len(ts)
                for r,z in zip(data,ts):
                    assert r['id']==z['id'] and tok.encode(prompt(r),add_special_tokens=False)==z['prompt_ids']
                    assert tok.encode(r['target'],add_special_tokens=False)+[151643]==z['target_ids'];encoded+=1
    for p in (HERE/'evaluations').glob('*/*/SUMMARY.json'):
        s=read(p);spec=s['spec'];task=spec['task'];rs=rows(p.parent/'responses.jsonl');data=rows(HERE/f"data/{task}_{spec['eval_split']}.jsonl")
        if spec.get('smoke'):data=data[:8]
        assert [r['id'] for r in rs]==[r['id'] for r in data];assert sha(p.parent/'responses.jsonl')==s['responses_sha256']
        for r,gold in zip(rs,data):
            if task=='banking77':
                v=r['label_logprobs'];assert len(v)==77 and all(math.isfinite(x) for x in v)
                pred=f'{max(range(77),key=lambda j:v[j]):02d}';assert pred==r['predicted_code'] and r['gold_code']==gold['target'] and r['content_correct']==(pred==gold['target'])
            else:
                assert toks[spec['model']].decode(r['token_ids'],skip_special_tokens=False)==r['text']
                assert 151643 not in r['token_ids'] and r['length']==len(r['token_ids'])
                assert r['capped']==(not r['native_eos'] and r['length']>=256)
                for k,v in score(r['text'],gold).items():assert r[k]==v,(p,r['id'],k)
            checked+=1
        for k,v in aggregate(rs,task).items():assert abs(v-s[k])<1e-9,(p,k)
        if task=='e2e_clean':
            streams=reference_streams(data);hyp=[r['text'] for r in rs]
            assert abs(BLEU_METRIC.corpus_score(hyp,streams).score-s['primary'])<1e-9
            assert abs(CHRF_METRIC.corpus_score(hyp,streams).score-s['chrfpp'])<1e-9
        else:assert read(p.parent/'CANDIDATE_AUDIT.json')['argmax_equal']
        evals+=1
    for task in TASKS:
        for m in MODELS:
            for seed in range(8100,8105):
                paths=[HERE/'checkpoints'/f'{task}_{m}_{a}_s{seed}' for a in ['hidden','hidden_budget','hidden_both']]
                audits=[read(p/'INITIALIZATION.json') for p in paths]
                assert len({a['shared_hidden_initialization_sha256'] for a in audits})==1
                assert audits[1]['trainable_parameters']==audits[2]['trainable_parameters']
                assert len({sha(p/'TRAIN_ORDER.json') for p in paths})==1
                for p in paths:
                    tr=read(p/'TRAINING.json');assert tr['steps']==64 and tr['examples']==2048 and tr['frozen_before']==tr['frozen_after']
                    assert tr['reload_loss_error']==0 and tr['optimizer_whitelist_verified'] and tr['optimizer_fp32']
                    assert sha(p/'adapter.safetensors')==tr['adapter_sha256']
    write(HERE/'RESULT_AUDIT.json',{'at':now(),'status':'passed','evaluations':evals,'responses':checked,'tokenizations':encoded,'paired_initializations_and_orders':20,'public_corpus_metrics_recomputed':True})
    print('audited',checked,'responses;',encoded,'token records;',evals,'evaluations')
if __name__=='__main__':main()
