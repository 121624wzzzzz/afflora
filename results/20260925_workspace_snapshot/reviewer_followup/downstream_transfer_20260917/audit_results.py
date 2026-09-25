"""Re-decode and independently check every reported prediction and pairing."""
import numpy as np
from transformers import AutoTokenizer
from common import *
from scoring import score,aggregate,parse,validate
from prepare import official_engine

def main():
    for manifest in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/manifest)['files'].items():assert sha(HERE/rel)==h,rel
    jobs=read(HERE/'FORMAL_JOBS.json');checked=[];n=0;official_checks=0
    tokenizers={m:AutoTokenizer.from_pretrained(read(HERE/'models.json')[m]['path'],local_files_only=True) for m in MODELS}
    engines={s:official_engine(s) for s in ['dev','test']}
    from lib.query import Query
    for spec in jobs:
        cp=HERE/'checkpoints'/spec['name'];assert read(cp/'COMPLETE.json')['status']=='passed'
        assert read(cp/'spec.json')==spec
        initial=read(cp/'INITIALIZATION.json')
        if spec['arm']!='base':
            tr=read(cp/'TRAINING.json');assert tr['steps']==64 and tr['examples']==2048
            assert tr['frozen_before']==tr['frozen_after']==initial['frozen_before']
            assert tr['optimizer_fp32'] and tr['optimizer_whitelist_verified'] and tr['reload_loss_error']==0
            assert tr['adapter_sha256']==sha(cp/'adapter.safetensors')
            ref=HERE/'checkpoints'/f"{spec['task']}_{spec['model']}_hidden_s{spec['seed']}"
            assert initial['shared_hidden_initialization_sha256']==read(ref/'INITIALIZATION.json')['shared_hidden_initialization_sha256']
            assert read(cp/'TRAIN_ORDER.json')==read(ref/'TRAIN_ORDER.json')
            if spec['arm']=='hidden_budget':
                both=HERE/'checkpoints'/f"{spec['task']}_{spec['model']}_hidden_both_s{spec['seed']}"/'INITIALIZATION.json'
                assert initial['trainable_parameters']==read(both)['trainable_parameters']
        for split in ['dev','test']:
            path=HERE/'evaluations'/spec['name']/split;summary=read(path/'SUMMARY.json');records=rows(path/'responses.jsonl')
            data=rows(HERE/f"data/{spec['task']}_{split}.jsonl");tokens=read(HERE/f"tokens/{spec['task']}_{spec['model']}_{split}.json")
            assert [r['id'] for r in records]==[r['id'] for r in data]==[r['id'] for r in tokens]
            assert summary['input_ids_sha256']==htext(canonical([r['prompt_ids'] for r in tokens]))
            assert summary['responses_sha256']==sha(path/'responses.jsonl')
            tok=tokenizers[spec['model']];label_ids=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABC']
            for r,g,tokens_row in zip(records,data,tokens):
                assert tokens_row['prompt_ids']==tok.encode(prompt(g),add_special_tokens=False)
                target=g['target'] if spec['task']=='anli_r1' else canonical(g['target'])
                assert tokens_row['target_ids']==tok.encode(target,add_special_tokens=False)+[151643]
                if spec['task']=='anli_r1':
                    assert all(np.isfinite(r['label_logits']))
                    predicted='ABC'[int(np.argmax(r['label_logits']))]
                    assert r['predicted_label']==predicted
                    assert r['content_correct']==(predicted==g['target'])
                    assert r['unrestricted_correct']==(r['unrestricted_next_token_id']==label_ids['ABC'.index(g['target'])])
                    assert r['unrestricted_valid']==(r['unrestricted_next_token_id'] in label_ids)
                else:
                    assert tok.decode(r['token_ids'],skip_special_tokens=False)==r['text']
                    for k,v in score(r['text'],g).items():assert r[k]==v,(spec['name'],g['id'],k)
                    if r['query_valid']:
                        q,_=parse(r['text']);validate(q,len(g['header']));qp=Query.from_dict(q);qg=Query.from_dict(g['target'])
                        assert r['lf_correct']==(qp==qg)
                        try:ex=engines[split].execute_query(g['table_id'],qp,lower=True)==g['gold_execution']
                        except Exception:ex=False
                        assert r['content_correct']==ex,(spec['name'],g['id'],'official mismatch')
                        official_checks+=1
                    assert r['capped']==(not r['native_eos'] and len(r['token_ids'])>=256)
                n+=1
            for k,v in aggregate(records,spec['task']).items():assert abs(summary[k]-v)<1e-10
            checked.append({'name':spec['name'],'split':split,'responses':len(records),'primary':summary['primary']})
    write(HERE/'RESULT_AUDIT.json',{'at':now(),'status':'passed','responses_checked':n,'official_prediction_execution_checks':official_checks,'runs':checked})
    print('Audit passed',n,'responses;',official_checks,'official prediction checks',flush=True)
if __name__=='__main__':main()
