"""Re-decode saved output IDs, rescore all responses, and verify frozen inputs."""
from common import *
from scoring import score,aggregate
from transformers import AutoTokenizer

def main():
    checks=[]
    for manifest in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/manifest)['files'].items():assert sha(HERE/rel)==h,rel
        checks.append({'manifest':manifest,'status':'passed'})
    models=read(HERE/'models.json');tokenizers={m:AutoTokenizer.from_pretrained(v['path'],local_files_only=True) for m,v in models.items()}
    counts={};reference_inputs={};summaries=[]
    for p in sorted((HERE/'evaluations').glob('*/*/SUMMARY.json')):
        saved=read(p);spec=saved['spec'];task=spec['task'];split=spec.get('eval_split','pilot_dev')
        data=rows(HERE/f'data/{task}_{split}.jsonl');tokens=read(HERE/f"tokens/{task}_{spec['model']}_{split}.json")
        output=rows(p.parent/'responses.jsonl');assert len(output)==len(data)==saved['n']
        assert [r['id'] for r in data]==[r['id'] for r in output]==[r['id'] for r in tokens]
        input_hash=htext(canonical([r['prompt_ids'] for r in tokens]));assert input_hash==saved['input_ids_sha256']
        key=(task,spec['model'],split);assert reference_inputs.setdefault(key,input_hash)==input_hash
        rescored=[]
        for row,record in zip(data,output):
            assert tokenizers[spec['model']].decode(record['token_ids'],skip_special_tokens=False)==record['text']
            result=score(record['text'],row)
            for k,v in result.items():assert record[k]==v,(p,k)
            assert record['length']==len(record['token_ids'])
            assert not(record['native_eos'] and record['capped'])
            rescored.append(result)
        result=aggregate(rescored,task)
        for k,v in result.items():assert saved[k]==v,(p,k)
        assert sha(p.parent/'responses.jsonl')==saved['responses_sha256']
        counts[str(p.parent.relative_to(HERE))]=len(output);summaries.append(saved)
    training=[]
    for p in sorted((HERE/'checkpoints').glob('*/TRAINING.json')):
        t=read(p);a=read(p.parent/'INITIALIZATION.json');assert t['frozen_before']==t['frozen_after']==a['frozen_before']
        assert t['optimizer_fp32'] and t['optimizer_whitelist_verified'] and t['reload_loss_error']==0
        assert t['adapter_sha256']==sha(p.parent/'adapter.safetensors');training.append(p.parent.name)
    write(HERE/'RESULT_AUDIT.json',{'at':now(),'status':'passed','source_checks':checks,'response_counts':counts,'responses_verified':sum(counts.values()),'training_runs':training})
    write(HERE/'RESULTS.json',{'at':now(),'summaries':summaries})
    print('passed',sum(counts.values()),'responses',len(training),'training runs')
if __name__=='__main__':main()
