from common import *
from scoring import score,aggregate
from transformers import AutoTokenizer

def main():
    assert read(HERE/'HELDOUT_COMPLETE.json')['status']=='passed'
    manifest=read(HERE/'HELDOUT_FROZEN.json')
    for rel,h in dict(manifest['data_files'],**manifest['code_files']).items():assert sha(HERE/rel)==h,rel
    models=read(HERE/'models.json');toks={m:AutoTokenizer.from_pretrained(v['path'],local_files_only=True) for m,v in models.items()}
    refs={};counts={};summaries=[]
    for p in sorted((HERE/'heldout').glob('*/*/SUMMARY.json')):
        s=read(p);spec=s['spec'];split=s['eval_split'];task=spec['task'];model=spec['model']
        data=rows(HERE/f'data/{task}_{split}.jsonl');tokens=read(HERE/f'tokens/{task}_{model}_{split}.json');output=rows(p.parent/'responses.jsonl')
        assert [r['id'] for r in data]==[r['id'] for r in tokens]==[r['id'] for r in output]
        assert len(output)==s['n'];h=htext(canonical([r['prompt_ids'] for r in tokens]));assert s['input_ids_sha256']==h
        assert refs.setdefault((task,model,split),h)==h
        rescored=[]
        for row,r in zip(data,output):
            assert toks[model].decode(r['token_ids'],skip_special_tokens=False)==r['text']
            a=score(r['text'],row)
            for k,v in a.items():assert r[k]==v
            assert r['length']==len(r['token_ids']);assert not(r['native_eos'] and r['capped']);rescored.append(a)
        for k,v in aggregate(rescored,task).items():assert s[k]==v
        assert s['responses_sha256']==sha(p.parent/'responses.jsonl')
        reuse=read(p.parent/'REUSE.json');cp=Path(reuse['checkpoint']);assert sha(cp/'spec.json')==reuse['spec_sha256']
        assert sha(cp/'INITIALIZATION.json')==reuse['initialization_sha256']
        assert read(cp/'INITIALIZATION.json')['frozen_before']==reuse['original_tensor_digest']
        if spec['arm']!='base':assert sha(cp/'adapter.safetensors')==reuse['adapter_sha256']
        counts[str(p.parent.relative_to(HERE))]=len(output);summaries.append(s)
    assert len(counts)==106;assert sum(counts.values())==109442,(len(counts),sum(counts.values()))
    write(HERE/'HELDOUT_AUDIT.json',{'at':now(),'status':'passed','response_counts':counts,'total_responses':sum(counts.values()),'configurations':len(counts),'input_and_reuse_identity_checks':'passed'})
    write(HERE/'HELDOUT_RESULTS.json',{'at':now(),'summaries':summaries});print('passed',sum(counts.values()),'held-out responses')
if __name__=='__main__':main()
