"""End-to-end integrity, selection rule and actual common-initialization audit."""
import collections
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from common import *
from design import *

def main():
    assert read(HERE/'SCHEDULER_COMPLETE.json')['status']=='passed'
    for f in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/f)['files'].items():assert sha(HERE/rel)==h,(f,rel)
    cfg=read(HERE/'models.json')[MODEL]
    for p,h in cfg['files'].items():assert sha(p)==h,p
    token_checks=0;tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
    for split in ['train','dev','confirm']:
        data=rows(HERE/f'data/wikisql_{split}.jsonl');tokens=read(HERE/f'tokens/wikisql_{MODEL}_{split}.json');assert len(data)==len(tokens)
        for r,q in zip(data,tokens):
            assert r['id']==q['id'] and q['prompt_ids']==tok.encode(prompt(r),add_special_tokens=True)
            assert q['target_ids']==tok.encode(canonical(r['target']),add_special_tokens=False)+[TOKEN_EOS_ID];token_checks+=1
    sel=read(HERE/'SELECTION.json')
    assert sel['code_frozen_sha256']==sha(HERE/'CODE_FROZEN.json') and sel['data_frozen_sha256']==sha(HERE/'DATA_FROZEN.json')
    assert sel['confirmation_jobs_sha256']==sha(HERE/'CONFIRMATION_JOBS.json')
    for rel,h in sel['inputs'].items():assert sha(HERE/rel)==h,rel
    for arm,rs in sel['scores'].items():
        assert sel['selected'][arm]==CANDIDATES[arm][max(range(6),key=lambda i:(rs[i]['mean'],-i))]
        for i,r in enumerate(rs):
            actual=[read(HERE/'evaluations'/spec('search',arm,i,s)['name']/'dev/SUMMARY.json')['primary'] for s in SEARCH_SEEDS]
            assert r['seed_scores']==actual and r['mean']==sum(actual)/2
    state=read(HERE/'STATE.json');assert all(j['state']=='passed' for j in state['jobs'])
    paired=collections.defaultdict(list);responses=official=0;frozen=set()
    for j in state['jobs']:
        s=j['spec'];validate_spec(s);root=HERE/'checkpoints'/s['name'];a=read(HERE/'audits'/f'{s["name"]}.json')
        assert a['status']=='passed';responses+=a['responses'];official+=a['official_sql_executions']
        assert sha(root/'adapter.safetensors')==read(root/'TRAINING.json')['adapter_sha256']
        frozen.add(read(root/'INITIALIZATION.json')['frozen_before'])
        for tag,h in a['summary_sha256'].items():
            p=HERE/'evaluations'/s['name']/tag;assert sha(p/'SUMMARY.json')==h
            assert read(p/'SUMMARY.json')['responses_sha256']==sha(p/'responses.jsonl')
        if s['stage'] in ['search','confirmation']:paired[s['seed']].append(s)
    assert len(frozen)==1
    tensors_checked=0
    for seed,ss in paired.items():
        ref=None;boundary=None;order=None
        for s in ss:
            root=HERE/'checkpoints'/s['name'];d=load_file(str(root/'initial_adapter.safetensors'))
            common={k:(v[:8] if '.lora_A.' in k else v[:,:8]).contiguous() for k,v in d.items() if '.lora_' in k}
            if ref is None:ref=common
            else:
                assert common.keys()==ref.keys() and all(torch.equal(v,ref[k]) for k,v in common.items());tensors_checked+=len(common)
            bd={k:v for k,v in d.items() if k.startswith('boundary_')}
            if bd:
                if boundary is None:boundary=bd
                else:assert bd.keys()==boundary.keys() and all(torch.equal(v,boundary[k]) for k,v in bd.items())
            o=read(root/'TRAIN_ORDER.json')
            if order is None:order=o
            else:assert o==order
    write(HERE/'FINAL_AUDIT.json',{'at':now(),'status':'passed','jobs':len(state['jobs']),'responses':responses,
                                 'official_valid_query_checks':official,'tokens_reencoded':token_checks,
                                 'paired_seeds':len(paired),'actual_shared_tensors_compared':tensors_checked,
                                 'selection_sha256':sha(HERE/'SELECTION.json'),'analysis_sha256':sha(HERE/'ANALYSIS.json')})
    print('final audit passed',flush=True)
if __name__=='__main__':main()
