import sys
sys.dont_write_bytecode=True
from pathlib import Path
from common import HERE, CHAT, now, prompt_ids, read, rows, sha, write

def main():
    assert not (HERE/'manifest.json').exists(),'Already frozen; do not overwrite.'
    assert read(HERE/'smoke_state.json')['phase']=='complete'
    audit=read(HERE/'REUSE_AUDIT.json');smokes={}
    for j in audit['endpoints']:
        if j['seed'] not in [42,None]:continue
        folder=HERE/'smoke'/j['name'];complete=read(folder/'COMPLETE.json')
        assert complete['status']=='passed'
        for p,h in complete['files'].items():assert sha(p)==h,p
        for p,h in complete['identity']['implementation_sha256'].items():assert sha(p)==h,p
        assert complete['identity']['protocol_sha256']==sha(HERE/'DESIGN.md')
        assert read(folder/'REPLAY_AUDIT.json')['status']=='passed'
        smokes[j['name']]=sha(folder/'COMPLETE.json')
    from transformers import AutoTokenizer
    length_audit={}
    for alias,base in read(CHAT/'manifest.json')['models'].items():
        tokenizer=AutoTokenizer.from_pretrained(base['path'],use_fast=True)
        stats={}
        for task in ['cmrc','c3']:
            values=rows(HERE/f'data/{task}_eval.jsonl');max_prompt=0;max_total=0;max_answer=0
            for r in values:
                prompt=prompt_ids(tokenizer,task,r)
                assert prompt and all(isinstance(t,int) for t in prompt)
                answers=r['answers'] if task=='cmrc' else r['choices']
                lengths=[]
                for ans in answers:
                    ids=tokenizer.encode(ans,add_special_tokens=False)
                    assert ids and tokenizer.decode(ids)==ans,(r['id'],'answer roundtrip')
                    lengths.append(len(ids))
                max_prompt=max(max_prompt,len(prompt));max_answer=max(max_answer,max(lengths))
                max_total=max(max_total,len(prompt)+max(max(lengths),256 if task=='cmrc' else 0))
            assert max_total<=8192,(alias,task,max_total)
            stats[task]={'questions':len(values),'max_prompt_tokens':max_prompt,'max_answer_tokens':max_answer,'max_total_tokens':max_total,'truncation_needed':False}
        length_audit[alias]=stats
    write(HERE/'LENGTH_AUDIT.json',length_audit)
    paths=[HERE/n for n in ['common.py','evaluate.py','run.py','prepare.py','fetch_data.py','freeze.py','cmrc_official_py3.py',
        'DESIGN.md','DATA_PROVENANCE.json','DATA_AUDIT.json','REUSE_AUDIT.json','LENGTH_AUDIT.json']]
    paths += [p for d in ['data','vendor'] for p in (HERE/d).rglob('*') if p.is_file()]
    manifest={'created_at':now(),'scope':'Chinese transfer with existing adapters, no task-specific training',
        'endpoints':audit['endpoints'],'local_sha256':{str(p):sha(p) for p in paths},'smoke_complete_sha256':smokes,
        'reuse_audit_sha256':sha(HERE/'REUSE_AUDIT.json'),'versions':audit['versions']}
    write(HERE/'manifest.json',manifest)
    print('FROZEN',now(),len(paths),'local inputs;',len(smokes),'smoke gates passed',flush=True)
    print(length_audit)

if __name__=='__main__':main()
