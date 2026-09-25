import importlib.metadata
import json
import re
import sys
sys.dont_write_bytecode=True
from common import HERE, CHAT, checkpoint_hashes, now, read, rows, sha, write

def normalize(x): return re.sub(r'\s+', ' ', x).strip()

def dump_rows(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values)
    if path.exists(): assert path.read_text()==text
    else:path.write_text(text)

def data():
    provenance=read(HERE/'DATA_PROVENANCE.json')
    for repo in provenance['repositories'].values():
        for p,v in repo['files'].items():assert sha(p)==v['sha256']
    all_rows={}
    for split in ['train','dev']:
        values=[]
        for article in read(HERE/f'vendor/cmrc2018/squad-style-data/cmrc2018_{split}.json')['data']:
            for para in article['paragraphs']:
                for q in para['qas']:
                    assert q['answers'] and all(a['text'] for a in q['answers'])
                    values.append({'id':q['id'],'cluster':para['id'],'context':para['context'],
                        'question':q['question'],'answers':[a['text'] for a in q['answers']]})
        all_rows['cmrc_'+('eval' if split=='dev' else 'train')]=values
    for split in ['train','dev','test']:
        values=[]
        for subset in ['d','m']:
            for i,entry in enumerate(read(HERE/f'vendor/c3/data/c3-{subset}-{split}.json')):
                context='\n'.join(entry[0])
                import hashlib
                cluster=hashlib.sha256(normalize(context).encode()).hexdigest()
                for qi,q in enumerate(entry[1]):
                    assert q['answer'] in q['choice']
                    values.append({'id':f'{subset}:{split}:{i}:{qi}','cluster':cluster,
                        'original_document_id':entry[2],'subset':subset,'context':context,
                        'question':q['question'],'choices':q['choice'],'gold':q['choice'].index(q['answer']),
                        'gold_indices':[k for k,v in enumerate(q['choice']) if v==q['answer']]})
        all_rows['c3_'+('eval' if split=='test' else split)]=values
    for key,values in all_rows.items():
        assert len({r['id'] for r in values})==len(values)
        dump_rows(HERE/'data'/f'{key}.jsonl',values)
    # Mechanical Python 2 -> Python 3 compatibility only; retain upstream metric algorithms.
    raw=(HERE/'vendor/cmrc2018/squad-style-data/cmrc2018_evaluate.py').read_text()
    converted=raw.replace("reload(sys)\nsys.setdefaultencoding('utf8')\n",'')
    converted=converted.replace("str(in_str).decode('utf-8')",'str(in_str)')
    converted=converted.replace("str(prediction_file[query_id]).decode('utf-8')",'str(prediction_file[query_id])')
    converted=converted.replace("ur'[\\u4e00-\\u9fa5]'", "r'[\\u4e00-\\u9fa5]'")
    (HERE/'cmrc_official_py3.py').write_text(converted)
    sft={split:rows(CHAT/f'data/{split}.jsonl') for split in ['train','dev','test']}
    all_questions={normalize(c['content']) for rs in sft.values() for r in rs for c in r['conversations'] if c['role']=='user'}
    report={'created_at':now(),'counts':{},'overlaps':{},'scorer_conversion':[
        'remove Python 2 sys.setdefaultencoding/reload', 'remove UTF-8 decode on Python 3 str', 'ur regex prefix to r'],
        'scope':'No score-based filtering. CMRC public dev, C3 test. Exact overlap checks do not establish pretraining cleanliness.'}
    for key,values in all_rows.items():
        report['counts'][key]={'questions':len(values),'clusters':len({x['cluster'] for x in values})}
        if key.startswith('c3'):report['counts'][key]['duplicate_correct_choices']=[x['id'] for x in values if len(x['gold_indices'])>1]
        report['overlaps'][key+'_sft_exact_question']=[x['id'] for x in values if normalize(x['question']) in all_questions]
    for task in ['cmrc','c3']:
        train={(normalize(x['context']),normalize(x['question'])) for x in all_rows[task+'_train']}
        report['overlaps'][task+'_train_eval_context_question']=[x['id'] for x in all_rows[task+'_eval'] if (normalize(x['context']),normalize(x['question'])) in train]
    write(HERE/'DATA_AUDIT.json',report)

def reuse():
    original=read(CHAT/'manifest.json');audit=read(CHAT/'FINAL_AUDIT.json');state=read(CHAT/'state.json')
    assert audit['status']=='passed' and state['phase']=='complete'
    assert sha(CHAT/'manifest.json')==audit['manifest_sha256']
    exclusions={x['path'] for x in audit['documented_manifest_exclusions']}
    assert exclusions=={str(CHAT/'RESULTS.md')}
    verified={}
    for p,h in original['sha256'].items():
        if p in exclusions:continue
        assert sha(p)==h,p;verified[p]=h
    for n,v in original['versions'].items():assert importlib.metadata.version(n)==v,n
    for model in original['models'].values():
        for p,h in model['files'].items():assert sha(p)==h,p;verified[p]=h
    endpoints=state['matrix']+state['references']
    assert len(endpoints)==26
    for j in endpoints:
        assert state['jobs'][j['name']]['status']=='complete'
        cp=j['checkpoint'];current=checkpoint_hashes(cp)
        if not j.get('reference'):assert current==read(f'{cp}/TRAIN_COMPLETE.json')['checkpoint_hashes']
        assert current==read(CHAT/f"ifeval/{j['name']}/generation_metadata.json")['checkpoint_files']
        for name,h in current.items():verified[str(__import__('pathlib').Path(cp)/name)]=h
        report=CHAT/f"reports/{j['name']}.test.json";verified[str(report)]=sha(report)
        j.update(reuse=True,checkpoint_hashes=current)
    for name in ['manifest.json','FINAL_AUDIT.json','ANALYSIS_SOURCES.json']:verified[str(CHAT/name)]=sha(CHAT/name)
    write(HERE/'REUSE_AUDIT.json',{'status':'hashes_passed_pending_per_endpoint_gpu_replay','checked_at':now(),
        'endpoints':endpoints,'verified_sha256':verified,'documented_prior_exclusions':audit['documented_manifest_exclusions'],
        'versions':original['versions']})
    print('Verified original inputs, complete model SHA-256 and all 26 checkpoint identities.',flush=True)

if __name__=='__main__':
    data();reuse();print(read(HERE/'DATA_AUDIT.json')['counts'])
