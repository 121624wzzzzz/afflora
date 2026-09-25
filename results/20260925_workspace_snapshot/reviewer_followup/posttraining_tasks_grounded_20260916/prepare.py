import re,collections
from common import *
from scoring import score
V1=HERE.parent/'posttraining_tasks_20260916'

def leaves(x):
    if isinstance(x,dict):return [v for y in x.values() for v in leaves(y)]
    if isinstance(x,list):return [v for y in x for v in leaves(y)]
    return [x]
def grounded(r):
    if not r['target']:return False
    text=r['text'].casefold()
    for v in leaves([c['arguments'] for c in r['target']]):
        if type(v)==str:
            if not v or v.casefold() not in text:return False
        elif type(v) in [int,float]:
            if not re.search(r'(?<![\w.])'+re.escape(str(v))+r'(?![\w.])',text):return False
        elif type(v)==bool:
            if str(v).casefold() not in text:return False
        else:return False
    return True

def components(data):
    parent=list(range(len(data)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(data):
        keys=['query:'+re.sub(r'\s+',' ',r['text']).strip().casefold()]
        for t in r['tools']:
            keys.append('name:'+re.sub(r'[^\w]','',t['name']).casefold());keys.append('signature:'+htext(canonical(t)))
        for key in keys:
            if key in seen:parent[root(i)]=root(seen[key])
            else:seen[key]=i
    groups=collections.defaultdict(list)
    for i,r in enumerate(data):groups[root(i)].append(r)
    return sorted(groups.values(),key=lambda g:(-len(g),htext(g[0]['id'])))

def main():
    from transformers import AutoTokenizer
    assert not (HERE/'DATA_FROZEN.json').exists()
    # Verify every reused source row file against the V1 pre-evaluation manifest.
    old=read(V1/'DATA_FROZEN.json');models=read(HERE/'models.json');verified=[]
    for task in TASKS:
        for split in ['train','dev','test']:
            rel=f'data/{task}_{split}.jsonl';assert sha(V1/rel)==old['files'][rel];verified.append({'file':str(V1/rel),'sha256':sha(V1/rel)})
    for m,v in models.items():
        for path,h in v['files'].items():assert sha(path)==h;verified.append({'file':path,'sha256':h})
    write(HERE/'REUSE_CHECK.json',{'at':now(),'status':'passed','verified_files':verified,'source_manifest_sha256':sha(V1/'DATA_FROZEN.json'),'fitted_adapters_reused':False})
    tokenizers={m:AutoTokenizer.from_pretrained(models[m]['path'],local_files_only=True) for m in MODELS};metadata={};exclusions=[]
    for task in TASKS:
        metadata[task]={}
        for split in ['train','dev','test']:
            original=rows(V1/f'data/{task}_{split}.jsonl');data=[]
            for r in original:
                if task=='toolace':
                    if not grounded(r):exclusions.append({'id':r['id'],'split':split,'reason':'no_call_or_non_literal_argument'});continue
                else:
                    r['gold_spans']=r['target'];r['target']=[]
                    for e in r['gold_spans']:
                        positions=[i for i in range(len(r['text'])) if r['text'].startswith(e['text'],i)]
                        occurrence=positions.index(e['start']);assert positions[occurrence]+len(e['text'])-1==e['end']
                        r['target'].append({'type':e['type'],'text':e['text'],'occurrence':occurrence})
                assert score(canonical(r['target']),r)['content_correct'];data.append(r)
            write_rows(HERE/f'data/{task}_{split}.jsonl',data)
            metadata[task][split]={'source_n':len(original),'retained_n':len(data),'excluded_n':len(original)-len(data)}
            for m,tok in tokenizers.items():
                tokens=[{'id':r['id'],'prompt_ids':tok.encode(prompt(r),add_special_tokens=False),
                    'target_ids':tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]} for r in data]
                assert all(len(r['prompt_ids'])<=1536 for r in tokens)
                if split!='test':assert all(len(r['target_ids'])<=512 for r in tokens)
                write(HERE/f'tokens/{task}_{m}_{split}.json',tokens)
                if split!='test':
                    n=2048 if split=='train' else 200;assert len(data)>=n
                    write_rows(HERE/f'data/{task}_pilot_{split}.jsonl',data[:n]);write(HERE/f'tokens/{task}_{m}_pilot_{split}.json',tokens[:n])
    write(HERE/'DATA_AUDIT.json',{'at':now(),'splits':metadata,'exclusions':exclusions,
        'grounding_rule':'Each leaf argument is a case-insensitive literal substring or delimited number in the user request; no-call examples excluded. Applies to all partitions before V2 model outputs.',
        'limitations':'This checks literal support, not complete annotation semantics. Copy-heavy positive-call subset; no abstention, temporal inference or full ToolACE claim.',
        'ner_alignment':'Overlapping exact occurrences in source text, zero based; no gold consulted to align a model prediction; unchanged official character-span gold.',
        'split_policy':'Preserved V1 API-disjoint partitions; entity train/internal-dev/public-dev preserved.'})
    write(HERE/'DATA_FROZEN.json',{'at':now(),'files':{str(p.relative_to(HERE)):sha(p) for folder in ['data','tokens'] for p in (HERE/folder).glob('*') if p.is_file()}})
    print(canonical(metadata),flush=True)
if __name__=='__main__':main()
