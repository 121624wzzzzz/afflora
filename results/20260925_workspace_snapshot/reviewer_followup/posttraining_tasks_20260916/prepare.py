import ast, collections, re, zipfile, random, sys
from common import *
from scoring import valid_calls, score

def convert_calls(text,names):
    s=text.strip()
    if not s.startswith('['):
        if any(re.search(re.escape(n)+r'\s*\(',s) for n in names):raise ValueError('embedded_call')
        return []
    # Replace known call names only outside string literals and at list boundaries.
    mapping={n:'fn_'+str(i) for i,n in enumerate(names)};rev={v:k for k,v in mapping.items()}
    out='';i=0;quote=None
    while i<len(s):
        ch=s[i]
        if quote:
            out+=ch;i+=1
            if ch=='\\' and i<len(s):out+=s[i];i+=1
            elif ch==quote:quote=None
            continue
        if ch in ['"',"'"]:quote=ch;out+=ch;i+=1;continue
        matched=False
        if out.rstrip().endswith(('[',',')):
            for name in sorted(names,key=len,reverse=True):
                if s.startswith(name,i) and s[i+len(name):].lstrip().startswith('('):
                    out+=mapping[name];i+=len(name);matched=True;break
        if not matched:out+=ch;i+=1
    tree=ast.parse(out,mode='eval').body
    if not isinstance(tree,ast.List):raise ValueError('not_call_list')
    def literal(node):
        if isinstance(node,ast.Name) and node.id in ['null','true','false']:return {'null':None,'true':True,'false':False}[node.id]
        if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.List):return [literal(x) for x in node.elts]
        return ast.literal_eval(node)
    result=[]
    for node in tree.elts:
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id not in rev or node.args:raise ValueError('unsupported_call')
        args={}
        for kw in node.keywords:
            if kw.arg is None or kw.arg in args:raise ValueError('invalid_keyword')
            args[kw.arg]=literal(kw.value)
        result.append({'name':rev[node.func.id],'arguments':args})
    return result

def tool_rows():
    source=read(HERE/'raw/toolace.json');result=[];excluded=collections.Counter();details=[]
    for i,r in enumerate(source):
        try:
            marker='Here is a list of functions in JSON format that you can invoke:'
            tail=r['system'].split(marker,1)[1].lstrip();tools=json.JSONDecoder().raw_decode(tail)[0]
            assert isinstance(tools,list) and tools
            names=[t['name'] for t in tools];assert len(names)==len(set(names))
            turns=r['conversations'];assert turns[0]['from']=='user' and turns[1]['from']=='assistant'
            target=convert_calls(turns[1]['value'],names)
            assert valid_calls(target,tools),'gold_schema_invalid'
            text=turns[0]['value'];assert isinstance(text,str) and text.strip()
            row={'task':'toolace','id':f'toolace_{i}','text':text,'tools':tools,'target':target,'source_index':i}
            assert score(canonical(target),row)['content_correct']
            result.append(row)
        except (ValueError,KeyError,IndexError,AssertionError,TypeError,SyntaxError) as e:
            reason=str(e)[:100] or type(e).__name__;excluded[reason]+=1;details.append({'source_index':i,'reason':reason})
    write(HERE/'data/toolace_exclusions.json',details)
    return result,{'raw_rows':len(source),'converted_rows':len(result),'excluded':dict(excluded)}

def entity_rows():
    with zipfile.ZipFile(HERE/'raw/cluener_public.zip') as z:
        print('zip',z.namelist(),flush=True)
        found={k:next(n for n in z.namelist() if n.endswith('/'+k+'.json') or n==k+'.json') for k in ['train','dev']}
        result={};excluded=[]
        for split,path in found.items():
            result[split]=[]
            for i,s in enumerate(z.read(path).decode().splitlines()):
                r=json.loads(s);gold=[]
                try:
                    for typ,entities in r['label'].items():
                        assert typ in TYPES
                        for text,spans in entities.items():
                            for begin,end in spans:
                                assert r['text'][begin:end+1]==text
                                gold.append({'type':typ,'text':text,'start':begin,'end':end})
                    gold.sort(key=lambda e:(e['start'],e['end'],e['type']))
                    row={'task':'cluener','id':f'cluener_{split}_{i}','text':r['text'],'target':gold,'source_index':i}
                    assert score(canonical(gold),row)['content_correct'];result[split].append(row)
                except AssertionError:excluded.append({'split':split,'index':i})
    return result,{'official_train':len(result['train']),'official_dev':len(result['dev']),'invalid_gold':excluded}

def components(data):
    parent=list(range(len(data)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(data):
        keys=['query:'+re.sub(r'\s+',' ',r['text']).strip().casefold()]
        for t in r['tools']:
            keys.append('name:'+re.sub(r'[^\w]','',t['name']).casefold())
            keys.append('signature:'+htext(canonical(t)))
        for key in keys:
            if key in seen:parent[root(i)]=root(seen[key])
            else:seen[key]=i
    groups=collections.defaultdict(list)
    for i,r in enumerate(data):groups[root(i)].append(r)
    return sorted(groups.values(),key=lambda g:(-len(g),htext(g[0]['id'])))

def main():
    from transformers import AutoTokenizer
    assert not (HERE/'DATA_FROZEN.json').exists()
    tool,a=tool_rows();entities,b=entity_rows();groups=components(tool)
    # Force largest component into train; assign remaining components to deficits.
    splits={'train':list(groups[0]),'dev':[],'test':[]};targets={'train':.70*len(tool),'dev':.15*len(tool),'test':.15*len(tool)}
    for g in groups[1:]:
        dest=max(splits,key=lambda k:(targets[k]-len(splits[k]))/max(targets[k],1))
        splits[dest].extend(g)
    a['components']={'n':len(groups),'largest_sizes':[len(g) for g in groups[:20]],'split_sizes':{k:len(v) for k,v in splits.items()}}
    for split,data in splits.items():
        for r in data:r['group']=htext('|'.join(sorted(t['name'] for t in r['tools'])))
    # Prove no name, schema, or normalized request crosses ToolACE partitions.
    for field,keyfn in [('names',lambda r:[re.sub(r'[^\w]','',t['name']).casefold() for t in r['tools']]),('schemas',lambda r:[canonical(t) for t in r['tools']]),('requests',lambda r:[re.sub(r'\s+',' ',r['text']).strip().casefold()])]:
        sets={k:{s for r in v for s in keyfn(r)} for k,v in splits.items()}
        for k,l in [('train','dev'),('train','test'),('dev','test')]:assert not sets[k]&sets[l],(field,k,l)
    official_texts={r['text'] for r in entities['dev']};pool=[];seen=set();dups=[]
    for r in entities['train']:
        if r['text'] in official_texts or r['text'] in seen:dups.append(r['id']);continue
        seen.add(r['text']);pool.append(r)
    pool.sort(key=lambda r:htext('6100:'+r['text']));entity_splits={'dev':pool[:600],'train':pool[600:],'test':entities['dev']};b['train_duplicates_excluded']=dups
    all_splits={'toolace':splits,'cluener':entity_splits};models=read(HERE/'models.json')
    tokenizers={m:AutoTokenizer.from_pretrained(models[m]['path'],local_files_only=True) for m in MODELS}
    exclusion=[];metadata={}
    for task,task_splits in all_splits.items():
        metadata[task]={}
        for split,data in task_splits.items():
            kept=[];tokens={m:[] for m in MODELS}
            for r in sorted(data,key=lambda x:htext('6100:'+x['id'])):
                tokrows={}
                for m,tok in tokenizers.items():
                    p=tok.encode(prompt(r),add_special_tokens=False);target=tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]
                    tokrows[m]={'id':r['id'],'prompt_ids':p,'target_ids':target}
                # Predictable input limit for ALL splits. Target limit only train/dev;
                # never filter test according to its answer length.
                bad=any(len(x['prompt_ids'])>1536 or (split!='test' and len(x['target_ids'])>512) for x in tokrows.values())
                if bad:exclusion.append({'id':r['id'],'split':split,'lengths':{m:[len(x['prompt_ids']),len(x['target_ids'])] for m,x in tokrows.items()}});continue
                kept.append(r)
                for m,x in tokrows.items():tokens[m].append(x)
            write_rows(HERE/f'data/{task}_{split}.jsonl',kept)
            for m,v in tokens.items():write(HERE/f'tokens/{task}_{m}_{split}.json',v)
            metadata[task][split]={'n':len(kept),'calls':sum(bool(r['target']) for r in kept),'ids_sha256':htext(canonical([r['id'] for r in kept]))}
            if split!='test':
                n=min(2048,len(kept)) if split=='train' else min(200,len(kept))
                write_rows(HERE/f'data/{task}_pilot_{split}.jsonl',kept[:n])
                for m,v in tokens.items():write(HERE/f'tokens/{task}_{m}_pilot_{split}.json',v[:n])
    write(HERE/'DATA_AUDIT.json',{'toolace':a,'cluener':b,'splits':metadata,'length_exclusions':exclusion,'max_prompt_tokens':1536,'generation_budget':512,'pilot_train_max':2048,'pilot_dev_max':200,'prepared_at':now()})
    files={str(p.relative_to(HERE)):sha(p) for folder in ['raw','data','tokens'] for p in (HERE/folder).glob('*') if p.is_file()}
    write(HERE/'DATA_FROZEN.json',{'at':now(),'files':files})
    print(json.dumps({'toolace':a,'cluener':b,'splits':metadata},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
