"""Gold-independent parsing; invalid responses stay in the denominator."""
import json, math
from collections import Counter
from common import canonical, TYPES

def reject_constants(value):raise ValueError(value)
def unique_object(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate key')
        d[k]=v
    return d
DECODER=json.JSONDecoder(object_pairs_hook=unique_object,parse_constant=reject_constants)
def parse(text):
    s=text.strip();route='leading_json';strict=False
    try:
        value,end=DECODER.raw_decode(s);strict=not s[end:].strip()
    except (ValueError,RecursionError):
        if not s.startswith('```'):return None,False,'invalid'
        parts=s.split('\n',1)
        if len(parts)!=2 or parts[0].strip() not in ['```','```json']:return None,False,'invalid'
        s=parts[1].strip();route='code_fence'
        try:value,end=DECODER.raw_decode(s)
        except (ValueError,RecursionError):return None,False,'invalid'
    if not isinstance(value,list):return None,False,'not_array'
    return value,strict,route

def valid_type(v,schema):
    kind=schema.get('type')
    kinds={'str':'string','int':'integer','float':'number','bool':'boolean','dict':'object','list':'array'}
    kind=kinds.get(kind,kind)
    if kind=='string':ok=isinstance(v,str)
    elif kind=='integer':ok=type(v)==int
    elif kind=='number':ok=type(v) in [int,float] and math.isfinite(v)
    elif kind=='boolean':ok=type(v)==bool
    elif kind=='array':ok=isinstance(v,list) and all(valid_type(z,schema.get('items',{})) for z in v)
    elif kind=='object':
        ok=isinstance(v,dict)
        if ok:
            props=schema.get('properties',{})
            ok=all(k in v for k in (schema.get('required') or [])) and all(k in props and valid_type(z,props[k]) for k,z in v.items())
    elif kind is None:ok=True
    else:ok=False
    return ok and ('enum' not in schema or v in schema['enum'])

def valid_calls(calls,tools):
    byname={t['name']:t for t in tools}
    return all(isinstance(c,dict) and set(c)=={'name','arguments'} and isinstance(c['name'],str)
        and c['name'] in byname and isinstance(c['arguments'],dict)
        and valid_type(c['arguments'],byname[c['name']].get('parameters',{'type':'object','properties':{}})) for c in calls)

def tool_score(value,row):
    gold=row['target'];valid=value is not None and valid_calls(value,row['tools'])
    pred=[] if value is None else value
    # Explicitly supplied documented defaults and omitted defaults are equivalent.
    # Number schemas allow int/float equivalence; booleans remain distinct.
    tools={t['name']:t for t in row['tools']}
    def normalize(c):
        args=dict(c['arguments']);props=tools[c['name']].get('parameters',{}).get('properties',{})
        for k,s in props.items():
            if k not in args and 'default' in s:args[k]=s['default']
            if k in args and s.get('type') in ['number','float'] and type(args[k]) in [int,float]:args[k]=float(args[k])
        return canonical({'name':c['name'],'arguments':args})
    semantic=valid and Counter(normalize(c) for c in pred)==Counter(normalize(c) for c in gold)
    names=value is not None and all(isinstance(c,dict) and isinstance(c.get('name'),str) for c in pred)
    names=names and Counter(c['name'] for c in pred)==Counter(c['name'] for c in gold)
    return {'content_correct':bool(semantic),'function_correct':bool(names),'schema_valid':bool(valid),'no_call_gold':not gold}

def entity_score(value,row):
    gold=row['target'];ps=Counter();pt=Counter();valid=value is not None
    for i,e in enumerate(value or []):
        item=isinstance(e,dict) and set(e)=={'type','text','start','end'}
        typed=isinstance(e,dict) and isinstance(e.get('type'),str) and e['type'] in TYPES and isinstance(e.get('text'),str) and bool(e['text'])
        span=item and typed and type(e['start'])==int and type(e['end'])==int and 0<=e['start']<=e['end']<len(row['text'])
        span=span and row['text'][e['start']:e['end']+1]==e['text']
        valid=valid and span
        ps[(e['type'],e['start'],e['end']) if span else ('INVALID',i)]+=1
        pt[(e['type'],e['text']) if typed else ('INVALID',i)]+=1
    gs=Counter((e['type'],e['start'],e['end']) for e in gold)
    gt=Counter((e['type'],e['text']) for e in gold)
    return {'schema_valid':bool(valid),'span_tp':sum((ps&gs).values()),'span_pred':sum(ps.values()),'span_gold':sum(gs.values()),
        'text_tp':sum((pt&gt).values()),'text_pred':sum(pt.values()),'text_gold':sum(gt.values()),
        'content_correct':bool(valid and ps==gs)}

def score(text,row):
    value,strict,route=parse(text)
    r=tool_score(value,row) if row['task']=='toolace' else entity_score(value,row)
    return dict(r,json_valid=value is not None,strict_json=strict,parser_route=route)

def aggregate(records,task):
    n=len(records);assert n
    out={k:100*sum(bool(r[k]) for r in records)/n for k in ['content_correct','schema_valid','json_valid','strict_json']}
    if task=='toolace':
        out['function_correct']=100*sum(r['function_correct'] for r in records)/n
        for nc in [False,True]:
            sub=[r for r in records if r['no_call_gold']==nc];out['no_call' if nc else 'call_required']={'n':len(sub),'accuracy':100*sum(r['content_correct'] for r in sub)/len(sub) if sub else None}
        out['primary']=out['content_correct']
    else:
        for kind in ['span','text']:
            tp=sum(r[kind+'_tp'] for r in records);p=sum(r[kind+'_pred'] for r in records);g=sum(r[kind+'_gold'] for r in records)
            out[kind+'_micro_f1']=200*tp/(p+g) if p+g else 100.;out[kind+'_counts']={'tp':tp,'pred':p,'gold':g}
        out['primary']=out['span_micro_f1']
    out['n']=n
    return out
