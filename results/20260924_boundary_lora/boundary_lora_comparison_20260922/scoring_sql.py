"""Gold-blind parsing; WikiSQL's official execution and LF conventions."""
import functools, math, re, sqlite3
from babel.numbers import parse_decimal, NumberFormatError
from common import *

AGG=['','MAX','MIN','COUNT','SUM','AVG']
OPS=['=','>','<']
NUM=re.compile(r'[-+]?\d*\.\d+|\d+')

def unique_pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d

def reject_constant(s):raise ValueError('nonfinite JSON')

def parse(text):
    s=text.strip()
    fenced=s.startswith('```')
    if s.startswith('```'):
        s=re.sub(r'^```(?:json)?\s*','',s,count=1)
    obj,end=json.JSONDecoder(object_pairs_hook=unique_pairs,parse_constant=reject_constant).raw_decode(s)
    trailing=s[end:].strip()
    return obj,not fenced and trailing==''

def validate(q,n):
    assert isinstance(q,dict) and set(q)=={'sel','agg','conds'}
    assert type(q['sel'])==int and 0<=q['sel']<n
    assert type(q['agg'])==int and 0<=q['agg']<6
    assert isinstance(q['conds'],list)
    for c in q['conds']:
        assert isinstance(c,list) and len(c)==3
        col,op,val=c
        assert type(col)==int and 0<=col<n
        assert type(op)==int and 0<=op<3
        assert type(val) in (str,int,float)
        assert type(val)!=float or math.isfinite(val)
    return q

@functools.lru_cache(None)
def connection(split):
    assert split in ('train','dev','test')
    p=(HERE/f'raw/wikisql/data/{split}.db').resolve()
    return sqlite3.connect(f'file:{p}?mode=ro',uri=True)

def execute(q,row,distinct_parameters=False):
    validate(q,len(row['header']))
    tid=row['table_id'];assert re.fullmatch(r'[0-9A-Za-z_-]+',tid)
    table=tid if tid.startswith('table') else 'table_'+tid.replace('-','_')
    col='col'+str(q['sel']);select=f'{AGG[q["agg"]]}({col})' if q['agg'] else col
    params={};conditions=[]
    for i,(c,op,v) in enumerate(q['conds']):
        if isinstance(v,str):v=v.lower()
        if row['types'][c]=='real' and not isinstance(v,(int,float)):
            try:v=float(parse_decimal(v))
            except NumberFormatError:v=float(NUM.findall(v)[0])
        # This default reproduces the official DBEngine, including repeated-column
        # placeholder behavior. The mathematically distinct variant is diagnosed.
        key=f'v{i}' if distinct_parameters else f'col{c}'
        params[key]=v;conditions.append(f'col{c} {OPS[op]} :{key}')
    sql=f'SELECT {select} AS result FROM {table}'
    if conditions:sql+=' WHERE '+' AND '.join(conditions)
    return [r[0] for r in connection(row['source_split']).execute(sql,params)]

def logical(q):
    return (q['sel'],q['agg'],frozenset((c,o,str(v).lower()) for c,o,v in q['conds']))

def score(text,row):
    assert row['task']=='wikisql'
    out={'content_correct':False,'lf_correct':False,'query_valid':False,'strict_json':False,
         'execution_ok':False,'empty_prediction':False,'duplicate_condition_column':False,
         'distinct_parameter_execution_correct':False}
    try:
        q,strict=parse(text);out['strict_json']=strict
        validate(q,len(row['header']));out['query_valid']=True
        out['lf_correct']=logical(q)==logical(row['target'])
        cols=[c[0] for c in q['conds']];out['duplicate_condition_column']=len(cols)!=len(set(cols))
        pred=execute(q,row);out['execution_ok']=True;out['empty_prediction']=pred==[]
        gold=row.get('gold_execution')
        if gold is None:gold=execute(row['target'],row)
        out['content_correct']=pred==gold
        out['distinct_parameter_execution_correct']=execute(q,row,True)==execute(row['target'],row,True)
    except (AssertionError,ValueError,TypeError,KeyError,IndexError,sqlite3.Error,OverflowError) as e:
        out['error']=type(e).__name__
    return out

def aggregate(records,task):
    result={'n':len(records),'primary':100*sum(x['content_correct'] for x in records)/len(records)}
    keys=['lf_correct','query_valid','strict_json','execution_ok','empty_prediction',
          'duplicate_condition_column','distinct_parameter_execution_correct'] if task=='wikisql' else ['unrestricted_correct','unrestricted_valid']
    for k in keys:result[k+'_pct']=100*sum(x[k] for x in records)/len(records)
    return result
