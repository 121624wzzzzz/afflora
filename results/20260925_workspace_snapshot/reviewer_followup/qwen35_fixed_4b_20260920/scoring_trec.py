"""Fixed class accuracy and official-normalized SQuAD2 multi-reference EM/F1."""
import re,string,collections
from common import *
def normalize(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def exact(pred,gold):return int(normalize(pred)==normalize(gold))
def f1(pred,gold):
 a=normalize(pred).split();b=normalize(gold).split()
 if not a or not b:return float(a==b)
 n=sum((collections.Counter(a)&collections.Counter(b)).values())
 return 2*n/(len(a)+len(b))
def unique_pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate key')
  d[k]=v
 return d
def reject(s):raise ValueError(s)
def parse(text):
 s=text.strip();fenced=s.startswith('```')
 if fenced:
  parts=s.split('\n',1)
  if len(parts)!=2 or parts[0].strip() not in ['```','```json']:return None,False
  s=parts[1].strip()
 try:v,end=json.JSONDecoder(object_pairs_hook=unique_pairs,parse_constant=reject).raw_decode(s)
 except (ValueError,RecursionError):return None,False
 if not isinstance(v,dict) or set(v)!={'answer'} or not isinstance(v['answer'],str):return None,False
 return v['answer'],not fenced and not s[end:].strip()
def score(text,row):
 assert row['task']=='squad2';pred,strict=parse(text);valid=pred is not None;golds=[x for x in row['answers'] if normalize(x)] or ['']
 return {'em':max(exact(pred,g) for g in golds) if valid else 0,'f1':max(f1(pred,g) for g in golds) if valid else 0.,'content_correct':bool(valid and max(exact(pred,g) for g in golds)),
  'schema_valid':valid,'strict_json':strict,'predicted_answer':pred,'gold_unanswerable':row['is_impossible'],
  'predicts_empty':valid and pred=='','answerability_correct':bool(valid and (pred=='')==row['is_impossible']),
  'nonempty_extractive':bool(valid and pred!='' and pred in row['context'])}
def aggregate(rs,task):
 n=len(rs);assert n
 if task=='trec50':
  cats=read(HERE/'data/categories.json');counts=[]
  for i in range(50):
   code=f'{i:02d}';tp=sum(r['predicted_code']==r['gold_code']==code for r in rs);p=sum(r['predicted_code']==code for r in rs);g=sum(r['gold_code']==code for r in rs)
   counts.append({'label':cats[i]['label'],'tp':tp,'pred':p,'gold':g,'f1':200*tp/(p+g) if p+g else 0.})
  return {'n':n,'primary':100*sum(r['content_correct'] for r in rs)/n,'macro_f1_all50':sum(r['f1'] for r in counts)/50,
   'coarse_accuracy':100*sum(cats[int(r['predicted_code'])]['label'].split(':')[0]==cats[int(r['gold_code'])]['label'].split(':')[0] for r in rs)/n,'per_class':counts}
 out={'n':n,'primary':100*sum(r['f1'] for r in rs)/n}
 for k in ['em','f1','schema_valid','strict_json','answerability_correct','predicts_empty','nonempty_extractive']:out[k+'_pct']=100*sum(r[k] for r in rs)/n
 for impossible,name in [(False,'answerable'),(True,'unanswerable')]:
  subset=[r for r in rs if r['gold_unanswerable']==impossible];out[name+'_n']=len(subset)
  out[name+'_f1_pct']=100*sum(r['f1'] for r in subset)/len(subset) if subset else None
  out[name+'_em_pct']=100*sum(r['em'] for r in subset)/len(subset) if subset else None
 return out
