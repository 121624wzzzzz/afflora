"""Independent full-string parser checks and sklearn metrics."""
import re,json
from sklearn.metrics import accuracy_score,f1_score

def independent_prediction(text,labels):
 t=text.strip()
 try:value=json.loads(t)
 except ValueError:value=None
 if isinstance(value,str) and value in labels:return value,True
 # Full-match each line; never scan for any label substring.
 pattern=re.compile(r'(?:(?:Intent|Emotion):\s*)?(?P<value>"(?:[^"\\]|\\.)*"|[a-z_0-9]+)')
 values=[]
 for line in filter(None,(s.strip() for s in t.splitlines())):
  match=pattern.fullmatch(line)
  if not match:return None,False
  v=match['value']
  if v.startswith('"'):
   try:v=json.loads(v)
   except ValueError:return None,False
  if v not in labels:return None,False
  values.append(v)
 if not values or len(set(values))!=1:return None,False
 return values[0],False

def audit_metrics(records,data,summary,labels):
 pred=[];strict=[]
 for r,g in zip(records,data):
  label,valid=independent_prediction(r['text'],labels)
  assert r['predicted_label']==label and r['strict_valid']==valid
  assert r['content_correct']==(label==g['target'])
  pred.append(label if label is not None else '__invalid__');strict.append(bool(valid and label==g['target']))
 y=[g['target'] for g in data]
 assert abs(summary['primary']-100*accuracy_score(y,pred))<1e-9
 assert abs(summary['macro_f1']-100*f1_score(y,pred,labels=sorted(set(y)),average='macro',zero_division=0))<1e-9
 assert abs(summary['strict_accuracy']-100*sum(strict)/len(strict))<1e-9
