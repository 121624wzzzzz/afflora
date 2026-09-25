import json
from common import HERE,read

def score(text,row):
 try:pred=json.loads(text.strip())
 except (ValueError,TypeError):pred=None
 valid=isinstance(pred,str) and pred in read(HERE/'data/labels.json')
 return dict(predicted_label=pred if valid else None,gold_label=row['target'],label_valid=valid,content_correct=bool(valid and pred==row['target']))

def aggregate(records,task):
 assert task=='clinc150' and records
 labels=sorted({r['gold_label'] for r in records});f1=[]
 for label in labels:
  tp=sum(r['gold_label']==label and r['predicted_label']==label for r in records)
  fp=sum(r['gold_label']!=label and r['predicted_label']==label for r in records)
  fn=sum(r['gold_label']==label and r['predicted_label']!=label for r in records)
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return dict(primary=100*sum(r['content_correct'] for r in records)/len(records),macro_f1=100*sum(f1)/len(f1),valid_pct=100*sum(r['label_valid'] for r in records)/len(records),n=len(records))
