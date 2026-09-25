from common import HERE,read
from label_parser import parse_label

def score(text,row):
 pred,strict=parse_label(text,read(HERE/f"data/{row['task']}_labels.json"))
 return dict(predicted_label=pred,gold_label=row['target'],label_valid=pred is not None,content_correct=pred==row['target'],strict_valid=strict,strict_correct=bool(strict and pred==row['target']))

def aggregate(records,task):
 assert task in ['clinc150','emotion'] and records
 labels=sorted({r['gold_label'] for r in records});f1=[]
 for label in labels:
  tp=sum(r['gold_label']==label and r['predicted_label']==label for r in records)
  fp=sum(r['gold_label']!=label and r['predicted_label']==label for r in records)
  fn=sum(r['gold_label']==label and r['predicted_label']!=label for r in records)
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 n=len(records)
 return dict(primary=100*sum(r['content_correct'] for r in records)/n,macro_f1=100*sum(f1)/len(f1),valid_pct=100*sum(r['label_valid'] for r in records)/n,strict_accuracy=100*sum(r['strict_correct'] for r in records)/n,strict_valid_pct=100*sum(r['strict_valid'] for r in records)/n,n=n)
