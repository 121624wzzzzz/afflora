from common import *
from label_parser import parse_label
from metric_audit import independent_prediction

def main():
 checked=0
 for task in TASKS:
  labels=read(HERE/f'data/{task}_labels.json')
  for label in labels:
   cases=[(canonical(label),(label,True)),(label,(label,False)),('Intent: '+label+'\nIntent: '+label,(label,False)),('Emotion: '+label,(label,False)),(label+' because this is correct',(None,False)),(label+'\n'+next(x for x in labels if x!=label),(None,False))]
   for text,expected in cases:
    assert parse_label(text,labels)==expected,(task,text)
    assert independent_prediction(text,labels)==expected,(task,text)
    checked+=1
 for text in ['', 'null', '[]', '{}', '"joy" trailing', 'enjoy', 'JOY', 'joy\nsadness']:
  assert parse_label(text,['joy','sadness'])==(None,False)
  assert independent_prediction(text,['joy','sadness'])==(None,False)
  checked+=1
 write(HERE/'PARSER_AUDIT.json',dict(at=now(),status='passed',cases=checked,includes_digit_labels=True))
 print(read(HERE/'PARSER_AUDIT.json'))
if __name__=='__main__':main()
