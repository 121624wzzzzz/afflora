"""Gold-blind exact label extraction; no fuzzy matching or first-answer guessing."""
import json

def parse_label(text,labels):
 text=text.strip()
 try:obj=json.loads(text)
 except (ValueError,TypeError):obj=None
 if isinstance(obj,str) and obj in labels:return obj,True
 # Accept a bare exact label or repeated identical label-only lines.
 # Reject explanations, multiple distinct labels, substrings, case/synonym guesses.
 lines=[x.strip() for x in text.splitlines() if x.strip()]
 if not lines:return None,False
 parsed=[]
 for line in lines:
  for prefix in ['Intent:','Emotion:']:
   if line.startswith(prefix):line=line[len(prefix):].strip();break
  try:value=json.loads(line)
  except ValueError:value=line
  if not isinstance(value,str) or value not in labels:return None,False
  parsed.append(value)
 return (parsed[0],False) if len(set(parsed))==1 else (None,False)
