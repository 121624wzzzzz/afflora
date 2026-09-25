from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures,hashlib,json,requests
HERE=Path(__file__).resolve().parent
URLS={
 'trec/train_5500.label':'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label',
 'trec/TREC_10.label':'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label',
 'trec/definition.html':'https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html',
 'squad/train-v2.0.json':'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
 'squad/dev-v2.0.json':'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
 'squad/evaluate-v2.0.py':'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/evaluate-v2.0.py'}
def get(item):
 name,url=item;s=requests.Session();s.trust_env=False;r=s.get(url,timeout=120);r.raise_for_status();p=HERE/'raw'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
 print(name,len(r.content),'downloaded',flush=True)
 return {'file':str(p.relative_to(HERE)),'url':url,'response_url':r.url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(get,URLS.items()))
 (HERE/'DATA_SOURCES.json').write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'status':'passed','files':files},indent=2)+'\n')
