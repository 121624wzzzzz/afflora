import pathlib,json,hashlib,shutil
ROOT=pathlib.Path(__file__).resolve().parent;PREV=ROOT.parent
SOURCES={'ner':('posttraining_tasks_grounded_20260916','c717690a663b6d58c70224b4b4f798e34343f1e0de49b8e8bf1df7e91d37ab5f'),
 'sql':('downstream_transfer_20260917','ddfcf8b6c43e39303d9900efeac54d68899bf23b51bfb59426a59637ffd1fdc6')}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
records=[];manifests={}
for key,(name,expected) in SOURCES.items():
 p=PREV/name;assert sha(p/'ARTIFACT_MANIFEST.json')==expected;manifests[key]=json.loads((p/'ARTIFACT_MANIFEST.json').read_text())['files']
def checked(key,rel):
 p=PREV/SOURCES[key][0]/rel;assert sha(p)==manifests[key][rel]['sha256'],p
 records.append({'source':str(p),'sha256':sha(p)});return p
def cp(key,rel,dest=None):
 p=checked(key,rel);q=ROOT/(dest or rel);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
for rel in ['modeling.py','run.py','common.py','audit_parameters.py','pipeline.py','monitor.py','source/affine_adapter.py','source/budget.py']:cp('sql',rel)
cp('ner','scoring.py','scoring_ner.py');cp('sql','scoring.py','scoring_sql.py')
cp('ner','common.py','provenance/ner_common.py');cp('sql','common.py','provenance/sql_common.py')
cp('ner','PROTOCOL.md','provenance/ner_PROTOCOL.md') if 'PROTOCOL.md' in manifests['ner'] else cp('ner','DESIGN.md','provenance/ner_DESIGN.md')
for old,new in [('pilot_train','train'),('pilot_dev','dev'),('test','test')]:cp('ner',f'data/cluener_{old}.jsonl',f'data/cluener_{new}.jsonl')
for split in ['train','dev','test']:cp('sql',f'data/wikisql_{split}.jsonl');cp('sql',f'raw/wikisql/data/{split}.db')
for rel in sorted(manifests['sql']):
 if rel.startswith(('raw/official_wikisql/','raw/reference_deps/')):cp('sql',rel)
cp('sql','SCORER_TESTS.json','provenance/sql_SCORER_TESTS.json');cp('ner','DATA_AUDIT.json','provenance/ner_DATA_AUDIT.json');cp('sql','DATA_AUDIT.json','provenance/sql_DATA_AUDIT.json')
anchors=[]
for key,task in [('ner','cluener'),('sql','wikisql')]:
 base=PREV/SOURCES[key][0]
 paths=(base/('heldout' if key=='ner' else 'evaluations')).glob('*/*/SUMMARY.json')
 for f in paths:
  a=json.loads(f.read_text());s=a['spec']
  if s['model']!='qwen25_15b_base' or s['task']!=task or s['arm'] not in ['base','hidden','hidden_budget','hidden_both']:continue
  if (a.get('eval_split') if key=='ner' else s.get('eval_split'))!='test':continue
  if s.get('smoke'):continue
  name=s['name'];destination=f'anchors/{task}/{name}';cp(key,str(f.relative_to(base)),destination+'/SUMMARY.json');cp(key,str((f.parent/'responses.jsonl').relative_to(base)),destination+'/responses.jsonl')
  for fn in ['spec.json','INITIALIZATION.json','COMPLETE.json']+([] if s['arm']=='base' else ['TRAINING.json','TRAIN_ORDER.json']):cp(key,f'checkpoints/{name}/{fn}',destination+'/'+fn)
  if s['arm']!='base':checked(key,f'checkpoints/{name}/adapter.safetensors')
  anchors.append({'task':task,'model':s['model'],'arm':s['arm'],'seed':s['seed'],'directory':destination,'summary':a})
 assert len([a for a in anchors if a['task']==task])==16,task
 for split,old in [('train','pilot_train' if key=='ner' else 'train'),('dev','pilot_dev' if key=='ner' else 'dev'),('test','test')]:cp(key,f'tokens/{task}_qwen25_15b_base_{old}.json',f'anchors/tokens/{task}_{split}.json')
(ROOT/'ANCHOR_INDEX.json').write_text(json.dumps(anchors,ensure_ascii=False,indent=2)+'\n')
(ROOT/'SOURCE_REUSE.json').write_text(json.dumps({'status':'passed','sources':SOURCES,'verified_files':records,'note':'Copies are source starting points; local adaptations will be frozen before model execution. Anchors only reuse independently rescored historical results; no fitted adapters reused for new models.'},ensure_ascii=False,indent=2)+'\n')
print('verified/copied',len(records),'source files;',len(anchors),'1.5B anchor results')
