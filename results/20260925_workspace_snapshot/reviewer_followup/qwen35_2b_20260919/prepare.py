"""Re-encode verified shared benchmark splits and independently audit their identities."""
import sys,unicodedata,collections,types
from transformers import AutoTokenizer,AutoConfig
from common import *
from scoring import score
def official_engine(split):
 sys.path.insert(0,str(HERE/'raw/reference_deps'));sys.path.insert(0,str(HERE/'raw/official_wikisql'))
 from lib.dbengine import DBEngine
 return DBEngine(str(HERE/f'raw/wikisql/data/{split}.db'))
def reference_prompts():
 out={}
 for task,name in [('wikisql','reference_sql_common.py.txt'),('trec50','reference_trec_common.py.txt')]:
  ns={'__file__':str(HERE/'common.py')};exec(compile((HERE/'provenance'/name).read_text(),name,'exec'),ns);out[task]=ns['prompt']
 return out
def key(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def main():
 assert not (HERE/'DATA_FROZEN.json').exists()
 path=HERE.parents[2]/'models/Qwen3.5-2B-Base'
 official=read(HERE/'provenance/qwen35_2b_official_metadata.json');known={x['rfilename']:x for x in official['siblings']}
 verified={}
 for name in ['tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','config.json']:
  p=path/name;r=known[name];assert p.stat().st_size==r['size'];h=sha(p)
  if r.get('lfs'):assert h==r['lfs']['sha256']
  else:
   raw=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['blobId']
  verified[name]=h
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True);cfg=AutoConfig.from_pretrained(path,local_files_only=True).text_config
 assert (tok.bos_token_id,tok.eos_token_id,tok.pad_token_id)==(TOKEN_BOS_ID,TOKEN_EOS_ID,TOKEN_PAD_ID) and cfg.tie_word_embeddings==read(HERE/'models.json')[MODEL]['tie_word_embeddings']
 refs=reference_prompts();copied=read(HERE/'SOURCE_REUSE.json')['copied_files'];audit={'at':now(),'tokenizer_files':verified,'splits':{},'scope':'Shared previously evaluated benchmark splits; no previous Qwen3.5 task fit/prediction is reused.'}
 gold_checks=mutations=prompt_checks=0
 for task in TASKS:
  group_sets=[]
  for split in ['train','dev','confirm']:
   rel=f'data/{task}_{split}.jsonl';assert sha(HERE/rel)==copied[rel]['sha256']
   data=rows(HERE/rel);tokens=[]
   if task=='wikisql':
    engine=official_engine(data[0]['source_split']);from lib.query import Query
   else:
    cats=read(HERE/'data/categories.json');assert len(cats)==50
    filename='TREC_10.label' if split=='confirm' else 'train_5500.label';raw=(HERE/'raw/trec'/filename).read_bytes().decode('latin-1').splitlines()
   for i,r in enumerate(data):
    assert prompt(r)==refs[task](r);prompt_checks+=1
    if task=='wikisql':
     assert engine.execute_query(r['table_id'],Query.from_dict(r['target']),lower=True)==r['gold_execution'];assert score(canonical(r['target']),r)['content_correct'];gold_checks+=1
     if i<128:
      q=r['target']
      for alt in [dict(q,agg=(q['agg']+1)%6),dict(q,conds=[]),dict(q,sel=(q['sel']+1)%len(r['header']))]:
       try:ref=engine.execute_query(r['table_id'],Query.from_dict(alt),lower=True)==r['gold_execution']
       except Exception:ref=False
       ours=score(canonical(alt),r);assert ours['content_correct']==ref and ours['lf_correct']==(Query.from_dict(alt)==Query.from_dict(q));mutations+=1
    else:
     label,text=raw[r['source_index']].split(' ',1);assert label==r['label'] and text==r['text'] and cats[int(r['target'])]['label']==label;gold_checks+=1
    p=tok.encode(prompt(r),add_special_tokens=True);t=tok.encode(target_text(r),add_special_tokens=False)+[TOKEN_EOS_ID]
    assert p==tok.encode(prompt(r),add_special_tokens=False)
    assert len(p)+len(t)<=2048 and len(t)<=256
    assert not set(tok.all_special_ids)&set(p)
    assert t.count(TOKEN_EOS_ID)==1 and not(set(tok.all_special_ids)-{TOKEN_EOS_ID})&set(t)
    assert tok.encode(prompt(r)+target_text(r),add_special_tokens=True)==p+t[:-1]
    tokens.append({'id':r['id'],'prompt_ids':p,'target_ids':t})
   write(HERE/f'tokens/{task}_{MODEL}_{split}.json',tokens)
   groups={r['table_id'] if task=='wikisql' else key(r['text']) for r in data};group_sets.append(groups)
   audit['splits'][task+'_'+split]={'n':len(data),'groups':len(groups),'max_prompt':max(len(x['prompt_ids']) for x in tokens),'max_target':max(len(x['target_ids']) for x in tokens),'data_sha256':sha(HERE/rel),'prompt_sha256':htext(canonical([prompt(r) for r in data]))}
  assert all(not group_sets[i]&group_sets[j] for i in range(3) for j in range(i))
 codes=[tok.encode(f'{i:02d}',add_special_tokens=False) for i in range(50)];assert {len(x) for x in codes}=={2} and len({tuple(x) for x in codes})==50 and len({x[0] for x in codes})==5
 write(HERE/f'tokens/{MODEL}_codes.json',codes)
 assert len(rows(HERE/'data/trec50_confirm.jsonl'))==500
 from architecture import architecture_plan
 plan=architecture_plan(cfg);h=plan['hidden_parameters'];extra=65*cfg.hidden_size
 write(HERE/'BUDGET_PLAN.json',{MODEL:{'base':0,'hidden':h,'hidden_budget':h+extra,'hidden_both':h+extra}})
 write(HERE/'BUDGET_ALLOCATION.json',plan)
 audit.update(status='passed',gold_checks=gold_checks,mutation_checks=mutations,prompt_checks=prompt_checks)
 write(HERE/'DATA_AUDIT.json',audit)
 files={str(p.relative_to(HERE)):sha(p) for folder in ['data','tokens','raw'] for p in (HERE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 write(HERE/'DATA_FROZEN.json',{'at':now(),'files':files});print(canonical(audit),flush=True)
if __name__=='__main__':main()
