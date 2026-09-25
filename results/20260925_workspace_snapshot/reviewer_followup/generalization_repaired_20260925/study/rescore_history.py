"""Post-hoc diagnostic only; never overwrite source results or reselect their configuration."""
from common import *
from scoring import score,aggregate
from metric_audit import audit_metrics
from transformers import AutoTokenizer
import collections

def main():
 source=Path(read(HERE/'SOURCE.json')['implementation_source']);manifest=read(source/'OUTPUT_MANIFEST.json')
 assert sha(source/'OUTPUT_MANIFEST.json')==read(source/'FINAL_AUDIT.json')['manifest_sha256']
 for rel,h in read(source/'CODE_FROZEN.json')['files'].items():assert sha(source/rel)==h
 assert read(source/'BASE_AUDIT.json')['status']=='passed'
 tokens={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in read(source/'models.json').items()}
 out=[];source_hashes={};checked=0
 for ep in sorted((source/'evaluations').glob('*/test')):
  summ=read(ep/'SUMMARY.json');s=summ['spec']
  if s['stage'] not in ['confirmation','base']:continue
  source_hashes[str(ep/'SUMMARY.json')]=sha(ep/'SUMMARY.json')
  assert sha(ep/'responses.jsonl')==summ['responses_sha256']
  if s['stage']=='confirmation':assert sha(ep/'SUMMARY.json')==manifest['files'][str((ep/'SUMMARY.json').relative_to(source))]
  rs=rows(ep/'responses.jsonl');ds=rows(source/'data/clinc150_test.jsonl');assert [r['id'] for r in rs]==[r['id'] for r in ds]
  assert sha(source/'data/clinc150_test.jsonl')==sha(HERE/'data/clinc150_test.jsonl')
  new=[]
  for r,g in zip(rs,ds):
   assert tokens[s['model']].decode(r['token_ids'],skip_special_tokens=False)==r['text']
   z=score(r['text'],g);assert z['strict_correct']==r['content_correct']
   new.append(dict(z,id=r['id'],text=r['text']));checked+=1
  result=aggregate(new,'clinc150');audit_metrics(new,ds,result,read(HERE/'data/clinc150_labels.json'))
  assert abs(result['strict_accuracy']-summ['primary'])<1e-9
  out.append(dict(spec=s,metrics=result,source_summary_sha256=sha(ep/'SUMMARY.json'),source_responses_sha256=summ['responses_sha256']))
 groups=collections.defaultdict(list)
 for r in out:groups[r['spec']['model'],r['spec']['method'],r['spec']['stage']].append(r)
 lines=['# CLINC150 旧输出统一重评分（事后诊断）','原始指标和配置不改。内容分容许精确裸标签/一致重复标签，拒绝解释、多标签和模糊匹配。','','|模型|方法|阶段|n|原严格准确率|内容准确率|内容宏F1|','|---|---|---|---:|---:|---:|---:|']
 for (m,method,stage),rs in sorted(groups.items()):
  mean=lambda k:sum(r['metrics'][k] for r in rs)/len(rs)
  lines.append(f'|{m}|{method}|{stage}|{len(rs)}|{mean("strict_accuracy"):.4f}|{mean("primary"):.4f}|{mean("macro_f1"):.4f}|')
 (HERE/'historical_rescore/RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 write(HERE/'historical_rescore/RESULTS.json',dict(at=now(),posthoc=True,records=out))
 write(HERE/'historical_rescore/AUDIT.json',dict(at=now(),status='passed',responses=checked,source_hashes=source_hashes,note='Exact output hashes, token decoding, original strict metric preservation and independent parser/sklearn checked. No source artifacts modified.'))
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
