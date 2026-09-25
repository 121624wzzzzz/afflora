"""Post-hoc, gold-class error accounting; no new tests or model selection."""
from reporting_common import *
from collections import Counter
import argparse,csv

def rows(path):
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def build(preview=False):
 if not preview:assert read(ROOT/'FINAL_AUDIT.json')['status']=='passed'
 path=ROOT/'new_tasks';cats=read(path/'data/categories.json');codes=[f'{i:02d}' for i in range(50)]
 labels=dict(zip(codes,[c['label'] for c in cats]));source_hashes={'new_tasks/data/categories.json':sha(path/'data/categories.json')}
 datasets={s:rows(path/'data'/f'trec50_{s}.jsonl') for s in ['train','dev','test']}
 supports={s:Counter(r['target'] for r in rs) for s,rs in datasets.items()}
 for s in datasets:source_hashes[f'new_tasks/data/trec50_{s}.jsonl']=sha(path/'data'/f'trec50_{s}.jsonl')
 conditions=[];class_rows=[]
 for split in ['dev','test']:
  data=collect('new_tasks',split);gold={r['id']:r['target'] for r in datasets[split]};n=len(gold)
  assert n==len(datasets[split])
  for model in MODELS:
   if not all(series(data,'trec50',model,a,3) is not None for a in ['base','hidden','hidden_budget','hidden_both']):continue
   predictions={}
   for arm in ['hidden','hidden_budget','hidden_both']:
    for seed in range(9100,9103):
     record=data['trec50',model,arm,seed];p=record['path'].parent/'responses.jsonl';source_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
     assert source_hashes[p.relative_to(ROOT).as_posix()]==record['summary']['responses_sha256']
     rs=rows(p);by_id={r['id']:r for r in rs};assert len(rs)==len(by_id)==n and set(by_id)==set(gold)
     assert all(r['gold_code']==gold[k] and r['predicted_code'] in codes and r['content_correct']==(r['predicted_code']==gold[k]) for k,r in by_id.items())
     predictions[arm,seed]={k:r['predicted_code'] for k,r in by_id.items()}
     assert abs(100*sum(p==gold[k] for k,p in predictions[arm,seed].items())/n-record['primary'])<1e-9
   for control in ['hidden','hidden_budget']:
    per_seed=[];per_class={c:Counter() for c in codes};per_example={k:Counter() for k in gold}
    for seed in range(9100,9103):
     tally=Counter()
     for key,target in gold.items():
      h=predictions[control,seed][key];eu=predictions['hidden_both',seed][key]
      hc=h==target;ec=eu==target
      case='both_correct' if hc and ec else 'recovered' if ec else 'regressed' if hc else 'both_wrong'
      tally[case]+=1;per_class[target][case]+=1;per_example[key][case]+=1
      per_class[target]['control_correct']+=hc;per_class[target]['stack_correct']+=ec
      tally['coarse_control_correct']+=labels[h].split(':')[0]==labels[target].split(':')[0]
      tally['coarse_stack_correct']+=labels[eu].split(':')[0]==labels[target].split(':')[0]
     delta=100*(tally['recovered']-tally['regressed'])/n
     assert abs(delta-(data['trec50',model,'hidden_both',seed]['primary']-data['trec50',model,control,seed]['primary']))<1e-9
     per_seed.append({'seed':seed,**{k:tally[k] for k in ['both_correct','both_wrong','recovered','regressed','coarse_control_correct','coarse_stack_correct']},'delta_pp':delta})
    classes=[]
    for code in codes:
     count=per_class[code];row={'split':split,'model':model,'control':control,'code':code,'label':labels[code],
      **{s+'_unique_support':supports[s][code] for s in supports},
      **{k:count[k] for k in ['control_correct','stack_correct','recovered','regressed']},
      'net_correct_over_three_seeds':count['stack_correct']-count['control_correct']}
     classes.append(row);class_rows.append(row)
    assert sum(c['net_correct_over_three_seeds'] for c in classes)==sum(s['recovered']-s['regressed'] for s in per_seed)
    examples=[{'id':k,'gold_code':gold[k],'label':labels[gold[k]],'recovered_seeds':v['recovered'],'regressed_seeds':v['regressed']} for k,v in per_example.items() if v['recovered'] or v['regressed']]
    conditions.append({'split':split,'model':model,'control':control,'unique_examples':n,'seeds':[9100,9101,9102],
      'per_seed':per_seed,'mean_delta_pp':float(np.mean([s['delta_pp'] for s in per_seed])),
      'always_regressed_unique_examples':sum(v['regressed']==3 for v in per_example.values()),
      'always_recovered_unique_examples':sum(v['recovered']==3 for v in per_example.values()),'classes':classes,'changed_correctness_examples':examples})
 stem='TREC_ERROR_ANALYSIS'+('_PREVIEW' if preview else '')
 scope='Post-hoc descriptive analysis designed after observing the Qwen3-8B TREC dev/test reversal. Includes every complete preselected three-seed model block and all 50 gold classes, with no new tests, threshold tuning, or training changes. Repeated seeds evaluate the same examples; pooled counts are not independent sample sizes.'
 write(ROOT/(stem+'.json'),{'at':now(),'status':'passed','scope':scope,'source_sha256':source_hashes,'conditions':conditions})
 with (ROOT/(stem+'_CLASSES.csv')).open('w',newline='') as f:
  fields=['split','model','control','code','label','train_unique_support','dev_unique_support','test_unique_support','control_correct','stack_correct','recovered','regressed','net_correct_over_three_seeds']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(class_rows)
 lines=['# TREC50逐例错误核对','',
  '此诊断在观察到Qwen3-8B的开发/测试方向相反后设计，属于事后描述性分析。纳入所有已完成预定三种子的模型块、开发与测试两侧以及全部50个金标准类别；不增加检验、不调整类别阈值、不修改训练。','',
  'recovered表示对照答错而双侧答对，regressed表示对照答对而双侧答错。准确率差严格等于两者数量差除以该split样本数，已逐次核对原审计摘要。三种子反复评测同一批问题，不能把合计计数当成三倍独立样本。','',
  '|split|模型|对照|seed|改善例数|退步例数|差值pp|','|---|---|---|---:|---:|---:|---:|']
 for c in conditions:
  for s in c['per_seed']:lines.append(f"|{c['split']}|{LABELS[c['model']]}|{ARM_LABELS[c['control']]}|{s['seed']}|{s['recovered']}|{s['regressed']}|{s['delta_pp']:+.3f}|")
 lines+=['',f'全部类别的训练/开发/测试支持数和逐类变化见 [{stem}_CLASSES.csv]({stem}_CLASSES.csv)；逐问题跨种子改善/退步次数与数据哈希见 [{stem}.json]({stem}.json)。','',
  '类别支持数和错误分布可以定位差异，不能单独证明它由分布偏移、过拟合或某个分支的梯度冲突造成。宏F1、粗类别准确率和主准确率仍采用原冻结定义。','']
 (ROOT/(stem+'.md')).write_text('\n'.join(lines))
 print('TREC descriptive error accounting passed:',len(conditions),'model/split/control groups')
 return len(conditions)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args();build(args.preview)
