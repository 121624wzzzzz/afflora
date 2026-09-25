import argparse
import csv
import io
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
from shared import D,B,read,rows,sha,write,now,jobs

def summarize(raw,split,data):
    n=len(raw);tokens=sum(x['tokens'] for x in raw);den=tokens+n
    total=lambda k:sum(x[k] for x in raw)
    m={'items':n,'content_tokens':tokens,'content_micro_ce':total('nll')/tokens,
       'content_macro_ce':float(np.mean([x['nll']/x['tokens'] for x in raw])),
       'total_micro_ce':(total('nll')+total('eos_nll'))/den,
       'content_contribution':total('nll')/den,'first_contribution':total('first_nll')/den,
       'rest_contribution':total('rest_nll')/den,'eos_contribution':total('eos_nll')/den,
       'eos_ce':total('eos_nll')/n,'eos_top1':total('eos_top1')/n,
       'first_ce':total('first_nll')/n,'first_top1':total('first_top1')/n,'first_top5':total('first_top5')/n,
       'rest_ce':total('rest_nll')/(tokens-n),'rest_top1':total('rest_top1_correct')/(tokens-n),
       'content_top1':total('top1_correct')/tokens,'content_top5':total('top5_correct')/tokens,
       'content_margin':total('margin_sum')/tokens,'first_margin':total('first_margin')/n,
       'argmax_vs_topk1_different':total('argmax_vs_topk1_different'),'gold_tied_best':total('gold_tied_best')}
    assert abs(m['total_micro_ce']-m['first_contribution']-m['rest_contribution']-m['eos_contribution'])<1e-12
    by=defaultdict(list)
    for r in raw:by[r['row_id']].append(r)
    per=[]
    for r in data:
        key=r['id'] if split=='public_dev' else r['record_id'];v=by[key]
        refs=r['answers'] if split=='public_dev' else [r['conversations'][-1]['content']]
        assert [x['index'] for x in v]==list(range(len(refs)))
        ix=[refs.index(s) for s in dict.fromkeys(refs)]
        set_nll=-float(logsumexp([-v[i]['nll']-v[i]['eos_nll'] for i in ix]));assert set_nll>=-1e-5,(key,set_nll)
        per.append({'id':key,'answer_set_nll':set_nll,'answer_set_probability':float(np.exp(-set_nll)),
            'teacher_exact_any':any(v[i]['teacher_exact'] for i in ix),
            'content_macro_ce':float(np.mean([x['nll']/x['tokens'] for x in v]))})
    for k in ['answer_set_nll','answer_set_probability','teacher_exact_any']:m[k]=float(np.mean([r[k] for r in per]))
    return m,per

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    data={'internal_dev':rows(B/'data/dev.jsonl'),'public_dev':rows(B/'data/cmrc_eval.jsonl')}
    records=[];raws={};per={};sources={};missing=[]
    for j in jobs():
        folder=D/'outputs'/j['name'];marker=folder/'COMPLETE.json'
        if not marker.exists():missing.append(j['name']);continue
        complete=read(marker);assert complete['status']=='passed'
        for path,h in complete['files'].items():assert sha(path)==h,path;sources[path]=h
        for split in data:
            raw=rows(folder/f'{split}.jsonl');m,q=summarize(raw,split,data[split]);key=(j['model'],j['arm'],j['seed'],split)
            raws[key]=raw;per[key]=q;records.append({**{k:j[k] for k in ['name','model','arm','seed']},'split':split,**m})
    if not a.partial:assert not missing,missing
    indexed={(r['model'],r['arm'],r['seed'],r['split']):r for r in records}
    contrasts=[];lengths=[];concordance=[]
    metrics=['content_micro_ce','content_macro_ce','total_micro_ce','content_contribution','first_contribution','rest_contribution','eos_contribution',
        'eos_ce','eos_top1','first_ce','first_top1','first_top5','rest_ce','rest_top1','content_top1','content_top5','content_margin','first_margin',
        'answer_set_nll','answer_set_probability','teacher_exact_any']
    for model in sorted({j['model'] for j in jobs()}):
        for control in ['none','hidden_budget']:
            for split in data:
                if not all((model,arm,s,split) in indexed for arm in ['both',control] for s in [42,43,44]):continue
                for metric in metrics:
                    values=[indexed[model,'both',s,split][metric]-indexed[model,control,s,split][metric] for s in [42,43,44]]
                    mean=float(np.mean(values));rad=float(t.ppf(.975,2)*np.std(values,ddof=1)/np.sqrt(3))
                    contrasts.append({'model':model,'control':control,'split':split,'metric':metric,'seed_values':values,'mean':mean,'nominal_seed_ci95':[mean-rad,mean+rad]})
                if split!='public_dev':continue
                for seed in [42,43,44]:
                    l=raws[model,'both',seed,split];r=raws[model,control,seed,split]
                    assert [(x['row_id'],x['index'],x['tokens']) for x in l]==[(x['row_id'],x['index'],x['tokens']) for x in r]
                    token=np.array([x['tokens'] for x in l]);delta=np.array([(x['nll']-y['nll'])/x['tokens'] for x,y in zip(l,r)])
                    macro=float(delta.mean());micro=float(token@delta/token.sum());cov=float(np.mean((token-token.mean())*(delta-delta.mean()))/token.mean())
                    assert abs(micro-macro-cov)<1e-12
                    bins=[]
                    for low,high in [(1,4),(5,8),(9,16),(17,1000000)]:
                        mask=(token>=low)&(token<=high)
                        bins.append({'min_tokens':low,'max_tokens':high,'references':int(mask.sum()),'macro_delta':float(delta[mask].mean()),
                            'micro_delta':float(token[mask]@delta[mask]/token[mask].sum())})
                    lengths.append({'model':model,'control':control,'seed':seed,'macro_delta':macro,'micro_delta':micro,'covariance_term':cov,'bins':bins})
                    lg=read(B/'outputs'/f'{model}_both_hr8_sd{seed}'/'cmrc_scores.json')['per_example']
                    rg=read(B/'outputs'/f'{model}_{control}_hr8_sd{seed}'/'cmrc_scores.json')['per_example']
                    ce=np.array([x['content_macro_ce']-y['content_macro_ce'] for x,y in zip(per[model,'both',seed,split],per[model,control,seed,split])])
                    gd=np.array([x['avg']-y['avg'] for x,y in zip(lg,rg)])
                    em=np.array([x['em']-y['em'] for x,y in zip(lg,rg)])
                    concordance.append({'model':model,'control':control,'seed':seed,'questions':len(ce),'ce_better':int((ce<0).sum()),
                        'em_wins':int((em>0).sum()),'em_losses':int((em<0).sum()),'avg_wins':int((gd>0).sum()),'avg_losses':int((gd<0).sum()),
                        'ce_better_avg_worse':int(((ce<0)&(gd<0)).sum()),'ce_worse_avg_better':int(((ce>0)&(gd>0)).sum())})
    write(D/'RESULTS.json',{'updated_at':now(),'complete':not missing,'completed':24-len(missing),'missing':missing,'records':records,
        'contrasts':contrasts,'length_decomposition':lengths,'generation_concordance':concordance,'source_sha256':sources,
        'scope':'Exploratory diagnostics on previously evaluated checkpoints/data; all nominal intervals descriptive, not a new confirmatory family.'})
    lines=['# 评测分解结果',f'完成 {24-len(missing)}/24 个 checkpoint；增量均为 both 减对照，CE/NLL 负值为改善。','',
        '| 模型 | 对照 | 划分 | 指标 | 三种子均值 | seeds 42 / 43 / 44 | 名义种子 95% CI |','| --- | --- | --- | --- | ---: | --- | --- |']
    for c in contrasts:
        if c['metric'] not in ['content_micro_ce','content_macro_ce','total_micro_ce','first_contribution','rest_contribution','eos_contribution','first_top1','rest_top1','answer_set_nll']:continue
        lines.append(f"| {c['model']} | {c['control']} | {c['split']} | {c['metric']} | {c['mean']:+.7f} | "+' / '.join(f'{v:+.7f}' for v in c['seed_values'])+' | '+str(c['nominal_seed_ci95'])+' |')
    (D/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if contrasts:
        stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=['model','control','split','metric','mean','seed42','seed43','seed44','ci_low','ci_high']);w.writeheader()
        for c in contrasts:w.writerow({**{k:c[k] for k in ['model','control','split','metric','mean']},**dict(zip(['seed42','seed43','seed44'],c['seed_values'])),**dict(zip(['ci_low','ci_high'],c['nominal_seed_ci95']))})
        (D/'contrasts.csv').write_text(stream.getvalue())
    print('Completed',24-len(missing),'/24; diagnostic contrasts',len(contrasts),flush=True)

if __name__=='__main__':main()
