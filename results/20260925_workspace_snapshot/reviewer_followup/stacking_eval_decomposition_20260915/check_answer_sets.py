"""Independent direct-probability check of the logsumexp aggregation."""
import math
from collections import defaultdict
from shared import D,B,read,rows,sha,write,now,jobs

d=read(D/'RESULTS.json');assert d['complete']
idx={(r['name'],r['split']):r for r in d['records']}
data=rows(B/'data/cmrc_eval.jsonl');records=[];max_error=0.;sources={str(D/'RESULTS.json'):sha(D/'RESULTS.json')}
for j in jobs():
    p=D/'outputs'/j['name']/'public_dev.jsonl';sources[str(p)]=sha(p)
    by=defaultdict(list)
    for r in rows(p):by[r['row_id']].append(r)
    ps=[];nlls=[];groups=defaultdict(list);sensitivity=defaultdict(list);ranges=[]
    for q in data:
        unique={};duplicates=defaultdict(list)
        for answer,r in zip(q['answers'],by[q['id']]):
            value=math.exp(-r['nll']-r['eos_nll'])
            if answer not in unique:unique[answer]=value
            duplicates[answer].append(value)
        probability=math.fsum(unique.values());assert 0<probability<=1+1e-6
        ps.append(probability);nlls.append(-math.log(probability));groups[len(unique)].append(probability)
        low=math.fsum(min(v) for v in duplicates.values());high=math.fsum(max(v) for v in duplicates.values())
        avg=math.fsum(math.fsum(v)/len(v) for v in duplicates.values())
        sensitivity['min'].append(low);sensitivity['max'].append(high);sensitivity['mean'].append(avg)
        ranges.append(high-low)
    m=idx[j['name'],'public_dev']
    e=max(abs(math.fsum(ps)/len(ps)-m['answer_set_probability']),abs(math.fsum(nlls)/len(nlls)-m['answer_set_nll']))
    assert e<1e-12;max_error=max(max_error,e)
    records.append({**{k:j[k] for k in ['name','model','arm','seed']},'by_unique_references':{str(k):{'n':len(v),'mean_probability':math.fsum(v)/len(v)} for k,v in groups.items()},
        'duplicate_choice_mean_probabilities':{k:math.fsum(v)/len(v) for k,v in sensitivity.items()},'max_question_probability_range':max(ranges),
        'mean_question_probability_range':math.fsum(ranges)/len(ranges),'questions_with_different_duplicates':sum(v>1e-12 for v in ranges)})
write(D/'ANSWER_SET_CHECK.json',{'status':'passed','checked_at':now(),'public_questions_checked':len(jobs())*len(data),
    'maximum_aggregate_error':max_error,'probability_bounds_checked':True,'duplicates_counted_once':True,'records':records,'source_sha256':sources,
    'scope':'Direct FP64 exp/sum/log check independent of scipy.logsumexp. Duplicate probabilities can differ across original BF16 batches; all min/max/mean selection sensitivity retained. Reference-count and numerical sensitivity diagnostics follow the aggregate mass result, not confirmation.'})
print('Answer-set check passed; max error',max_error,flush=True)
