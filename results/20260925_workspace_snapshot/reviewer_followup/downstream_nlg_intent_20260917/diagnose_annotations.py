"""Descriptive annotation/metric limits, no influence on fitting or primary selection."""
from common import *
from scoring import slots,CORRECTED,ORIGINAL

def main():
    result={}
    for split in ['train','dev','test']:
        data=rows(HERE/f'data/e2e_clean_{split}.jsonl');out={}
        for name,impl in [('original',ORIGINAL),('corrected',CORRECTED)]:
            n=ok=anyok=0;counts={k:0 for k in ['added','missing','wrong','repeated','gold_slots']}
            for r in data:
                vals=[slots(ref,r,impl) for ref in r['references']];n+=len(vals);ok+=sum(x['exact'] for x in vals);anyok+=any(x['exact'] for x in vals)
                for v in vals:
                    for k in counts:counts[k]+=v[k]
            out[name]={'references':n,'exact_references':ok,'exact_reference_pct':100*ok/n,'mrs_with_any_exact_reference':anyok,'mrs':len(data),'error_counts':counts}
        result[split]=out
    write(HERE/'ANNOTATION_DIAGNOSTICS.json',{'at':now(),'split_results':result,'note':'A heuristic consistency check on provided references, not a human annotation accuracy estimate. Never used to remove examples or choose outputs.'});print(canonical(result))
if __name__=='__main__':main()
