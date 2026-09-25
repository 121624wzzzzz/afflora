"""Descriptive stratification and pre-chosen qualitative examples; no model selection."""
from common import *
from scoring import aggregate
import numpy as np

def main():
    out={'at':now(),'note':'Post-fit descriptive diagnostics, not additional confirmatory tests. All examples retained in primary.'}
    data=rows(HERE/'data/e2e_clean_test.jsonl');multi={r['id']:len({k for k,v in r['attributes']})<len(r['attributes']) for r in data}
    panel=sorted(data,key=lambda r:htext('v4-fixed-qualitative-panel|'+r['id']))[:10];lines=['# Fixed qualitative panel','', 'Ten test MRs selected by a fixed hash; seed8100 only, no best-output selection.','']
    for m in MODELS:
        out[m]={};byarm={}
        for arm in ['base','hidden','hidden_budget','hidden_both']:
            names=[f'e2e_clean_{m}_base'] if arm=='base' else [f'e2e_clean_{m}_{arm}_s{s}' for s in range(8100,8105)];rs=[rows(HERE/'evaluations'/name/'test/responses.jsonl') for name in names];byarm[arm]={r['id']:r for r in rs[0]}
            strata={}
            for subset in ['all','single_value','multiple_values']:
                ss=[aggregate([r for r in records if subset=='all' or multi[r['id']]==(subset=='multiple_values')],'e2e_clean') for records in rs]
                strata[subset]={k:float(np.mean([s[k] for s in ss])) for k in ss[0]};strata[subset]['non_repetition_ser_pct']=sum(strata[subset]['slots_'+k+'_pct'] for k in ['added','missing','wrong'])
            out[m][arm]=strata
        for r in panel:
            lines.extend(['## '+m+' / '+r['id'],'','Attributes: '+canonical(r['attributes']),'','References: '+canonical(r['references']),''])
            for arm in ['base','hidden','hidden_budget','hidden_both']:lines.extend(['**'+arm+'**: '+byarm[arm][r['id']]['text'],''])
    write(HERE/'CONTENT_DIAGNOSTICS.json',out);(HERE/'QUALITATIVE_PANEL.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
