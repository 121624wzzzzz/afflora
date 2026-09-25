from common import *
import collections

def main():
 groups=collections.defaultdict(list)
 for ap in (HERE/'audits').glob('*.json'):
  s=read(HERE/'checkpoints'/ap.stem/'spec.json')
  if s['stage']!='confirmation' or s['smoke']:continue
  e=read(HERE/'evaluations'/ap.stem/'test/SUMMARY.json');groups[s['model'],s['task'],s['method']].append(e)
 out=[]
 for (m,t,method),rs in sorted(groups.items()):
  out.append(dict(model=m,task=t,method=method,n=len(rs),metrics={k:sum(r[k] for r in rs)/len(rs) for k in ['primary','strict_accuracy','macro_f1','valid_pct','strict_valid_pct','capped_pct']},parameters=dict(total=rs[0]['spec']['expected_total_parameters'],boundary=rs[0]['spec']['expected_boundary_parameters'])))
 write(HERE/'SECONDARY_METRICS.json',dict(at=now(),groups=out,note='Content accuracy is prespecified primary; strict metrics are diagnostics; n<5 is incomplete.'))
if __name__=='__main__':main()
