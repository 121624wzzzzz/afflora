"""Dev-only sensitivity of LR selection; diagnostic, never changes frozen selection."""
from common import *
import itertools

def main():
 p=HERE/'SELECTION_FROZEN.json'
 if not p.exists():return
 result=[]
 for row in read(p)['selection']:
  scores={float(k):v for k,v in row['dev_scores'].items()};assert all(len(v)==3 for v in scores.values())
  def choose(indices):return max(scores,key=lambda r:(sum(scores[r][i] for i in indices)/len(indices),-abs(r-1),-r))
  ranks=sorted(scores,key=lambda r:sum(scores[r])/3,reverse=True)
  result.append(dict(architecture=row['architecture'],selected=row['ratio'],dev_means={r:sum(v)/3 for r,v in scores.items()},best_minus_second=sum(scores[ranks[0]])/3-sum(scores[ranks[1]])/3,single_seed_choices=[choose([i]) for i in range(3)],leave_one_seed_out_choices=[choose([i for i in range(3) if i!=j]) for j in range(3)],selected_at_grid_edge=row['ratio'] in [min(scores),max(scores)]))
 write(HERE/'SELECTION_DIAGNOSTICS.json',dict(at=now(),note='Dev-only stability diagnostics; no test feedback; does not alter frozen choices.',results=result))
if __name__=='__main__':main()
