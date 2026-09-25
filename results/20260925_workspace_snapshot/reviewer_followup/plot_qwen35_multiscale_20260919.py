"""Scientific forest plot of the completed, sealed multiscale comparison."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent/'qwen35_multiscale_20260919'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=json.loads((ROOT/'RESULTS.json').read_text());assert a['status']=='passed'
 for source in a['sources']:
  p=Path(source['path']);assert sha(p/'SEAL_MANIFEST.json')==source['manifest_sha256']
  manifest=json.loads((p/'SEAL_MANIFEST.json').read_text())
  assert sha(p/'ANALYSIS.json')==manifest['files']['ANALYSIS.json']['sha256']
 sizes=['0.8B','2B','4B','9B'];fig,axes=plt.subplots(1,2,figsize=(11.4,5.4),sharey=True)
 styles={'hidden':('#526176','s',-.13),'hidden_budget':('#166c80','o',.13)}
 for ax,task in zip(axes,['wikisql','trec50']):
  for c in a['comparisons']:
   if c['task']!=task:continue
   color,marker,offset=styles[c['control']];i=sizes.index(c['size']);lo,hi=c['corrected_ci95'];y=i+offset
   ax.hlines(y,lo,hi,color=color,lw=1.7,linestyles='dashed' if c['historical'] else 'solid')
   ax.plot([lo,hi],[y,y],marker='|',color=color,linestyle='none',ms=7)
   ax.plot(c['mean_pp'],y,marker,color=color,ms=5)
  ax.axvline(0,color='#999999',ls=':',lw=1.2);ax.grid(axis='x',alpha=.15)
  ax.set_title('WikiSQL execution accuracy' if task=='wikisql' else 'TREC50 candidate accuracy',fontsize=11)
  ax.set_xlabel('H+E+U minus control (percentage points)')
  ax.set_yticks(range(4),['0.8B','2B','4B (historical)','9B'])
  ax.tick_params(axis='y',labelleft=True);ax.spines[['top','right']].set_visible(False)
 axes[0].invert_yaxis();axes[0].set_ylabel('Qwen3.5 Base text decoder')
 handles=[Line2D([0],[0],color=styles[k][0],marker=styles[k][1],label=label) for k,label in [('hidden','vs. H'),('hidden_budget','vs. exact-budget H')]]
 fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(.5,.07),frameon=False)
 fig.suptitle('Boundary-adapter stacking across Qwen3.5 sizes',fontsize=13)
 fig.text(.5,.02,'Five paired training seeds; adjusted 95% t intervals. New 0.8B/2B/9B: family 12. Historical 4B: family 4.\nIntervals condition on fixed data and selected configurations; no joint data/tuning uncertainty.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.14,1,.94])
 for ext in ['png','pdf']:fig.savefig(ROOT/f'stacking_by_size.{ext}',dpi=200)
 plt.close(fig)
 p=ROOT/'RESULTS_ZH.md';text=p.read_text();assert 'stacking_by_size.png' not in text
 text+='\n![四个尺寸的叠加增量及校正种子区间](stacking_by_size.png)\n'
 p.write_text(text)
 print('Completed-data multiscale plot written.',flush=True)
if __name__=='__main__':main()
