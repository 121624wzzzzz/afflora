import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import HERE, read, sha, write

report=read(HERE/'RESULTS.json');assert report['complete']
cs=report['primary_contrasts'];fig,ax=plt.subplots(figsize=(9.4,5.5))
labels=[]
for i,c in enumerate(cs):
    y=len(cs)-1-i;m=c['mean_effect_pp'];s=c['seed_ci95'];b=c['cluster_bootstrap_ci95']
    ax.plot(s,[y+.09,y+.09],color='#9AA5B1',linewidth=2)
    ax.plot(b,[y-.09,y-.09],color='#2166AC',linewidth=4)
    ax.scatter([m,m],[y+.09,y-.09],color=['#677482','#2166AC'],s=18,zorder=3)
    model='Qwen3-0.6B' if c['model'].startswith('qwen3') else 'Qwen2.5-1.5B'
    control='hidden' if c['control']=='none' else 'matched budget'
    labels.append(f"{c['task'].upper()} | {model} | vs {control}")
ax.set_yticks(range(len(cs)),labels[::-1]);ax.axvline(0,color='#333333',linewidth=1,linestyle='--')
ax.set_xlabel('Both minus control (percentage points)')
ax.set_title('Chinese task transfer: paired incremental effects')
ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color='#9AA5B1',lw=2,label='Training seeds: nominal 95% t interval'),
    Line2D([0],[0],color='#2166AC',lw=4,label='Document clusters: nominal 95% bootstrap')],loc='best',fontsize=8)
fig.text(.01,.012,'CMRC: (EM+F1)/2. C3: mean-token-likelihood accuracy. 3 seeds; cluster intervals conditional on fitted models.\nUnadjusted intervals shown; Bonferroni-adjusted intervals for 8 primary effects are reported in RESULTS.json.',fontsize=7.5)
fig.tight_layout(rect=(0,.07,1,1))
out=HERE/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'paired_increments.png',dpi=200);fig.savefig(out/'paired_increments.pdf');plt.close(fig)
write(out/'sources.json',{'results_sha256':sha(HERE/'RESULTS.json'),'plot_script_sha256':sha(HERE/'plot.py'),
    'outputs':{str(p):sha(p) for p in out.iterdir() if p.suffix in ['.png','.pdf']}})

