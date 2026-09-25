import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from common import HERE,read,sha,write

d=read(HERE/'RESULTS.json');assert d['complete']
fig,axes=plt.subplots(1,2,figsize=(11,4.1),gridspec_kw={'width_ratios':[1,1.3]})
arms=['none','output','both','hidden_budget'];colors=['#64748B','#7CA8C5','#2166AC','#A9B7C0']
for i,model in enumerate(['qwen3_06b_chat','qwen25_15b_chat']):
    for j,arm in enumerate(arms):
        vals=[100*r['avg'] for r in d['records'] if r['model']==model and r['arm']==arm]
        x=i*5+j;axes[0].bar(x,np.mean(vals),color=colors[j],width=.8);axes[0].scatter([x-.15,x,x+.15],vals,s=12,color='#111111',zorder=4)
axes[0].set_xticks([1.5,6.5],['Qwen3-0.6B','Qwen2.5-1.5B']);axes[0].set_ylabel('CMRC (EM + F1) / 2 (%)')
axes[0].set_title('Task-specific SFT: all three seeds')
axes[0].legend([Line2D([0],[0],color=c,lw=5) for c in colors],['Hidden','+ output','+ both','Matched budget'],fontsize=8,loc='lower right')
cs=d['primary_contrasts'];labels=[]
for i,c in enumerate(cs):
    y=len(cs)-1-i;m=c['mean_effect_pp'];axes[1].plot(c['seed_ci95'],[y+.09,y+.09],color='#9AA5B1',lw=2)
    axes[1].plot(c['cluster_bootstrap_ci95'],[y-.09,y-.09],color='#2166AC',lw=4)
    axes[1].scatter([m,m],[y+.09,y-.09],s=18,color=['#677482','#2166AC'],zorder=3)
    labels.append(('0.6B' if c['model'].startswith('qwen3') else '1.5B')+' vs '+('hidden' if c['control']=='none' else 'budget'))
axes[1].set_yticks(range(4),labels[::-1]);axes[1].axvline(0,color='#333333',ls='--',lw=1)
axes[1].set_xlabel('Both minus control (percentage points)');axes[1].set_title('Paired incremental effects')
axes[1].legend([Line2D([0],[0],color='#9AA5B1',lw=2),Line2D([0],[0],color='#2166AC',lw=4)],['Seed t interval','Document bootstrap'],fontsize=8,loc='best')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.text(.01,.01,'Nominal 95% intervals. Bootstrap conditions on 3 fitted models; family=4 adjusted intervals are in RESULTS.json.\nCMRC public dev (previously evaluated for transfer); fresh task-specific SFT on a disjoint training split.',fontsize=7.5)
fig.tight_layout(rect=(0,.09,1,1));out=HERE/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'cmrc_task_sft.png',dpi=200);fig.savefig(out/'cmrc_task_sft.pdf');plt.close(fig)
write(out/'sources.json',{'results_sha256':sha(HERE/'RESULTS.json'),'plot_script_sha256':sha(HERE/'plot.py'),
    'outputs':{str(p):sha(p) for p in out.iterdir() if p.suffix in ['.png','.pdf']}})
