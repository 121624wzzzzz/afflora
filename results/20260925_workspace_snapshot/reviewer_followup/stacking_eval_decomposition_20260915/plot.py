import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from shared import D,read,sha,write

d=read(D/'RESULTS.json');assert d['complete']
idx={(r['model'],r['control'],r['split'],r['metric']):r for r in d['contrasts']}
keys=[(m,c) for m in ['qwen3_06b_chat','qwen25_15b_chat'] for c in ['none','hidden_budget']]
labels=[('0.6B' if m.startswith('qwen3') else '1.5B')+' vs '+('hidden' if c=='none' else 'budget') for m,c in keys]
fig,axes=plt.subplots(1,3,figsize=(15,5.3))
for i,(m,c) in enumerate(keys):
    y=3-i
    for offset,metric,color in [(.10,'content_macro_ce','#CB6E28'),(-.10,'content_micro_ce','#2166AC')]:
        r=idx[m,c,'public_dev',metric];axes[0].plot(r['nominal_seed_ci95'],[y+offset]*2,color=color,lw=2)
        axes[0].scatter(r['mean'],y+offset,color=color,s=20,zorder=3)
axes[0].set_title('Public dev: answer-content CE')
axes[0].legend([Line2D([0],[0],color=c,lw=3) for c in ['#CB6E28','#2166AC']],['Question weighted','Token weighted'],fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2)
for ax,split,title in zip(axes[1:],['internal_dev','public_dev'],['Internal dev: total CE components','Public dev: total CE components']):
    for i,(m,c) in enumerate(keys):
        y=3-i;pos=neg=0.
        for metric,color in [('first_contribution','#2166AC'),('rest_contribution','#70A6B5'),('eos_contribution','#CB6E28')]:
            v=idx[m,c,split,metric]['mean'];left=pos if v>=0 else neg
            ax.barh(y,v,left=left,color=color,height=.5)
            if v>=0:pos+=v
            else:neg+=v
        ax.scatter(idx[m,c,split,'total_micro_ce']['mean'],y,color='black',s=23,zorder=4)
    ax.set_title(title)
    ax.legend([Line2D([0],[0],color=c,lw=5) for c in ['#2166AC','#70A6B5','#CB6E28']]+[Line2D([0],[0],marker='o',color='black',lw=0)],
        ['First token','Later tokens','End token','Total'],fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2)
for ax in axes:
    ax.set_yticks(range(4),labels[::-1]);ax.axvline(0,color='#555555',ls='--',lw=1)
    ax.set_xlabel('Both minus control (lower is better)');ax.spines[['top','right']].set_visible(False)
    ax.ticklabel_format(style='sci',axis='x',scilimits=(-3,-3))
fig.text(.01,.01,'Same 24 CMRC-task checkpoints; all 3 seeds. Left: descriptive nominal 95% seed t intervals. Right panels: mean additive contributions.\nExploratory decomposition on previously evaluated data; official EM/F1 and earlier conclusions are retained.',fontsize=9)
fig.tight_layout(rect=(0,.1,1,1));out=D/'figures';out.mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/f'evaluation_decomposition.{ext}',dpi=200)
plt.close(fig)
write(out/'sources.json',{'results_sha256':sha(D/'RESULTS.json'),'plot_sha256':sha(D/'plot.py'),
    'outputs':{str(p):sha(p) for p in out.iterdir() if p.suffix in ['.png','.pdf']}})
