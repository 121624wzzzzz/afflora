"""Presentation only: all primary inference comes from frozen analyze.py."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import *

COLORS = {'hidden':'#4376a5','hidden_budget':'#bd752d','hidden_both':'#187c69'}
LABELS = {'hidden':'LoRA','hidden_budget':'Budget LoRA*','hidden_both':'LoRA + A-LoRA'}

def save(fig, name):
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def main():
    a=read(HERE/'TEST_ANALYSIS.json')
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.6))
    for task,ax in zip(TASKS,axes):
        cs=[next(c for c in a['historical_15b_anchors'] if c['task']==task)]
        cs += [next(c for c in a['conditions'] if c['task']==task and c['model']==m) for m in MODELS]
        for arm in COLORS:
            means=[c['means'][arm] for c in cs]
            stds=[np.std(c['values'][arm],ddof=1) for c in cs]
            ax.errorbar([1.5,3,7],means,yerr=stds,color=COLORS[arm],label=LABELS[arm],fmt='o-',capsize=3,lw=1.5)
        ax.set_xticks([1.5,3,7],['1.5B\nHistorical','3B\nNew','7B\nNew'])
        ax.set_title('CLUENER' if task=='cluener' else 'WikiSQL')
        ax.set_ylabel('Span micro-F1' if task=='cluener' else 'Execution accuracy (%)')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=9)
    fig.suptitle('Qwen2.5 Base: fixed data and hyperparameters, five seeds')
    fig.text(.03,.015,'Error bars: seed SD, not significance intervals. *7B control has 512 more trainable parameters.\nPreviously positive tasks selected; 3B has tied embeddings and 7B does not. Size trends are descriptive.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94));save(fig,'model_size_scores')

    records=[(c,x) for c in a['conditions'] for x in c['comparisons']]
    fig,ax=plt.subplots(figsize=(9.2,5.5));ys=np.arange(len(records))[::-1]
    labels=[]
    for y,(c,x) in zip(ys,records):
        lo,hi=x['bonferroni8_ci'];v=x['mean']
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=COLORS[x['control']],lw=1.6)
        task='CLUENER' if c['task']=='cluener' else 'WikiSQL'
        size='3B' if c['model']=='qwen25_3b_base' else '7B'
        control='ordinary LoRA' if x['control']=='hidden' else 'budget LoRA'+(' (+512)' if size=='7B' else ' (exact)')
        labels.append(f'{task} / {size} / vs {control}')
    ax.set_yticks(ys,labels);ax.axvline(0,ls='--',color='#555',lw=1)
    ax.set_xlabel('LoRA + A-LoRA minus control (F1 points / percentage points)')
    ax.set_title('New 3B / 7B contrasts: five paired seeds')
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.text(.03,.02,'Two-sided 95% seed-t intervals, Bonferroni adjustment across 8 pre-specified contrasts.\nConditional on fixed public test examples and selected tasks; historical 1.5B results are outside this family.',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,1));save(fig,'new_model_effects')

if __name__=='__main__':main()
