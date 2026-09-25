"""Presentation only: visualize the frozen analysis without changing its rules."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from run import specs

def main():
    results=read(HERE/'RESULTS.json');models=list(results['aggregates'])
    labels={'none':'LoRA','both':'LoRA + boundary','interior':'LoRA + interior','hidden_budget':'LoRA + rank budget'}
    model_labels={'qwen3_06b_chat':'Qwen3-0.6B','qwen25_15b_chat':'Qwen2.5-1.5B-Instruct'}
    colors=['#707070','#2563a6','#ad6e21','#418769']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'height_ratios':[1.25,1]})
    for j,model in enumerate(models):
        ax=axes[0,j];data={}
        for s in specs('confirmation'):
            if s['model']==model:data[s['arm'],s['seed']]=read(Path(s['checkpoint'])/'test_metrics.json')['primary']['accuracy']
        for seed in range(2002,2007):ax.plot(range(4),[data[a,seed] for a in ARMS],color='#bbbbbb',lw=.7,alpha=.55,zorder=1)
        for i,arm in enumerate(ARMS):
            yy=[data[arm,s] for s in range(2002,2007)]
            ax.scatter([i]*5,yy,color=colors[i],s=26,zorder=3)
            ax.plot([i-.17,i+.17],[np.mean(yy)]*2,color=colors[i],lw=3,zorder=4)
        ax.set_xticks(range(4),[labels[a] for a in ARMS],rotation=17,ha='right')
        ax.set_ylabel('Test accuracy (%)');ax.set_title(model_labels[model]);ax.grid(axis='y',alpha=.16)
        ax=axes[1,j]
        cc=[c for c in results['comparisons'] if c['model']==model]
        for i,c in enumerate(cc):
            x=c['primary_adjusted'];mean=x['mean'];lo,hi=x['ci95']
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#2563a6',capsize=4)
            ax.annotate(f'{mean:+.2f}',(mean,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.axvline(0,color='#888888',lw=1,ls='--')
        ax.set_yticks(range(3),['Boundary minus LoRA','Boundary minus rank budget','Boundary minus interior'])
        ax.invert_yaxis();ax.set_ylim(2.6,-.6);ax.set_xlabel('Accuracy difference (percentage points)');ax.grid(axis='x',alpha=.16)
    fig.suptitle('SciQ: additive accuracy and adapter placement',fontsize=15,y=.995)
    fig.text(.5,.005,'5 fresh seeds; 998 unambiguous test questions. Bottom: 95% intervals adjusted across 6 primary comparisons.\nLearning rate selected independently per arm using validation only. Intervals across seeds condition on this test set.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.98]);out=HERE/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'sciq_placement.png',dpi=180,bbox_inches='tight');fig.savefig(out/'sciq_placement.pdf',bbox_inches='tight')

if __name__=='__main__':main()
