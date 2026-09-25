import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

p=read(HERE/'PREDICTION_ANALYSIS.json');a=read(HERE/'ABLATION_ANALYSIS.json')
models=['qwen3_06b_chat','qwen25_15b_chat'];labels=['Qwen3-0.6B','Qwen2.5-1.5B']
comp=[next(c for c in p['comparisons'] if c['model']==m and c['control']=='hidden_budget') for m in models]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(left=.085,right=.975,top=.86,bottom=.19,hspace=.62,wspace=.32)
fig.suptitle('SciQ: why bilateral A-LoRA gives little net accuracy gain',fontsize=16,y=.975)
fig.text(.5,.93,'Post hoc diagnostics | 998 questions, 4 cyclic option orders, 5 training seeds per model',ha='center',color='#555555')
x=np.arange(2)
for off,key,label,color in [(-.18,'all_rotations_corrected','Wrong → correct','#238b74'),(.18,'all_rotations_regressed','Correct → wrong','#cb624b')]:
    bars=ax[0,0].bar(x+off,[c['totals'][key] for c in comp],.34,label=label,color=color)
    ax[0,0].bar_label(bars,padding=3)
ax[0,0].set(xticks=x,xticklabels=labels,ylabel='Repeated evaluation occurrences',ylim=(0,350),title='A. Corrections and regressions nearly cancel\nCompared with equal-budget hidden LoRA')
ax[0,0].legend(frameon=False,fontsize=9)
ax[0,1].axvline(0,color='#aaaaaa',lw=1)
for j,m in enumerate(models):
    records=[('Full − equal-budget LoRA',comp[j]['rotation_gain']),
             ('Full − both maps disabled',a['models'][m]['paired_effects']['full_minus_both_off']['rotations']['accuracy'])]
    for i,(name,r) in enumerate(records):
        y=j*3+i;mean=r['mean'];lo,hi=r['ci95']
        ax[0,1].errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color=['#2974a3','#875baf'][j])
ax[0,1].set(yticks=[0,1,3,4],yticklabels=['Q3: full − budget','Q3: full − maps off','Q2.5: full − budget','Q2.5: full − maps off'],
    xlabel='Accuracy difference (percentage points)',title='B. Small differences with broad seed intervals',ylim=(4.7,-.7))
for off,key,label,color in [(-.18,'disagreed','Changed prediction','#bc6641'),(.18,'agreed','Same prediction','#778b9e')]:
    bars=ax[1,0].bar(x+off,[c['baseline_top_two_margin'][key]['median'] for c in comp],.34,label=label,color=color)
    ax[1,0].bar_label(bars,fmt='%.2f',padding=3)
ax[1,0].set(xticks=x,xticklabels=labels,ylabel='Median top-two score gap (logits)',ylim=(0,7.5),title='C. Changed answers start near a decision tie')
ax[1,0].legend(frameon=False,fontsize=9)
bottom=np.zeros(2)
for key,label,color in [('semantic_stable_fraction','Stable across semantic-option rotations','#397f9d'),
        ('global_letter_fraction','Global answer-letter offset','#dba04b'),('order_dependent_remainder_fraction','Remaining order-sensitive change','#9b8bab')]:
    v=np.array([c['logit_change_decomposition_mean'][key]*100 for c in comp])
    bars=ax[1,1].bar(x,v,bottom=bottom,width=.48,label=label,color=color)
    for i,y in enumerate(v):
        if y>7:ax[1,1].text(i,bottom[i]+y/2,f'{y:.1f}%',ha='center',va='center',color='white',fontsize=9)
    bottom+=v
ax[1,1].set(xticks=x,xticklabels=labels,ylabel='Centered score-change energy (%)',ylim=(0,105),title='D. Global letter preference explains only part')
ax[1,1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13))
fig.text(.085,.025,'Intervals: nominal paired-seed 95% t intervals; not new confirmatory tests.\nMaps-off keeps jointly trained hidden LoRA fixed. Rotation-stable score changes need not favor the correct answer.',fontsize=9,color='#555555')
(HERE/'figures').mkdir(exist_ok=True)
fig.savefig(HERE/'figures'/'mechanism_diagnosis.png',dpi=180)
fig.savefig(HERE/'figures'/'mechanism_diagnosis.pdf')
