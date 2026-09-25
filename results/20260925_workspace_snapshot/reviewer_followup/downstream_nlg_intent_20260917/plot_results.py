"""Plots consume frozen statistical outputs; no inference or metric selection."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import *

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
a=read(HERE/'TEST_ANALYSIS.json');fig,ax=plt.subplots(figsize=(10,6));y=0;labels=[];ticks=[]
for c in a['conditions']:
    for x in c['comparisons']:
        mean=x['mean'];lo,hi=x['bonferroni8_ci'];color='#147D92' if lo>0 else '#606D80'
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=4,lw=2)
        task='Banking77 (accuracy pp)' if c['task']=='banking77' else 'E2E (BLEU points)'
        model='0.6B' if c['model']=='qwen3_06b_base' else '1.5B';control='LoRA' if x['control']=='hidden' else 'equal-budget LoRA'
        labels.append(task+' / '+model+' / vs '+control);ticks.append(y);y+=1
    y+=.4
ax.set_yticks(ticks,labels);ax.invert_yaxis();ax.axvline(0,color='#999999',linestyle='--',lw=1)
ax.set_xlabel('LoRA + A-LoRA minus control');ax.set_title('Held-out effects across five paired training seeds',loc='left',weight='bold',pad=15)
ax.grid(axis='x',alpha=.18);fig.text(.02,.025,'Intervals: paired seed t, Bonferroni correction for 8 pre-specified contrasts. Fixed test splits; no task-specific tuning.',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.055,1,1]);out=HERE/'figures';out.mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(out/f'heldout_effects.{ext}',dpi=180,bbox_inches='tight')
write(out/'PLOT_SOURCE.json',{'at':now(),'source_sha256':sha(HERE/'TEST_ANALYSIS.json'),'files':{str(p.name):sha(p) for p in out.glob('heldout_effects.*')}})
