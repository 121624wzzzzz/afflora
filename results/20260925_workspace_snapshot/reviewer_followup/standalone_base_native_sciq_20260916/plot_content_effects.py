"""Presentation of predeclared content effects; does not alter analysis."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import *

def main():
    r=read(HERE/'RESULTS.json');assert read(HERE/'FINAL_AUDIT.json')['status']=='passed'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'height_ratios':[1,1]})
    fig.subplots_adjust(left=.13,right=.98,bottom=.21,top=.85,hspace=.43,wspace=.33)
    fig.suptitle('Base task adaptation: content accuracy and uncertainty',fontsize=16,y=.98)
    fig.text(.5,.93,'SciQ | 998 canonical questions | 5 fresh training seeds | native completion + EOS',ha='center',color='#555')
    colors={'input':'#297f9c','output':'#885ba6','hidden_r8':'#bf7849'}
    labels={'qwen3_06b_base':'Qwen3-0.6B-Base','qwen25_15b_base':'Qwen2.5-1.5B (Base)'}
    names={'input':'Input A-LoRA','output':'Output A-LoRA','hidden_r8':'Hidden LoRA r8'}
    for col,(model,o) in enumerate(r['models'].items()):
        base=o['base']['primary']['accuracy'];a=axes[0,col];b=axes[1,col]
        a.axhline(base,color='#888',ls=':',lw=1.2,label=f'Unadapted Base {base:.2f}%')
        all_values=[base]
        for i,arm in enumerate(ARMS):
            c=next(x for x in r['primary_contrasts'] if x['model']==model and x['arm']==arm and x['metric']=='candidate_accuracy')
            values=np.array(c['effect_pp']['seeds'])+base;all_values.extend(values)
            a.scatter(i+np.linspace(-.14,.14,5),values,color=colors[arm],s=23,alpha=.75)
            mean=o['arms'][arm]['primary']['accuracy'];a.scatter(i,mean,marker='D',s=45,color=colors[arm],edgecolors='black',linewidths=.5,zorder=4)
            a.annotate(f'{mean:.2f}%',(i,float(values.max())),xytext=(0,9),textcoords='offset points',ha='center',color=colors[arm])
            effect=c['effect_pp']['mean']
            for offset,key,color,marker in [(-.10,'family12_ci95',colors[arm],'o'),(.10,'question', '#555','s')]:
                low,high=c['effect_pp'][key] if key!='question' else c['question_bootstrap95_conditioned_on_fitted_seeds']
                y=i+offset;b.plot([low,high],[y,y],color=color,lw=2);b.plot([low,high],[y,y],linestyle='',marker='|',color=color)
                b.scatter(effect,y,color=color,marker=marker,s=25,zorder=3)
        a.set(title=labels[model],ylabel='Candidate accuracy (%)',xticks=range(3),xticklabels=[names[arm]+'\n'+f"{o['arms'][arm]['parameters']:,} params" for arm in ARMS],ylim=(min(all_values)-1.0,max(all_values)+1.3))
        a.tick_params(axis='x',labelsize=8);a.legend(frameon=False,loc='lower right',fontsize=8)
        b.axvline(0,color='#888',ls=':',lw=1.2)
        b.set(yticks=range(3),yticklabels=[names[arm] for arm in ARMS],xlabel='Accuracy change from Base (percentage points)',ylim=(2.5,-.5))
        b.tick_params(axis='y',labelsize=9);b.set_title('Content gain: two sources of uncertainty',fontsize=11)
    fig.text(.13,.045,'Top: dots are seeds; diamonds are means. Accuracy axis is zoomed. Hidden LoRA has a much larger parameter budget.\nBottom: colored intervals = seed t intervals, Bonferroni family12; gray = nominal95% question bootstrap conditional on fitted models.\nStrict-format generation is reported separately, so learning to stop after a letter is not displayed as a content-accuracy gain.',fontsize=9,color='#555')
    (HERE/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(HERE/f'figures/base_content_effects.{ext}',dpi=180)
    print(HERE/'figures/base_content_effects.png')
if __name__=='__main__':main()
