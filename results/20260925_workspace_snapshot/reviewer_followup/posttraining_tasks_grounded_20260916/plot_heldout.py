from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    data=read(HERE/'HELDOUT_ANALYSIS.json');rows_=data['primary_contrasts']
    fig,ax=plt.subplots(figsize=(10,6));labels=[]
    for i,r in enumerate(rows_):
        color='#236b9f' if r['control']=='hidden' else '#c27827'
        ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['lower']],[r['upper']-r['mean']]],fmt='o',color=color,capsize=4)
        model='Qwen3 0.6B Base' if r['model'].startswith('qwen3_') else 'Qwen2.5 1.5B Base'
        task='Tool calls' if r['task']=='toolace' else 'Entity spans'
        control='LoRA' if r['control']=='hidden' else 'equal-parameter LoRA'
        labels.append(f'{task} / {model}\nvs {control}')
    ax.axvline(0,color='#666666',linewidth=1);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    ax.set_xlabel('LoRA + input/output A-LoRA improvement (score points)')
    ax.set_title('Held-out evaluation: additive content effects',loc='left')
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.text(.02,.015,'95% seed-t intervals, Bonferroni family 8. 2,048 train; tool test n=710, NER public-dev n=1,343. One shared LR.',fontsize=9)
    fig.tight_layout(rect=[0,.045,1,1]);(HERE/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(HERE/'figures'/f'heldout_additive_effects.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
