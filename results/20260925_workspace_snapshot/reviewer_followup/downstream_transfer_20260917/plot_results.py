"""Presentation only; reads frozen-analysis outputs without changing metrics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import *

def main():
    data=read(HERE/'TEST_ANALYSIS.json')['primary_contrasts']
    fig,ax=plt.subplots(figsize=(8.5,5.2));y=np.arange(len(data))[::-1]
    for i,(r,pos) in enumerate(zip(data,y)):
        color='#167d8d' if r['control']=='hidden' else '#bc6c25'
        ax.errorbar(r['mean'],pos,xerr=[[r['mean']-r['lower']],[r['upper']-r['mean']]],fmt='o',color=color,capsize=4,lw=1.8)
    ax.set_yticks(y,[(('ANLI R1' if r['task']=='anli_r1' else 'WikiSQL')+' / '+('0.6B' if r['model']=='qwen3_06b_base' else '1.5B')+' / vs '+('LoRA' if r['control']=='hidden' else 'matched budget')) for r in data])
    ax.axvline(0,color='#444',ls='--',lw=1);ax.set_xlabel('LoRA + A-LoRA minus control (percentage points)')
    ax.set_title('Held-out downstream transfer: five paired seeds')
    ax.text(0,-.18,'Bonferroni-adjusted 95% seed-t intervals across 8 primary contrasts.\nANLI: label accuracy; WikiSQL: execution accuracy on 1,024 fixed test examples.',transform=ax.transAxes,fontsize=9)
    ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(rect=(0,.07,1,1));out=HERE/'figures';out.mkdir(exist_ok=True)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'heldout_effects.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
