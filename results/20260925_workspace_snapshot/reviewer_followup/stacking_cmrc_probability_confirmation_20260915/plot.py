import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import HERE,read

def main():
    r=read(HERE/'RESULTS.json');assert r['complete']
    pairs=[('qwen3_06b_chat','none'),('qwen3_06b_chat','hidden_budget'),('qwen25_15b_chat','none'),('qwen25_15b_chat','hidden_budget')]
    labels=['Qwen3-0.6B vs LoRA','Qwen3-0.6B vs equal budget','Qwen2.5-1.5B vs LoRA','Qwen2.5-1.5B vs equal budget']
    panels=[('answer_set_probability',100,'Complete-reference probability Δ (pp)','seed_bonferroni_family4'),
        ('answer_set_nll',1,'Answer-set NLL Δ (nats)','seed_nominal95'),('avg',100,'Generation (EM + F1)/2 Δ (pp)','seed_nominal95')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),sharey=True)
    for ax,(metric,scale,title,interval) in zip(axes,panels):
        for index,(model,control) in enumerate(pairs):
            c=next(x for x in r['contrasts'] if x['model']==model and x['control']==control and x['split']=='public' and x['metric']==metric)
            v=c[interval];y=3-index;color='#2468A0' if index<2 else '#BF652B'
            lo,hi=[scale*x for x in v['ci']];mean=scale*v['mean']
            ax.plot([lo,hi],[y,y],color=color,lw=2);ax.scatter([mean],[y],s=38,color=color,zorder=4)
            ax.scatter([scale*x for x in v['per_seed'].values()],y+np.linspace(-.14,.14,5),marker='|',s=70,alpha=.5,color=color)
        ax.axvline(0,ls='--',lw=1,color='#777777');ax.set_xlabel(title);ax.set_yticks(range(3,-1,-1),labels)
        ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0);ax.set_ylim(-.5,3.5)
        ax.set_title('Primary: corrected intervals' if interval=='seed_bonferroni_family4' else 'Supporting: nominal intervals',fontsize=11)
    fig.suptitle('A-LoRA stacked on hidden LoRA: five fresh training seeds',fontsize=14,y=.97)
    fig.text(.5,.025,'Dots: paired mean; ticks: individual seeds. Primary: Bonferroni family 4, paired t df=4. Supporting: unadjusted 95% t intervals.\nSame previously examined CMRC public dev (3,219 questions); FP32 eager inference. Higher probability/AVG and lower NLL are better.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.92));out=HERE/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'new_seed_confirmation.png',dpi=190);fig.savefig(out/'new_seed_confirmation.pdf');plt.close(fig)

if __name__=='__main__':main()
