"""Export scientific figures only after each study's full audit passes."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
PAIRS=[('output','none'),('both','none'),('both','hidden_budget')]
LABELS=['Output vs.\nhidden LoRA','Bilateral vs.\nhidden LoRA','Bilateral vs.\nhidden budget']


def errorbar(ax,values,scale,label,color,offset=0,interval='ci95'):
    means=np.asarray([v['mean']*scale for v in values])
    bounds=np.asarray([sorted([x*scale for x in v[interval]]) for v in values])
    ax.errorbar(np.arange(3)+offset,means,yerr=np.array([means-bounds[:,0],bounds[:,1]-means]),
                fmt='o',capsize=3,markersize=5,color=color,label=label,linewidth=1.5)


def main():
    p=argparse.ArgumentParser();p.add_argument('study',choices=['calibration','chat']);args=p.parse_args()
    root=HERE/args.study
    audit=json.loads((root/'FINAL_AUDIT.json').read_text());assert audit['status']=='passed'
    data_file=root/('summary.json' if args.study=='calibration' else 'FINAL_ANALYSIS.json')
    s=json.loads(data_file.read_text())
    if args.study=='calibration':
        assert s['completed']==24
        models={'qwen3_06b':'Qwen3-0.6B-Base','qwen25_7b':'Qwen2.5-7B-Base'}
    else:
        assert s['status']=='complete'
        models={'qwen3_06b_chat':'Qwen3-0.6B (post-trained)','qwen25_15b_chat':'Qwen2.5-1.5B-Instruct'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.3,7.0),sharex=True,sharey='row')
    for col,(model,title) in enumerate(models.items()):
        c=[s['contrasts'][f'{model}: {a} - {b}'] for a,b in PAIRS]
        axes[0,col].set_title(title)
        if args.study=='calibration':
            errorbar(axes[0,col],[r['raw_ce'] for r in c],-1000,'Before temperature fitting','#777777',-.08)
            errorbar(axes[0,col],[r['calibrated_ce'] for r in c],-1000,'After dev-only temperature fitting','#2166AC',.08)
            errorbar(axes[1,col],[r['content_token_accuracy'] for r in c],100,'Teacher-forced content accuracy','#2166AC')
        else:
            errorbar(axes[0,col],[r['ce'] for r in c],-1000,'Training-seed interval','#2166AC')
            errorbar(axes[1,col],[r['strict'] for r in c],100,'Training-seed interval','#2166AC',-.06)
            # Prompt intervals are conditional on the fitted seed models.
            errorbar(axes[1,col],[r['strict'] for r in c],100,'Conditional paired-prompt interval','#B35806',.06,
                     'prompt_bootstrap_ci95_conditional_on_fitted_models')
        for row in range(2):
            axes[row,col].axhline(0,color='#555555',linewidth=.9,linestyle='--')
            axes[row,col].grid(axis='y',alpha=.18)
            axes[row,col].set_xticks(np.arange(3),LABELS)
            axes[row,col].set_xlim(-.4,2.4)
    axes[0,0].set_ylabel('Test CE reduction (×1000)\nPositive favors the addition')
    if args.study=='calibration':
        axes[1,0].set_ylabel('Content top-1 accuracy gain\n(percentage points)')
        title='Does a scalar-temperature control explain the stacking gain?'
        foot='Hidden rank 8; 3 training seeds. Nominal paired t 95% intervals; no multiplicity correction.\nTemperature selected separately per arm on dev. Teacher-forced accuracy is not free-generation accuracy.'
        handles,labels=axes[0,0].get_legend_handles_labels()
    else:
        axes[1,0].set_ylabel('IFEval strict prompt gain\n(percentage points)')
        title='Additional benefit on verified chat-ready checkpoints'
        foot='Hidden rank 8; 3 training seeds; all 541 IFEval prompts. No multiplicity correction.\nBlue: nominal paired t 95% seed interval. Orange: paired-prompt 95% bootstrap, conditional on fitted models.'
        handles,labels=axes[1,0].get_legend_handles_labels()
    fig.suptitle(title,fontsize=13,y=.995)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.025,foot,ha='center',fontsize=8.7)
    fig.tight_layout(rect=(0,.10,1,.89))
    out=root/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'{args.study}_increments.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    paths=[root/'manifest.json',root/'FINAL_AUDIT.json',data_file,Path(__file__).resolve()]
    (out/'SOURCE_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    (out/'CAPTION.md').write_text(title+'\n\n'+foot+'\n\nPositive favors the added adapter. See the study report for all raw cells, valid-539 sensitivity and material limitations. These data do not establish broad cross-task utility or an optimized hidden-rank allocation frontier.\n')
    print(out/f'{args.study}_increments.png')


if __name__=='__main__':main()
