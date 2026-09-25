"""Descriptive plots from audited, pre-selected complete seed sets only."""
from reporting_common import *
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'base':'#777777','hidden':'#277DA1','hidden_budget':'#8C6BB1',
        'hidden_both':'#D95319','input':'#21918C','output':'#D29C18','both':'#AA3377'}
NAMES={'base':'Base','hidden':'H','hidden_budget':'Budget H','hidden_both':'H+E+U',
       'input':'E only','output':'U only','both':'E+U (no H)'}
FAMILIES=[('Qwen2.5',MODELS[:4],['0.5B','1.5B','3B','7B']),
          ('Qwen3',MODELS[4:],['0.6B','1.7B','4B','8B'])]
METRICS={'cluener':'CLUENER span micro-F1','wikisql':'WikiSQL execution accuracy (%)',
         'trec50':'TREC50 accuracy (percentage points)', 'squad2':'SQuAD2 F1 (points)'}
ROWS=[]

def save(fig,out,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def record(kind,task,model,arm,n,values):
    ROWS.append([kind,task,model,arm,n,float(np.mean(values)),
                 float(np.std(values,ddof=1)) if len(values)>1 else '',json.dumps(list(map(float,values)))])

def model_series(data,out,standalone=False):
    arms=['base','input','output','both'] if standalone else ['hidden','hidden_budget','hidden_both']
    fig,axs=plt.subplots(2,2,figsize=(12,7.2),sharex='col',sharey='row')
    for ti,task in enumerate(TASKS['core']):
        for fi,(family,models,ticks) in enumerate(FAMILIES):
            ax=axs[ti,fi]
            for ai,arm in enumerate(arms):
                means=[];sds=[]
                for m in models:
                    v=series(data,task,m,arm,3)
                    means.append(np.mean(v) if v is not None else np.nan)
                    sds.append(np.std(v,ddof=1) if v is not None and len(v)>1 else 0)
                    if v is not None:record('standalone' if standalone else 'core',task,m,arm,1 if arm=='base' else 3,v)
                # NaN gaps preserve missing model conditions rather than joining across them.
                ax.errorbar(np.arange(4)+(ai-(len(arms)-1)/2)*.055,means,yerr=sds,
                            color=COLORS[arm],label=NAMES[arm],marker='o',markersize=4.2,
                            linewidth=1.25,linestyle='--',capsize=3,alpha=.95)
            absent=[ticks[i] for i,m in enumerate(models) if any(series(data,task,m,a,3) is None for a in arms)]
            if absent:ax.text(.02,.97,'Incomplete: '+', '.join(absent),transform=ax.transAxes,va='top',fontsize=8,color='#555555')
            ax.set_xticks(range(4),ticks);ax.set_xlim(-.35,3.35);ax.grid(axis='y',alpha=.22)
            if fi==0:ax.set_ylabel(METRICS[task])
            ax.set_title(family if ti==0 else '')
            if ti==1:ax.set_xlabel('Model size (architecture also changes)')
    handles,labels=axs[0,0].get_legend_handles_labels()
    title='Boundary-only adaptation, frozen base weights' if standalone else 'Internal LoRA and boundary additions across model series'
    fig.suptitle(title,y=1.02,fontsize=15)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.985),ncol=len(arms),frameon=False)
    fig.text(.5,-.018,'Same pre-selected three seeds per fitted arm; bars = seed SD, not confidence intervals. Base evaluated once. Missing cells are omitted.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.01,1,.94))
    save(fig,out,'standalone_model_series' if standalone else 'core_model_series')

def untied(data,out):
    comparisons=[('hidden_input','hidden','H+E − H'),('hidden_output','hidden','H+U − H'),
                 ('hidden_both','hidden','H+E+U − H'),('hidden_both','hidden_input','H+E+U − (H+E)'),
                 ('hidden_both','hidden_output','H+E+U − (H+U)')]
    colors=['#21918C','#D29C18','#D95319','#718355','#AA3377']
    fig,axs=plt.subplots(2,2,figsize=(14,7.8))
    for ti,task in enumerate(TASKS['core']):
        for mi,m in enumerate(['qwen25_7b_base','qwen3_8b_base']):
            ax=axs[ti,mi]
            n=next((n for n in [5,3] if all(series(data,task,m,a,n) is not None for a in ARMS)),None)
            ax.axhline(0,color='#555555',lw=.9);ax.grid(axis='y',alpha=.2)
            ax.set_title(f'{LABELS[m]} / {task}'+(f' / n={n}' if n else ' / incomplete'))
            ax.set_xticks(range(5),[c[2] for c in comparisons],rotation=20,ha='right',fontsize=9)
            ax.set_ylabel('Paired score difference (points)')
            if n is None:
                ax.text(.5,.5,'No complete pre-selected seed block',ha='center',transform=ax.transAxes,fontsize=10,color='#777777')
                continue
            for i,(treatment,control,label) in enumerate(comparisons):
                v=np.array(series(data,task,m,treatment,n))-np.array(series(data,task,m,control,n))
                ax.bar(i,np.mean(v),width=.64,color=colors[i],alpha=.35)
                ax.scatter(i+np.linspace(-.18,.18,n),v,color=colors[i],s=29,edgecolor='white',linewidth=.45,zorder=3)
                ax.plot([i-.26,i+.26],[np.mean(v)]*2,color=colors[i],lw=2)
                record('untied_delta',task,m,treatment+'-'+control,n,v)
    fig.suptitle('Untied models: paired effects of E and U additions',y=1.015,fontsize=15)
    fig.text(.5,-.015,'Dots = individual paired seeds; bars = mean. Five seeds when the full block is complete; otherwise the complete three-seed block. See tables for corrected intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.015,1,.98));save(fig,out,'untied_side_effects')

def new_tasks(data,out):
    fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for ti,task in enumerate(TASKS['new_tasks']):
        ax=axs[ti];ax.axhline(0,color='#555555',lw=.9);ax.grid(axis='y',alpha=.2)
        missing=[]
        for i,m in enumerate(MODELS):
            if not all(series(data,task,m,a,3) is not None for a in ['base','hidden','hidden_budget','hidden_both']):
                missing.append(LABELS[m]);continue
            for j,control in enumerate(['hidden','hidden_budget']):
                v=np.array(series(data,task,m,'hidden_both',3))-np.array(series(data,task,m,control,3))
                x=i+[-.16,.16][j];color=COLORS[control]
                ax.scatter(x+np.linspace(-.06,.06,3),v,s=24,alpha=.6,color=color,zorder=3)
                ax.plot([x-.12,x+.12],[np.mean(v)]*2,lw=2.5,color=color,zorder=4)
                record('new_task_delta',task,m,'hidden_both-'+control,3,v)
        ax.set_ylabel('H+E+U − control\n'+METRICS[task]);ax.set_title(task)
        if missing:ax.text(.99,.98,'Incomplete: '+', '.join(missing),transform=ax.transAxes,ha='right',va='top',fontsize=8,color='#666666',wrap=True)
    axs[1].set_xticks(range(8),[LABELS[m].replace('-','\n') for m in MODELS],fontsize=9)
    axs[1].set_xlim(-.5,7.5)
    fig.legend([Line2D([0],[0],color=COLORS[a],lw=2) for a in ['hidden','hidden_budget']],
               ['H+E+U − H','H+E+U − Budget H'],loc='upper center',bbox_to_anchor=(.5,.985),ncol=2,frameon=False)
    fig.suptitle('New tasks: paired gain over two internal-LoRA controls',y=1.015,fontsize=15)
    fig.text(.5,-.012,'Dots = three pre-selected paired seeds; horizontal marks = means. Missing blocks are omitted. SQuAD2 uses the fixed balanced 1,024-example subset.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.015,1,.94));save(fig,out,'new_task_effects')

def main():
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
    if not args.preview:assert read(ROOT/'FINAL_AUDIT.json')['status']=='passed'
    out=ROOT/('figure_previews' if args.preview else 'figures');out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    core=collect('core');new=collect('new_tasks')
    model_series(core,out);model_series(core,out,True);untied(core,out);new_tasks(new,out)
    with (out/'plotted_values.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['kind','task','model','arm_or_contrast','n','mean','seed_sd','raw_values']);w.writerows(ROWS)
    write(out/'PLOT_DATA_STATUS.json',{'at':now(),'preview':args.preview,'plotted_rows':len(ROWS),'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.png','.pdf','.svg','.csv']}})
    print(f'Wrote four descriptive figures to {out}')

if __name__=='__main__':main()
