"""Post-outcome cosmetic repair only: avoid overlapping labels at equal accuracy.

This reads the unchanged summary. It neither recomputes statistics nor selects
models. Original analysis code remains identical to its pre-evaluation seal.
"""
from core import ROOT,write_json,sha
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

rows=list(csv.DictReader((ROOT/'analysis/summary.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
styles=[('bank','Bank','#254c78','D'),('random_probe','Random probe','#299b8f','o'),
        ('greedy_probe','Greedy probe','#d08d32','x'),('random_full','Random full','#a24d70','s')]
for ax,p in zip(axes,[17,23]):
    for m,label,color,marker in styles:
        r=next(r for r in rows if int(r['p'])==p and r['method']==m)
        kwargs=dict(s=90 if m=='random_probe' else 62,marker=marker,label=label,linewidths=1.8,zorder=3)
        if m=='random_probe':kwargs.update(facecolors='none',edgecolors=color)
        else:kwargs['color']=color
        ax.scatter(float(r['mean_charged_seconds']),100*float(r['mean_test_acc']),**kwargs)
    ax.set_ylim(96.8,100.65);ax.set_yticks([97,98,99,100]);ax.margins(x=.18)
    ax.set_xlabel('Mean charged wall time / query (s)');ax.set_ylabel('Held-out accuracy (%)')
    ax.set_title(f'p={p}; 20 directions');ax.grid(axis='y',alpha=.16,zorder=0)
fig.suptitle('Bank setup fully charged across 20 queries')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
for ext in ['png','svg','pdf']:fig.savefig(ROOT/'analysis'/f'cost-accuracy.{ext}',dpi=180)
write_json(ROOT/'analysis/figure-readability.json',dict(reason='Replace overlapping labels with distinguishable markers and legend; same coordinates and source data',
    source_summary_sha256=sha(ROOT/'analysis/summary.csv'),script_sha256=sha(Path(__file__)),statistics_changed=False))
