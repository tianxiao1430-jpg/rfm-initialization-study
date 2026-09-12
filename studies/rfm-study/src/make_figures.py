import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from components import circulant_projection
from analyze_confirmation import profile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.16,'savefig.dpi':190,'svg.fonttype':'none'})
COL={17:'#137E80',23:'#D97435',29:'#6976B5'}
def save(fig,name):
    for suffix in ['png','svg']:fig.savefig(OUT/f'{name}.{suffix}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
linear=json.loads((ROOT/'results/linear_checks.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
for p in [17,23,29]:
    xx=np.arange(3)+([17,23,29].index(p)-1)*.18
    vals=[np.mean([r['derivative_norm'] for r in linear if r['p']==p and r['kind']==k]) for k in ['circ','residual','diag']]
    ax[0].plot(xx,vals,'o-',color=COL[p],label=f'p={p}')
ax[0].set(xticks=range(3),xticklabels=['Circulant cross','Residual cross','Diagonal odd'],ylabel='Norm of first-update derivative',title='A  Local response at identity')
ax[0].legend(frameon=False)
for kind,label,color in [('cross','Original cross: circulant part',COL[17]),('residual','Removed circulant part','#B75755'),('diag','Diagonal odd: circulant part','#858585')]:
    a=np.load(ROOT/f'results/discovery/p17_s20_{kind}/states.npz');ys=[]
    for m in a['matrices']:
        b=(m[:17,17:]-m[:17,17:].T)/2;ys.append(max(1e-12,np.sqrt(2)*np.linalg.norm(circulant_projection(b))))
    ax[1].plot(a['steps'],ys,'o-',color=color,label=label)
ax[1].set_yscale('log');ax[1].set_xscale('symlog',linthresh=2)
ax[1].set(xticks=[0,1,2,5,10,59],xticklabels=['0','1','2','5','10','59'],xlabel='RFM evaluation',ylabel='Circulant cross norm',title='B  Example trajectory: p=17, seed=20')
ax[1].legend(frameon=False,fontsize=8,loc='lower right');ax[1].text(.02,.02,'Plot floor: 1e-12',transform=ax[1].transAxes,fontsize=8)
save(fig,'01_channel_selection')

c=json.loads((ROOT/'results/confirmation_analysis.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
for a,p in zip(ax,[17,23]):
    rr=[r for r in c['rows'] if r['p']==p];a.plot([0,1],[0,1],'--',color='#929292',lw=1)
    a.scatter([r['predicted_acc'] for r in rr],[r['actual_acc'] for r in rr],color=COL[p],s=38,alpha=.65,edgecolor='white',linewidth=.5)
    m=c['metrics'][str(p)];lo,hi=np.array(m['mae']['ci95'])*100
    a.set(xlim=(-.04,1.04),ylim=(-.04,1.04),xlabel='Frozen predicted accuracy',ylabel='Measured final accuracy',title=f'p={p}: {len(rr)} untouched directions')
    a.text(.05,.90,f"MAE {100*m['mae']['mean']:.1f} pp\n95% CI [{lo:.1f}, {hi:.1f}]",transform=a.transAxes,fontsize=10)
save(fig,'02_frozen_prediction')

i=json.loads((ROOT/'results/intervention_analysis.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
for p in [17,23]:
    rr=[r for r in i['rows'] if r['p']==p]
    ax[0].scatter([r['phase_worst'] for r in rr],[r['phase_best'] for r in rr],s=44,color=COL[p],alpha=.6,label=f'p={p}, n={len(rr)}')
ax[0].plot([0,1],[0,1],'--',color='#888',lw=1);ax[0].set(xlim=(-.04,1.04),ylim=(-.04,1.04),xlabel='Adverse signs: final accuracy',ylabel='Favorable signs: final accuracy',title='A  Same Fourier amplitudes, different signs');ax[0].legend(frameon=False,loc='lower right')
for x,key in enumerate(['phase_best_effect','phase_worst_effect','remove_effect']):
    for r in i['rows']:
        jitter=(r['seed']-110)*.01
        ax[1].scatter(x+jitter,r[key]*100,s=15,color=COL[r['p']],alpha=.45)
    m=i['metrics']['all'][key];lo,hi=m['seed_cluster_ci95'];mean=m['mean']
    ax[1].errorbar(x,mean*100,yerr=np.array([[mean-lo],[hi-mean]])*100,fmt='o',color='#202C3A',capsize=5,markersize=6)
ax[1].axhline(0,color='#777',lw=1);ax[1].set(xticks=range(3),xticklabels=['Favorable signs','Adverse signs','Remove circulant'],ylabel='Accuracy minus matched controls (pp)',title='B  Direction effects and seed-cluster 95% CI')
save(fig,'03_sign_interventions')

b=json.loads((ROOT/'results/boundary_analysis.json').read_text());settings=list(dict.fromkeys(r['setting'] for r in b['metrics']))
labels=['Default','Denom\n2.5','No\ncentering','Both','BW\n1.5','BW\n2.0','BW\n3.0','Eps\n.001','Eps\n.03']
fig,ax=plt.subplots(2,1,figsize=(12,4.8),layout='constrained')
for axis,metric,title,cmap in zip(ax,['mean_acc','mae'],['A  Mean final accuracy','B  MAE of unchanged default predictor'],['YlGnBu','YlOrRd']):
    mat=np.array([[next(r[metric] for r in b['metrics'] if r['p']==p and r['setting']==s) for s in settings] for p in [17,23]])
    axis.imshow(mat,vmin=0,vmax=1,cmap=cmap,aspect='auto');axis.grid(False)
    axis.set(xticks=range(len(settings)),xticklabels=labels,yticks=[0,1],yticklabels=['p=17','p=23'],title=title+' (10 fixed directions per cell)')
    for (r,col),v in np.ndenumerate(mat):axis.text(col,r,f'{v*100:.1f}'+('%' if metric=='mean_acc' else ' pp'),ha='center',va='center',color='white' if v>.65 else '#18212D',fontsize=10)
save(fig,'04_parameter_boundaries')

p=17;seed=100;model=np.load(ROOT/'frozen/quadratic_p17.npz');basis=model['basis'];q=model['q']
fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');aa=[]
for kind,color in [('phase_best',COL[17]),('phase_worst','#B75755')]:
    e=np.load(ROOT/f'initializations/p{p}_s{seed}_{kind}.npz')['direction'];a=np.einsum('kij,ij->k',basis,e);aa.append(a)
    ax[0].plot(np.arange(1,len(a)+1),a*a,'o-' if kind=='phase_best' else 'x--',color=color,label=kind.replace('phase_',''))
    obs=profile(np.load(ROOT/f'results/intervention/p{p}_s{seed}_{kind}/final.npz')['pred']);pr=.01**2*np.einsum('i,cij,j->c',a,q,a)
    ax[2].plot(range(p),obs,color=color,label=kind.replace('phase_','')+' observed');ax[2].plot(range(p),pr,':',color=color,label=kind.replace('phase_','')+' forecast')
ax[0].set(xlabel='Fourier mode k',ylabel='Squared basis coefficient',title='A  Fourier energies coincide');ax[0].legend(frameon=False)
ax[1].imshow(np.sign(aa),cmap='RdBu',vmin=-1,vmax=1,aspect='auto');ax[1].grid(False)
ax[1].set(xticks=range(len(aa[0])),xticklabels=range(1,len(aa[0])+1),yticks=[0,1],yticklabels=['Favorable','Adverse'],xlabel='Fourier mode k',title='B  Only the signs change')
for (r,col),v in np.ndenumerate(np.sign(aa)):ax[1].text(col,r,'+' if v>0 else '−',ha='center',va='center',color='white',fontsize=15)
ax[2].axvline(0,color='#777',lw=1,alpha=.6);ax[2].set(xlabel='Class offset c − 2a (mod p)',ylabel='Centered mean output score',title='C  Predicted versus observed profile');ax[2].legend(frameon=False,fontsize=8)
fig.suptitle('Preselected example: p=17, seed=100',fontsize=12)
save(fig,'05_fixed_energy_example')
print(json.dumps(dict(figures=5,formats=['png','svg'])))
