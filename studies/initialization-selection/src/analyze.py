"""Predeclared paired statistics and scientific figures; no selection changes."""
from core import *
from evaluate import write_csv
from benchmark import now
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['unmodified','bank','random_probe','greedy_probe','random_full']
COLORS=['#a5a9ae','#254c78','#299b8f','#d08d32','#a24d70']

def run():
    out=ROOT/'analysis';out.mkdir(exist_ok=False)
    rows=list(csv.DictReader((ROOT/'evaluation/directions.csv').open()))
    rng=np.random.default_rng(20260909)
    summaries=[];contrasts=[];costs=[]
    lookup={(int(r['p']),int(r['seed']),r['method']):r for r in rows}
    for p in [17,23]:
        bankcost=json.loads((ROOT/'banks'/f'p{p}'/'result.json').read_text())
        seeds=sorted(set(int(r['seed']) for r in rows if int(r['p'])==p))
        for m in METHODS:
            rr=[lookup[p,s,m] for s in seeds]
            a=np.array([float(r['test_acc']) for r in rr])
            dist=a[rng.integers(0,20,size=(10000,20))].mean(1)
            summaries.append(dict(p=p,method=m,directions=len(a),mean_test_acc=float(a.mean()),
                accuracy_ci_low=float(np.quantile(dist,.025)),accuracy_ci_high=float(np.quantile(dist,.975)),
                mean_test_normalized_margin=float(np.mean([float(r['test_normalized_margin']) for r in rr])),
                mean_validation_acc=float(np.mean([float(r['validation_acc']) for r in rr])),
                mean_validation_test_gap=float(np.mean([float(r['validation_test_gap']) for r in rr])),
                perfect_directions=int((a==1).sum()),zero_directions=int((a==0).sum()),
                mean_charged_fits=float(np.mean([float(r['charged_fits']) for r in rr])),
                total_charged_fits=int(sum(float(r['charged_fits']) for r in rr)),
                mean_online_seconds='' if m=='unmodified' else float(np.mean([float(r['online_seconds']) for r in rr])),
                mean_charged_seconds='' if m=='unmodified' else float(np.mean([float(r['charged_seconds']) for r in rr])),
                max_kernel_condition=max(float(r['final_kernel_condition']) for r in rr),
                max_relative_residual=max(float(r['relative_solve_residual']) for r in rr),
                near_top_ties=sum(int(r['near_top_ties_relative_1e8']) for r in rr)))
        b=np.array([float(lookup[p,s,'bank']['test_acc']) for s in seeds])
        mean_online={m:np.mean([float(lookup[p,s,m]['online_seconds']) for s in seeds]) for m in METHODS[1:]}
        for m in METHODS[2:]:
            a=np.array([float(lookup[p,s,m]['test_acc']) for s in seeds]);delta=b-a
            boot=delta[rng.integers(0,20,size=(10000,20))].mean(1)
            flips=rng.integers(0,2,size=(100000,20))*2-1
            null=(flips*delta).mean(1)
            pv=float((np.sum(np.abs(null)>=abs(delta.mean())-1e-15)+1)/100001)
            contrasts.append(dict(p=p,contrast=f'bank - {m}',mean_difference_pp=float(delta.mean()*100),
                ci_low_pp=float(np.quantile(boot,.025)*100),ci_high_pp=float(np.quantile(boot,.975)*100),
                p_sign_flip=pv,wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum())))
            gap=mean_online[m]-mean_online['bank']
            work=bankcost['setup_seconds']/gap if gap>0 else None
            per_fits=float(lookup[p,seeds[0],m]['online_fits'])
            costs.append(dict(p=p,competitor=m,bank_setup_seconds=bankcost['setup_seconds'],
                bank_mean_online_seconds=float(mean_online['bank']),competitor_mean_online_seconds=float(mean_online[m]),
                single_query_bank_seconds=float(bankcost['setup_seconds']+mean_online['bank']),
                bank_amortized_seconds_at_20=float(bankcost['setup_seconds']/20+mean_online['bank']),
                time_break_even_queries=work,
                fits_break_even_queries=bankcost['fit_count']/(per_fits-60),
                assumption='Fixed observed per-query competitor effort and latency; workload extrapolation, not a new accuracy experiment'))
    order=np.argsort([r['p_sign_flip'] for r in contrasts]);previous=0
    for rank,i in enumerate(order):
        adjusted=min(1.,max(previous,(6-rank)*contrasts[i]['p_sign_flip']))
        contrasts[i]['p_holm']=adjusted;previous=adjusted
    write_csv(out/'summary.csv',summaries);write_csv(out/'paired_comparisons.csv',contrasts);write_csv(out/'costs.csv',costs)
    # Aggregate per-candidate validation logs, including the bank's ranking record.
    candidates=[]
    for path in sorted((ROOT/'runs').glob('*/choice.json')):
        choice=json.loads(path.read_text())
        for i,r in enumerate(choice['selection']['records']):
            vm=r.get('validation',{})
            candidates.append(dict(p=choice['p'],seed=choice['seed'],method=choice['method'],evaluation=i,code=r['code'],
                selected=r['code']==choice['selection']['code'],fits=r['fits'],
                validation_acc=vm.get('acc',''),validation_normalized_margin=vm.get('normalized_margin',''),
                validation_mean_margin=vm.get('mean_margin',''),relative_residual=vm.get('relative_residual',''),
                bank_predicted_z=r.get('predicted_normalized_profile_margin',''),bank_candidates_scored=r.get('candidates_scored','')))
    write_csv(out/'candidates.csv',candidates)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for ax,p in zip(axes,[17,23]):
        ss=[s for s in summaries if s['p']==p];x=np.arange(5)
        a=np.array([s['mean_test_acc'] for s in ss])*100
        low=np.array([s['accuracy_ci_low'] for s in ss])*100;high=np.array([s['accuracy_ci_high'] for s in ss])*100
        ax.bar(x,a,color=COLORS,width=.65)
        ax.errorbar(x,a,yerr=[a-low,high-a],fmt='none',ecolor='#222',capsize=3)
        for xx,aa in zip(x,a):ax.text(xx,max(aa+3,high[xx]+2),f'{aa:.1f}',ha='center',fontsize=9)
        ax.set_xticks(x,['Original','Bank','Random\nprobe','Greedy\nprobe','Random\nfull'])
        ax.set_ylim(0,116);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Held-out accuracy (%)')
        ax.set_title(f'p={p}; 20 new directions')
    fig.suptitle('Fixed N=20 budget ceilings; 95% direction-bootstrap intervals')
    for ext in ['png','svg','pdf']:fig.savefig(out/f'accuracy.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained')
    for ax,p in zip(axes,[17,23]):
        seeds=sorted(set(int(r['seed']) for r in rows if int(r['p'])==p))
        mat=np.array([[float(lookup[p,s,m]['test_acc'])*100 for m in METHODS] for s in seeds])
        im=ax.imshow(mat,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        ax.set_xticks(range(5),['Original','Bank','Rand.\nprobe','Greedy\nprobe','Rand.\nfull'])
        ax.set_yticks(range(20),seeds,fontsize=8);ax.set_ylabel('Direction seed');ax.set_title(f'p={p}')
        for i in range(20):
            for j in range(5):ax.text(j,i,f'{mat[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if mat[i,j]>55 else '#222')
    fig.colorbar(im,ax=axes,label='Held-out accuracy (%)',shrink=.6)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'all-directions.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for ax,p in zip(axes,[17,23]):
        for m,c in zip(METHODS[1:],COLORS[1:]):
            s=next(s for s in summaries if s['p']==p and s['method']==m)
            ax.scatter(s['mean_charged_seconds'],100*s['mean_test_acc'],s=75,c=c,label=m)
            ax.annotate(m,(s['mean_charged_seconds'],100*s['mean_test_acc']),xytext=(5,6),textcoords='offset points',fontsize=8)
        ax.margins(x=.3,y=.3);ax.set_title(f'p={p}');ax.set_xlabel('Mean charged wall time / query (s)');ax.set_ylabel('Held-out accuracy (%)')
    fig.suptitle('Bank setup fully amortized over the measured 20 queries')
    for ext in ['png','svg','pdf']:fig.savefig(out/f'cost-accuracy.{ext}',dpi=180)
    plt.close(fig)
    fits=sum(s['total_charged_fits'] for s in summaries if s['method']!='unmodified')
    facts=dict(utc=now(),status='local completed pilot; not submitted or externally reviewed',independent_directions=40,
        selection_methods=4,selection_runs=160,baseline_references=40,bank_trajectories=104,
        candidate_records=len(candidates),training_trajectories=len(candidates)+104,kernel_fits=fits,
        summaries=summaries,paired_comparisons=contrasts,costs=costs,
        source_files={'directions':'evaluation/directions.csv','candidates':'analysis/candidates.csv','audit':'verification/audit.json'},
        analysis_code_sha256=sha(Path(__file__)))
    write_json(ROOT/'facts.json',facts)
    print(json.dumps(dict(fits=fits,summaries=summaries,contrasts=contrasts,costs=costs)),flush=True)

if __name__=='__main__':run()
