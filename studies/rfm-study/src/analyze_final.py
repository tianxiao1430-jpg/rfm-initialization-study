import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv,json
from pathlib import Path
import numpy as np
from analyze_confirmation import interval
ROOT=Path(__file__).resolve().parents[1]
def read(phase,name):
    s=json.loads((ROOT/f'results/{phase}/{name}/summary.json').read_text());assert s['status']=='completed'
    return s
def paired_stats(rows,key):
    values=np.array([r[key] for r in rows]);groups=[np.array([i for i,r in enumerate(rows) if r['p']==p]) for p in sorted(set(r['p'] for r in rows))]
    rng=np.random.default_rng(20260906)
    samples=np.concatenate([rng.choice(g,size=(10000,len(g)),replace=True) for g in groups],axis=1)
    bootstrap=values[samples].mean(1)
    signs=rng.choice([-1.,1.],size=(200000,len(values)))
    permutation=(signs@values)/len(values);p=(1+np.sum(abs(permutation)>=abs(values.mean())-1e-15))/200001
    # Same integer seeds occur in both moduli. Cluster them as a robustness check.
    seeds=sorted(set(r['seed'] for r in rows));sums=np.array([sum(r[key] for r in rows if r['seed']==s) for s in seeds]);sizes=np.array([sum(r['seed']==s for r in rows) for s in seeds])
    ix=rng.integers(0,len(seeds),size=(10000,len(seeds)));cb=sums[ix].sum(1)/sizes[ix].sum(1)
    cs=rng.choice([-1.,1.],size=(200000,len(seeds)));cp=(1+np.sum(abs(cs@sums/len(rows))>=abs(values.mean())-1e-15))/200001
    return dict(n_directions=len(values),n_seed_clusters=len(seeds),mean=float(values.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                p_two_sided_mc=float(p),seed_cluster_ci95=np.quantile(cb,[.025,.975]).tolist(),seed_cluster_p_two_sided_mc=float(cp))
def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    forecast=json.loads((ROOT/'frozen/forecasts.json').read_text());fr={(r['p'],r['seed']):r for r in forecast}
    plan=json.loads((ROOT/'frozen/intervention_plan.json').read_text());rows=[]
    for p,s in sorted(set((j['p'],j['seed']) for j in plan)):
        base=read('confirmation',f'p{p}_s{s}_cross');r=dict(p=p,seed=s,base_acc=base['final']['acc'])
        for j in [j for j in plan if j['p']==p and j['seed']==s]:
            variant=j['name'].split(f'p{p}_s{s}_')[1];r[variant]=read('intervention',j['name'])['final']['acc']
        for family in ['phase_best','phase_worst','remove']:
            r[family+'_control_mean']=float(np.mean([r[f'{family}_control{i}'] for i in range(3)]))
            r[family+'_effect']=r[family]-r[family+'_control_mean']
        r['best_minus_worst']=r['phase_best']-r['phase_worst'];rows.append(r)
    metrics={}
    for label,rr in [('all',rows)]+[(str(p),[r for r in rows if r['p']==p]) for p in [17,23]]:
        st={key:paired_stats(rr,key) for key in ['phase_best_effect','phase_worst_effect','remove_effect','best_minus_worst']}
        order=sorted(['phase_best_effect','phase_worst_effect','remove_effect'],key=lambda k:st[k]['p_two_sided_mc']);previous=0
        for i,k in enumerate(order):
            previous=max(previous,min(1,(3-i)*st[k]['p_two_sided_mc']));st[k]['holm_p']=previous
        failed=[r for r in rr if r['base_acc']<=.1];success=[r for r in rr if r['base_acc']>=.9]
        st['rescue']=dict(eligible=len(failed),phase_best_success=sum(r['phase_best']>=.9 for r in failed),
                           control_success=sum(r[f'phase_best_control{i}']>=.9 for r in failed for i in range(3)))
        st['damage']=dict(eligible=len(success),phase_worst_failure=sum(r['phase_worst']<=.1 for r in success),remove_failure=sum(r['remove']<=.1 for r in success))
        st['mean_acc']={key:float(np.mean([r[key] for r in rr])) for key in ['base_acc','phase_best','phase_worst','remove','phase_best_control_mean','phase_worst_control_mean','remove_control_mean']}
        metrics[label]=st
    write_csv(ROOT/'results/intervention.csv',rows)
    (ROOT/'results/intervention_analysis.json').write_text(json.dumps(dict(metrics=metrics,rows=rows),indent=2))
    br=[]
    for p in [17,23]:
        for seed in range(100,110):
            for setting in ['default','paper','uncentered','paper_uncentered','bw15','bw20','bw30','eps001','eps03']:
                s=read('confirmation' if setting=='default' else 'boundary',f'p{p}_s{seed}_'+('cross' if setting=='default' else setting))
                c=s['config'];acc=s['final']['acc'];pred=fr[(p,seed)]['predicted_acc']
                br.append(dict(p=p,seed=seed,setting=setting,acc=acc,predicted_acc=pred,error=abs(acc-pred),
                               denominator=2*c['bandwidth']**2 if c['formula']=='code' else c['bandwidth'],centering=c['centering'],epsilon=c['epsilon']))
    bm=[]
    for p in [17,23]:
        for setting in dict.fromkeys(r['setting'] for r in br):
            rr=[r for r in br if r['p']==p and r['setting']==setting]
            bm.append(dict(p=p,setting=setting,n=len(rr),mean_acc=float(np.mean([r['acc'] for r in rr])),
                           success_count=sum(r['acc']>=.9 for r in rr),mae=float(np.mean([r['error'] for r in rr]))))
    write_csv(ROOT/'results/boundary.csv',br)
    (ROOT/'results/boundary_analysis.json').write_text(json.dumps(dict(metrics=bm,rows=br),indent=2))
    allruns=[];totals={}
    for phase in ['discovery','probe','operator','confirmation','intervention','boundary']:
        phases=[]
        for path in sorted((ROOT/'results'/phase).iterdir()):
            s=json.loads((path/'summary.json').read_text());assert s['status']=='completed';c=s['config'];m=s['final']
            row=dict(phase=phase,name=path.name,**c,**{k:v for k,v in m.items() if k!='elapsed_s'},n_rows=s['n_rows'],elapsed_s=s['elapsed_s'])
            allruns.append(row);phases.append(row)
        totals[phase]=dict(runs=len(phases),records=sum(r['n_rows'] for r in phases),elapsed_s=sum(r['elapsed_s'] for r in phases))
    write_csv(ROOT/'results/all_runs.csv',allruns)
    summary=dict(totals=totals,runs=len(allruns),records=sum(r['n_rows'] for r in allruns),
                 statistical_units='40 new discovery directions; 50 p17 + 30 p23 confirmation directions. Paired interventions, deterministic basis probes and repeated old seeds are not extra independent directions.',
                 scope='local synthetic Gaussian RFM; no external feedback or publication')
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(intervention=metrics,boundary=bm,summary=summary),indent=2))
if __name__=='__main__':main()
