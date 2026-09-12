"""Audit frozen provenance, every stored score, geometry and reproducibility."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv,datetime,hashlib,json
from pathlib import Path
import numpy as np
from predict import predict
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    checked=0
    for file in ['freeze_manifest.json','execution_seal.json']:
        doc=json.loads((ROOT/'frozen'/file).read_text())
        for name,digest in doc['files'].items():assert sha(ROOT/name)==digest,name;checked+=1
    seal=json.loads((ROOT/'frozen/execution_seal.json').read_text());freeze_epoch=datetime.datetime.fromisoformat(seal['timestamp_utc']).timestamp()
    records=0;runs=0;max_score_error=0.;max_initial_norm_error=0.;min_relative_eig=0.;allsteps=[]
    for phase,plan in [('discovery','discovery_plan.json'),('probe','probe_plan.json'),('operator','operator_plan.json'),('confirmation','frozen/confirmation_plan.json'),('intervention','frozen/intervention_plan.json'),('boundary','frozen/boundary_plan.json')]:
        jobs=json.loads((ROOT/plan).read_text());paths=list((ROOT/'results'/phase).iterdir());assert len(paths)==len(jobs)
        for job in jobs:
            path=ROOT/'results'/phase/job['name'];summary=json.loads((path/'summary.json').read_text());c=summary['config'];assert summary['status']=='completed' and summary['error'] is None
            for k,v in job.items():
                if k!='name':assert c[k]==v,(path,k)
            env=json.loads((path/'environment.json').read_text())
            for name,digest in env['sources'].items():assert sha(ROOT/name)==digest,(path,name)
            if phase in ['confirmation','intervention','boundary']:assert (path/'config.json').stat().st_mtime>=freeze_epoch
            arr=np.load(path/'final.npz');h=arr['history'];p=c['p'];y=arr['y'].argmax(1);rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
            assert len(rows)==c['iterations']==len(h)==summary['n_rows'];assert np.isfinite(h).all();assert np.array_equal(y,(2*np.arange(p))%p)
            assert np.array_equal(h[-1],arr['pred']);assert np.isfinite(arr['m']).all()
            targetnorm=c['epsilon']*np.sqrt(2*p) if c['kind']!='none' else 0.
            normerror=abs(np.linalg.norm(arr['m0']-np.eye(2*p))-targetnorm);max_initial_norm_error=max(max_initial_norm_error,normerror);assert normerror<1e-12
            assert np.linalg.eigvalsh(arr['m0']).min()>0
            relmin=float(np.linalg.eigvalsh((arr['m']+arr['m'].T)/2).min()/max(1,np.linalg.norm(arr['m'])));min_relative_eig=min(min_relative_eig,relmin);assert relmin>=-1e-12
            state=np.load(path/'states.npz');assert np.array_equal(state['matrices'][-1],arr['m'])
            data=np.load(path/'data.npz');pairs=data['pairs'];train=data['train_mask'];assert np.array_equal(train,pairs[:,0]!=pairs[:,1]);assert len(pairs)==p*p
            assert str(data['train_hash'])==hashlib.sha256(pairs[train].tobytes()).hexdigest()
            if c['kind']=='custom':
                ip=ROOT/'initializations'/c['amount'];assert env['initializer_sha256']==sha(ip)
                e=np.load(ip)['direction'];assert np.max(abs(arr['m0']-(np.eye(2*p)+targetnorm*e)))<1e-14
            for t,(pred,row) in enumerate(zip(h,rows)):
                assert row['iteration']==t;truth=pred[np.arange(p),y];other=pred.copy();other[np.arange(p),y]=-np.inf;mar=truth-other.max(1)
                values=dict(acc=float(np.mean(pred.argmax(1)==y)),mse=float(np.mean((pred-arr['y'])**2)),mean_margin=float(mar.mean()),min_margin=float(mar.min()))
                for key,value in values.items():
                    error=abs(value-row[key]);max_score_error=max(max_score_error,error);assert error<1e-12,(path,t,key,error)
                allsteps.append(dict(phase=phase,run=path.name,p=p,seed=c['seed'],kind=c['kind'],**row))
            records+=len(rows);runs+=1
    with (ROOT/'results/all_steps.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allsteps[0]));w.writeheader();w.writerows(allsteps)
    forecasts=json.loads((ROOT/'frozen/forecasts.json').read_text());max_pred_diff=0.
    for r in forecasts:
        out=predict(r['p'],r['seed']);diff=abs(out['predicted_acc']-r['predicted_acc']);max_pred_diff=max(max_pred_diff,diff);assert diff<1e-12
    repeat=0
    for p in [17,29]:
        for seed in range(20,25):
            for kind in ['cross','diag']:
                name=f'p{p}_s{seed}_{kind}';a=np.load(ROOT/f'results/discovery/{name}/final.npz');b=np.load(ROOT/f'reference_v2/{name}/final.npz')
                for key in ['m0','m','pred','history']:assert np.array_equal(a[key],b[key]),(name,key)
                repeat+=1
    num=json.loads((ROOT/'results/numerical_check.json').read_text());assert num['passed']
    assert all(r['stored']['correct']==r['refined']['correct'] and ('mp_readout' not in r or r['mp_readout']['correct']==r['stored']['correct']) for r in num['results'])
    assert runs==844 and records==47940
    out=dict(passed=True,runs=runs,records=records,freeze_entries_checked=checked,max_metric_abs_difference=max_score_error,
             max_initial_norm_error=max_initial_norm_error,min_final_relative_eigenvalue=min_relative_eig,
             forecasts_recomputed=len(forecasts),max_forecast_difference=max_pred_diff,old_trajectories_bitwise=repeat,
             numerical_cases=len(num['results']),mp50_cases=sum('mp_readout' in r for r in num['results']),
             scope='all stored scores/configs/data and provenance; higher-precision final readout on listed cases; no arbitrary-precision training',
             statistical_note='same integer seeds across moduli may share RNG draws; within-modulus inference is primary; pooled seed-cluster sensitivity is supplied')
    (ROOT/'results/audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
