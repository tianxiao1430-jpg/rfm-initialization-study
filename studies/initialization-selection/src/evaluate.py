"""Separate held-out evaluator. Requires sealed completion of every selector."""
from core import *
from benchmark import check_freeze,now
import csv

def check_seals():
    check_freeze()
    seal=json.loads((ROOT/'selection-seal.json').read_text())
    assert seal['count']==160 and seal['freeze_sha256']==sha(ROOT/'freeze.json')
    for name,h in seal['files'].items(): assert sha(ROOT/name)==h,name
    bank=json.loads((ROOT/'banks-seal.json').read_text())
    for name,h in bank['files'].items(): assert sha(ROOT/name)==h,name
    return seal

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    seal=check_seals();plan=json.loads((ROOT/'plan.json').read_text())
    out=ROOT/'evaluation';out.mkdir(exist_ok=False)
    write_json(out/'start.json',dict(utc=now(),selection_seal_sha256=sha(ROOT/'selection-seal.json'),selection_sealed_utc=seal['utc']))
    rows=[]
    for job in plan['jobs']:
        p=job['p'];conf=plan['moduli'][str(p)];run=ROOT/'runs'/job['name']
        info=json.loads((run/'result.json').read_text())
        x,y,xv,yv=selection_data(p,conf['validation'])
        xt,yt=inputs(p,[(a,a) for a in conf['test']])
        bank=json.loads((ROOT/'banks'/f'p{p}'/'result.json').read_text())
        for method,filename in [(job['method'],'selected.npz')]+([('unmodified','unmodified.npz')] if job['method']=='random_full' else []):
            state=load_state(run/filename)
            pred=(rfm.kernel(x,xt,state.m,2.5).T@state.alpha).cpu().numpy()
            vp=(rfm.kernel(x,xv,state.m,2.5).T@state.alpha).cpu().numpy()
            saved=np.load(run/filename)
            expected=saved['validation_predictions'] if method=='unmodified' else saved['validation_history'][-1]
            assert np.array_equal(vp,expected)
            tm=metrics(pred,yt.cpu().numpy());vm=metrics(vp,yv.cpu().numpy())
            # Saved final-kernel conditioning is diagnostic; it never excludes a direction.
            cond=float(torch.linalg.cond(state.k))
            residual=float(torch.linalg.norm(state.k@state.alpha-y)/torch.linalg.norm(y))
            ordered=np.sort(pred,axis=1)
            topgap=ordered[:,-1]-ordered[:,-2]
            scale=np.sqrt(((pred-pred.mean(1,keepdims=True))**2).mean(1))
            small=int((topgap <= 1e-8*np.maximum(scale,1e-300)).sum())
            name=f'p{p}-s{job["seed"]}-{method}'
            np.savez_compressed(out/f'{name}.npz',predictions=pred,targets=yt.cpu().numpy(),
                                diagonal_indices=conf['test'],validation_predictions=vp)
            baseline=method=='unmodified'
            row=dict(p=p,seed=job['seed'],method=method,chosen_code=0 if baseline else info['chosen_code'],
                     test_acc=tm['acc'],test_correct=tm['correct'],test_n=tm['n'],
                     test_normalized_margin=tm['normalized_margin'],test_mean_margin=tm['mean_margin'],test_mse=tm['mse'],
                     validation_acc=vm['acc'],validation_normalized_margin=vm['normalized_margin'],
                     validation_test_gap=vm['acc']-tm['acc'],
                     online_fits=60 if baseline else info['fit_count'],
                     charged_fits=60 if baseline else info['fit_count']+(bank['fit_count']/20 if method=='bank' else 0),
                     online_updates=59 if baseline else info['update_count'],
                     unique_candidates=1 if baseline else info['unique_candidates'],
                     online_seconds='' if baseline else info['online_seconds'],
                     charged_seconds='' if baseline else info['online_seconds']+(bank['setup_seconds']/20 if method=='bank' else 0),
                     peak_cuda_allocated_bytes='' if baseline else info['peak_cuda_allocated_bytes'],
                     final_kernel_condition=cond,relative_solve_residual=residual,
                     near_top_ties_relative_1e8=small,source=str((run/filename).relative_to(ROOT)))
            rows.append(row)
        del state,x,y,xv,yv,xt,yt
    write_csv(out/'directions.csv',rows)
    write_json(out/'complete.json',dict(utc=now(),rows=len(rows),directions=40,selectors=160,baseline_references=40))
    print(json.dumps(dict(evaluation='complete',rows=len(rows))),flush=True)

if __name__=='__main__': setup();run()
