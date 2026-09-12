"""Exact Taylor coefficients of the first kernel readout at identity.

Only training labels are used in coefficient construction. Held-out labels are
used later to evaluate forecasts, never in the coefficient function.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
import rfm
import torch
import numpy as np
from components import split_odd
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def geometry(x,z,e):
    return ((x@e)*x).sum(1)[:,None]+((z@e)*z).sum(1)[None,:]-2*x@e@z.T

def coefficients(x,y,xt,e,bw=2.5):
    identity=torch.eye(x.shape[1]);k0=rfm.kernel(x,x,identity,bw);b0=rfm.kernel(x,xt,identity,bw)
    h=-geometry(x,x,e)/(2*bw*bw);ht=-geometry(x,xt,e)/(2*bw*bw)
    k1=k0*h;k2=.5*k0*h*h;b1=b0*ht;b2=.5*b0*ht*ht
    lu,piv=torch.linalg.lu_factor(k0)
    a0=torch.linalg.lu_solve(lu,piv,y)
    a1=torch.linalg.lu_solve(lu,piv,-k1@a0)
    a2=torch.linalg.lu_solve(lu,piv,-k1@a1-k2@a0)
    f0=b0.T@a0;f1=b1.T@a0+b0.T@a1;f2=b2.T@a0+b1.T@a1+b0.T@a2
    return f0,f1,f2

if __name__=='__main__':
    old=ROOT.parent/'rfm-numerics/v2/results/followup';rows=[];checks=[]
    for p in [17,29]:
        x,y,xt,yt,*_=rfm.dataset(p)
        for seed in range(20,25):
            for kind in ['break','cross','diag']:
                arr=np.load(old/f'p{p}_s{seed}_{kind}'/'final.npz')
                e=torch.from_numpy((arr['m0']-np.eye(2*p))/.01)
                f0,f1,f2=coefficients(x,y,xt,e)
                raw=rfm.score(f2,yt)['acc'];centered=rfm.score(f2-f2.mean(0),yt)['acc']
                target=arr['pred']-arr['pred'].mean(0)
                model=f2.numpy()-f2.numpy().mean(0)
                cos=float(np.sum(target*model)/(np.linalg.norm(target)*np.linalg.norm(model)))
                rows.append(dict(p=p,seed=seed,kind=kind,final_acc=float(np.mean(arr['pred'].argmax(1)==arr['y'].argmax(1))),
                                 taylor_prediction_acc=raw,centered_taylor_prediction_acc=centered,spatial_cosine=cos,
                                 max_abs_first_coefficient=float(f1.abs().max()),f2_spatial_norm=float(torch.linalg.norm(f2-f2.mean(0)))))
                if p==17 and seed==20:
                    errors=[]
                    for eps in [.002,.001,.0005]:
                        m=torch.eye(2*p)+eps*e;k=rfm.kernel(x,x,m)
                        pred=rfm.kernel(x,xt,m).T@torch.linalg.solve(k,y)
                        errors.append(float(torch.linalg.norm(pred-f0-eps*f1-eps*eps*f2)))
                    checks.append(dict(kind=kind,eps=[.002,.001,.0005],remainder_l2=errors))
                np.savez_compressed(ROOT/'results'/f'taylor_p{p}_s{seed}_{kind}.npz',f0=f0.numpy(),f1=f1.numpy(),f2=f2.numpy())
    out=dict(scope='initial_kernel_fit_only_no_AGOP_proof',rows=rows,checks=checks)
    (ROOT/'results/taylor.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
