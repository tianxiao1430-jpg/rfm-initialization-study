"""Fixed-M readout verification: extended-precision refinement and 50-digit LU.

This does not retrain the RFM in arbitrary precision. The tiny antisymmetric
component of saved M is removed and its norm reported.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]
mp.mp.dps=50

def inputs(p):
    pairs=np.array([(a,b) for a in range(p) for b in range(p)])
    x=np.concatenate([np.eye(p)[pairs[:,0]],np.eye(p)[pairs[:,1]]],axis=1)
    y=np.eye(p)[pairs.sum(1)%p]
    train=pairs[:,0]!=pairs[:,1]
    return pairs,x[train],y[train],x[~train],y[~train],train

def kernel(x,z,m,denom):
    n=((x@m)*x).sum(1)
    nz=((z@m)*z).sum(1)
    d=n[:,None]+nz[None,:]-2*((x@m)@z.T)
    return np.exp(-np.maximum(d,0)/denom)

def score(pr,y):
    labels=y.argmax(1);correct=pr[np.arange(len(pr)),labels]
    other=pr.copy();other[np.arange(len(pr)),labels]=-np.inf
    margin=correct-other.max(1)
    return dict(correct=int(np.sum(pr.argmax(1)==labels)),n_test=len(pr),
                mean_margin=float(margin.mean()),min_margin=float(margin.min()))

def mp_readout(p,m,denom):
    pairs,x,y,xt,yt,train=inputs(p)
    mm=mp.matrix([[mp.mpf(float(v)) for v in row] for row in m])
    mm=(mm+mm.T)/2
    tp=pairs[train];vp=pairs[~train]
    def entry(u,v):
        uu=[int(u[0]),p+int(u[1])];vv=[int(v[0]),p+int(v[1])]
        delta={}
        for i in uu:delta[i]=delta.get(i,0)+1
        for i in vv:delta[i]=delta.get(i,0)-1
        dist=sum(mp.mpf(ci*cj)*mm[i,j] for i,ci in delta.items() for j,cj in delta.items())
        return mp.exp(-dist/mp.mpf(denom))
    K=mp.matrix([[entry(u,v) for v in tp] for u in tp])
    B=mp.matrix([[entry(u,v) for v in vp] for u in tp])
    P,L,U=mp.lu(K)
    # Factor once, solve every label column with triangular substitutions.
    Y=P*mp.matrix(y.tolist());n=len(tp);A=mp.matrix(n,p)
    for c in range(p):
        z=[mp.mpf(0)]*n
        for i in range(n):z[i]=(Y[i,c]-sum(L[i,j]*z[j] for j in range(i)))/L[i,i]
        for i in reversed(range(n)):A[i,c]=(z[i]-sum(U[i,j]*A[j,c] for j in range(i+1,n)))/U[i,i]
    out=B.T*A
    residual=mp.norm(K*A-mp.matrix(y.tolist()))/mp.norm(mp.matrix(y.tolist()))
    return np.array([[str(v) for v in row] for row in out.tolist()],dtype=np.longdouble),str(residual)

def run(relative,mp_check=False):
    path=ROOT/'results'/relative
    cfg=json.loads((path/'config.json').read_text());arr=np.load(path/'final.npz')
    m=arr['m'];stored=arr['pred'];p=cfg['p']
    _,x,y,xt,yt,_=inputs(p)
    ld=np.longdouble
    ml=m.astype(ld);ml=(ml+ml.T)/2
    denom=2*cfg['bandwidth']**2 if cfg['formula']=='code' else cfg['bandwidth']
    K=kernel(x.astype(ld),x.astype(ld),ml,ld(denom));B=kernel(x.astype(ld),xt.astype(ld),ml,ld(denom))
    K+=ld(cfg['ridge'])*np.eye(len(x),dtype=ld)
    factors=lu_factor(K.astype(float));A=lu_solve(factors,y).astype(ld)
    residuals=[]
    for _ in range(5):
        R=y.astype(ld)-K@A
        residuals.append(float(np.linalg.norm(R)/np.linalg.norm(y)))
        A+=lu_solve(factors,R.astype(float)).astype(ld)
    refined=B.T@A
    centered=stored-stored.mean(1,keepdims=True)
    result=dict(run=relative,scope='fixed_symmetrized_M_readout_only',
                longdouble_mantissa_bits=int(np.finfo(ld).nmant+1),
                kernel_condition_2=float(np.linalg.cond(K.astype(float))),
                m_relative_asymmetry=float(np.linalg.norm(m-m.T)/np.linalg.norm(m)),
                refinement_residuals=residuals,
                final_extended_residual=float(np.linalg.norm(K@A-y)/np.linalg.norm(y)),
                max_abs_prediction_difference=float(np.max(np.abs(refined-stored))),
                relative_centered_signal_difference=float(np.linalg.norm(refined-stored)/max(np.linalg.norm(centered),1e-30)),
                stored=score(stored,yt),refined=score(refined,yt))
    if mp_check:
        start=time.monotonic();precise,residual=mp_readout(p,m,denom)
        result.update(mp_decimal_digits=50,mp_readout=score(precise,yt),mp_relative_residual=residual,
                      mp_max_difference_vs_extended=float(np.max(np.abs(precise-refined))),
                      mp_elapsed_s=time.monotonic()-start)
    np.savez_compressed(ROOT/'results'/('readout_'+relative.replace('/','_')+'.npz'),stored=stored,refined=refined,y=yt)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    cases=[('screen/'+name,name.startswith('p7_')) for name in
           ['p7_none','p7_break','p7_preserve','p11_break','s0_none','s0_break','s0_diag','s0_cross','s3_break','s20_cross']]
    cases += [('followup/p17_s21_cross',True),('followup/p17_s20_cross',False)]
    results=[run(path,check) for path,check in cases]
    (ROOT/'results/numerical.json').write_text(json.dumps(dict(passed=all(r['stored']['correct']==r['refined']['correct'] for r in results),results=results),indent=2))
