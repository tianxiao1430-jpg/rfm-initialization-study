"""Construct finite-amplitude quadratic response; evaluate discovery only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from directions_v2 import mode
ROOT=Path(__file__).resolve().parents[1]

def profile(pred):
    p=len(pred);out=np.array([np.roll(pred[a],-2*a) for a in range(p)]).mean(0)
    return out-out.mean()

def scalar(a,q):
    v=np.einsum('i,cij,j->c',a,q,a)
    return float((v[0]-v[1:].max())/np.sqrt(np.mean((v-v.mean())**2))),v

def main():
    path=ROOT/'models';path.mkdir(exist_ok=True);checks=[]
    for p in [17,23]:
        n=(p-1)//2;eps=.002;op=ROOT/'results/operator'
        base=profile(np.load(op/f'p{p}_none/final.npz')['pred']);q=np.zeros((p,n,n))
        for k in range(n):
            q[:,k,k]=(profile(np.load(op/f'p{p}_mode{k+1}/final.npz')['pred'])-base)/eps**2
        for k in range(n):
            for j in range(k+1,n):
                pp=(profile(np.load(op/f'p{p}_pair{k+1}_{j+1}/final.npz')['pred'])-base)/eps**2
                q[:,k,j]=q[:,j,k]=pp-.5*(q[:,k,k]+q[:,j,j])
        basis=np.stack([mode(p,k+1) for k in range(n)])
        np.savez_compressed(path/f'quadratic_p{p}.npz',q=q,basis=basis,epsilon=eps,baseline=base)
        for k,j in [(1,2),(1,3),(2,3)]:
            a=np.zeros(n);a[k-1]=1/np.sqrt(2);a[j-1]=-1/np.sqrt(2)
            pred=eps**2*np.einsum('i,cij,j->c',a,q,a)
            obs=profile(np.load(op/f'p{p}_difference{k}_{j}/final.npz')['pred'])-base
            checks.append(dict(p=p,check=f'difference_{k}_{j}',relative_l2=float(np.linalg.norm(pred-obs)/np.linalg.norm(obs)),correct_argmax=bool(pred.argmax()==obs.argmax())))
        obs=profile(np.load(op/f'p{p}_mode1_half/final.npz')['pred'])-base
        pred=.001**2*q[:,0,0]
        checks.append(dict(p=p,check='half_amplitude',relative_l2=float(np.linalg.norm(pred-obs)/np.linalg.norm(obs)),correct_argmax=bool(pred.argmax()==obs.argmax())))
    qdata=np.load(path/'quadratic_p17.npz');q=qdata['q'];basis=qdata['basis'];rows=[]
    for seed in range(30,70):
        arr=np.load(ROOT/f'results/discovery/p17_s{seed}_cross/final.npz')
        e=(arr['m0']-np.eye(34))/(.01*np.sqrt(34));a=np.einsum('kij,ij->k',basis,e)
        x,v=scalar(a,q);y=float((arr['pred'].argmax(1)==arr['y'].argmax(1)).mean())
        observed=profile(arr['pred']);pred=.01**2*v
        rows.append(dict(seed=seed,x=x,acc=y,profile_relative_l2=float(np.linalg.norm(pred-observed)/np.linalg.norm(observed)),profile_cosine=float(np.dot(pred,observed)/(np.linalg.norm(pred)*np.linalg.norm(observed)))))
    x=np.array([r['x'] for r in rows]);y=np.array([r['acc'] for r in rows]);loo=[]
    for i in range(len(x)):
        ix=np.arange(len(x))!=i;loo.append(float(IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(x[ix],y[ix]).predict(x[i:i+1])[0]))
    fitted=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(x,y)
    calibration=dict(x=fitted.X_thresholds_.tolist(),y=fitted.y_thresholds_.tolist(),training_seeds=list(range(30,70)),
                     meaning='predict final fraction correct across p fixed inputs, isotonic fit on 40 discovery direction outcomes',
                     scalar='(q_profile[0]-max(q_profile[1:])) / RMS(center(q_profile)); uses known modular-addition task rule',
                     constant=float(y.mean()))
    (path/'calibration_candidate.json').write_text(json.dumps(calibration,indent=2))
    out=dict(checks=checks,rows=rows,spearman=float(spearmanr(x,y).statistic),loo_mae=float(np.mean(abs(y-loo))),
             loo_mse=float(np.mean((y-loo)**2)),loo_success_agreement=float(np.mean((np.array(loo)>=.9)==(y>=.9))),
             constant_loo_mae=float(np.mean(abs(y-(y.sum()-y)/(len(y)-1)))),
             diagonal_energy_fraction=float(np.sum(np.einsum('cii->ci',q)**2)/np.sum(q*q)))
    (ROOT/'results/quadratic_discovery.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
