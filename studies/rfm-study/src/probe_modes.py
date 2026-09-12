"""Discovery-only modal diagnostics and analytic first-AGOP derivative."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
import rfm
from components import circulant_projection,cross_matrix
from taylor import geometry
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)

def first_derivative(p,e):
    x,y,*_=rfm.dataset(p);bw=2.5;m=torch.eye(2*p);k=rfm.kernel(x,x,m,bw)
    alpha=torch.linalg.solve(k,y);dk=k*(-geometry(x,x,e)/(2*bw*bw));da=torch.linalg.solve(k,-dk@alpha)
    n,d=x.shape;c=y.shape[1]
    def grad_terms(aa,xm,kk):
        a=(aa.reshape(n,c,1)@xm.reshape(n,1,d)).reshape(n,c*d)
        return (kk.T@a).reshape(n,c,d)-((aa.T@kk).T.reshape(n,c,1)@xm.reshape(n,1,d))
    g=-grad_terms(alpha,x,k)/(bw*bw);g-=g.mean(0)
    dg=-(grad_terms(da,x,k)+grad_terms(alpha,x@e,k)+grad_terms(alpha,x,dk))/(bw*bw);dg-=dg.mean(0)
    agop=torch.einsum('ncd,nce->de',g,g)/n
    dagop=(torch.einsum('ncd,nce->de',dg,g)+torch.einsum('ncd,nce->de',g,dg))/n
    v,q=torch.linalg.eigh(agop);sq=v.clamp(min=0).sqrt();den=sq[:,None]+sq[None,:]
    projected=q.T@dagop@q;out=torch.where(den>1e-7,projected/den.clamp(min=1e-7),0.)
    dm=q@out@q.T
    return dm,dict(min_eigen=float(v.min()),nonzero_min=float(v[v>1e-12].min()),
                    sylvester_residual=float(torch.linalg.norm((q@torch.diag(sq)@q.T)@dm+dm@(q@torch.diag(sq)@q.T)-dagop)/torch.linalg.norm(dagop)))

def main():
    out={'derivative':[], 'directions':[]}
    for p in [17,29]:
        idx=np.arange(p);basis=[]
        for k in range(1,(p+1)//2):
            b=np.sin(2*np.pi*k*(idx[None,:]-idx[:,None])/p);e=cross_matrix(b);e/=np.linalg.norm(e);basis.append(e)
        modes=[]
        for e in basis:
            dm,meta=first_derivative(p,torch.from_numpy(e));modes.append(dm.numpy())
        mat=np.einsum('aij,bij->ab',np.array(basis),np.array(modes));reconstruction=np.einsum('ab,aij->bij',mat,np.array(basis))
        out['derivative'].append(dict(p=p,matrix=mat.tolist(),outside_relative=float(np.linalg.norm(np.array(modes)-reconstruction)/np.linalg.norm(modes)),meta=meta))
        for s in range(20,25):
            for kind in ['circ','cross']:
                path=ROOT/f'results/discovery/p{p}_s{s}_{kind}';a=np.load(path/'final.npz');b=a['m0'][:p,p:];c=circulant_projection(b);z=circulant_projection(a['m'][:p,p:]);z=(z-z.T)/2
                ratio=float(np.sum(c*z)/np.sum(c*c));offset=(a['pred'].argmax(1)-2*np.arange(p))%p
                out['directions'].append(dict(p=p,seed=s,kind=kind,modal_shape_error=float(np.linalg.norm(z-ratio*c)/np.linalg.norm(z)),gain=ratio,
                                              beta=np.fft.fft(c[0]).imag[1:(p+1)//2].tolist(),final_beta=np.fft.fft(z[0]).imag[1:(p+1)//2].tolist(),offsets=offset.tolist()))
        e=torch.from_numpy(basis[0]);dm,meta=first_derivative(p,e);x,y,*_=rfm.dataset(p)
        k=rfm.kernel(x,x,torch.eye(2*p));alpha=torch.linalg.solve(k,y);base=rfm.update(x,torch.eye(2*p),alpha,k)[0]
        checks=[]
        for h in [.001,.0003,.0001]:
            deltas=[]
            for sign in [1,-1]:
                m=torch.eye(2*p)+sign*h*e;k=rfm.kernel(x,x,m);a=torch.linalg.solve(k,y);deltas.append(rfm.update(x,m,a,k)[0])
            fd=(deltas[0]-deltas[1])/(2*h);checks.append(dict(h=h,relative_error=float(torch.linalg.norm(fd-dm)/torch.linalg.norm(dm))))
        out['derivative'][-1]['finite_difference_checks']=checks
    (ROOT/'results/modes.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
