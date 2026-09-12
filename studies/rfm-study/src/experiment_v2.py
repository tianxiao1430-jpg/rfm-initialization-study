"""RFM mechanism study. Frozen v1 kernel/gradient/AGOP implementation reused."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
import rfm
import torch
import numpy as np
from components import circulant_projection,features,cross_matrix

def unused_original_direction(p,seed,kind,amount=None):
    d=rfm.perturbation(p,seed,'break')
    if kind=='diag':
        d[:p,p:]=0;d[p:,:p]=0
        return d/torch.linalg.norm(d)
    d[:p,:p]=0;d[p:,p:]=0
    d/=torch.linalg.norm(d)
    if kind=='cross':return d
    b=d[:p,p:].numpy();c=circulant_projection(b);r=b-c
    if kind=='circ':b=c
    elif kind=='residual':b=r
    elif kind=='boost':
        q=float(amount)
        b=np.sqrt(q)*c/np.linalg.norm(c)+np.sqrt(1-q)*r/np.linalg.norm(r)
    elif kind=='centered':
        h=np.eye(p)-np.ones((p,p))/p;b=h@b@h
    elif kind=='rotation_control':
        # Match the angle to the amount-energy circulant boost, at fixed norm.
        q=float(amount);unit=b/np.linalg.norm(b)
        target=np.sqrt(q)*c/np.linalg.norm(c)+np.sqrt(1-q)*r/np.linalg.norm(r)
        cosine=float(np.sum(unit*target));cosine=np.clip(cosine,-1,1)
        rng=np.random.default_rng(seed+730001)
        v=rng.normal(size=(p,p));v=(v-v.T)/2
        v-=np.sum(v*unit)*unit;v/=np.linalg.norm(v)
        b=cosine*unit+np.sqrt(max(0,1-cosine*cosine))*v
    else:raise ValueError(kind)
    e=torch.from_numpy(cross_matrix(b));return e/torch.linalg.norm(e)

from directions_v2 import initial_direction

def run(config,path):
    c=dict(p=17,seed=0,kind='cross',epsilon=.01,amount=None,formula='code',centering=True,
           eig_cutoff=0.,ridge=0.,bandwidth=2.5,iterations=60,device='cuda',dtype='float64',timeout=600)
    c.update(config);path=Path(path);path.mkdir(parents=True,exist_ok=False)
    (path/'config.json').write_text(json.dumps(c,indent=2))
    torch.set_default_dtype(getattr(torch,c['dtype']));torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    env=rfm.environment();env['sources']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/components.py',ROOT/'vendor/rfm.py',ROOT/'src/experiment.py',ROOT/'src/directions_v2.py']}
    (path/'environment.json').write_text(json.dumps(env,indent=2))
    start=time.monotonic();rows=[];preds=[];states=[];state_steps=[];status='completed';error=None
    try:
        p=c['p'];dt=getattr(torch,c['dtype']);device=c['device']
        x,y,xt,yt,_,_,data=rfm.dataset(p,dtype=dt,device=device)
        m0=torch.eye(2*p,dtype=torch.float64)
        if c['kind']!='none':m0+=c['epsilon']*(2*p)**.5*initial_direction(p,c['seed'],c['kind'],c['amount'])
        assert torch.linalg.eigvalsh(m0).min()>0
        np.savez_compressed(path/'data.npz',**data)
        if c['kind'] not in ['none','diag']:
            feats=features(m0[:p,p:].numpy())
        else:feats={}
        (path/'features.json').write_text(json.dumps(feats,indent=2))
        m=m0.to(dtype=dt,device=device);a0=None
        bw=c['bandwidth'] if c['formula']=='code' else (c['bandwidth']/2)**.5
        for t in range(c['iterations']):
            if time.monotonic()-start>c['timeout']:raise TimeoutError('per-run budget')
            k=rfm.kernel(x,x,m,bw);kr=k+c['ridge']*torch.eye(len(x),device=device,dtype=dt)
            alpha=torch.linalg.solve(kr,y);pred=rfm.kernel(x,xt,m,bw).T@alpha
            assert bool(torch.isfinite(pred).all())
            scores=rfm.score(pred,yt)
            row=dict(iteration=t,**scores,relative_residual=float(torch.linalg.norm(kr@alpha-y)/torch.linalg.norm(y)),
                     symmetry_defect=float(torch.linalg.norm(m-rfm.swapped(m))/torch.linalg.norm(m)),elapsed_s=time.monotonic()-start)
            rows.append(row);preds.append(pred.cpu().numpy())
            with (path/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if t in [0,1,2,5,10,c['iterations']-1]:states.append(m.cpu().numpy());state_steps.append(t)
            if t!=c['iterations']-1:
                g=rfm.gradients(x,m,alpha,k,bw,centering=c['centering'])
                agop=torch.zeros_like(m)
                for batch in torch.split(g,2):agop+=torch.sum(batch.transpose(1,2)@batch,dim=0)
                agop/=len(g)
                if t==0:a0=agop.cpu().numpy()
                ev,q=torch.linalg.eigh(agop);ev=torch.where(ev>c['eig_cutoff']*ev.max(),ev,0)
                m=q@torch.diag(ev.sqrt())@q.T
        np.savez_compressed(path/'final.npz',m0=m0.numpy(),m=m.cpu().numpy(),pred=pred.cpu().numpy(),y=yt.cpu().numpy(),history=np.stack(preds))
        np.savez_compressed(path/'states.npz',steps=state_steps,matrices=np.stack(states),agop0=a0)
    except Exception:
        status='failed';error=traceback.format_exc();(path/'error.txt').write_text(error)
    result=dict(status=status,config=c,n_rows=len(rows),final=rows[-1] if rows else None,error=error,elapsed_s=time.monotonic()-start)
    (path/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(run=path.name,status=status,acc=rows[-1]['acc'] if rows else None,elapsed_s=result['elapsed_s'])),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();failures=0
    for job in json.loads(Path(args.plan).read_text()):
        name=job.pop('name');path=Path(args.output)/name
        if path.exists():raise FileExistsError(f'Refusing to overwrite or silently skip {path}')
        failures+=run(job,path)['status']!='completed'
    print(json.dumps(dict(finished=True,failures=failures)),flush=True)
