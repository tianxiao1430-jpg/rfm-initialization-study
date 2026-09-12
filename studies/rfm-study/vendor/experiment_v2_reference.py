"""Mechanism follow-up. V1 implementation is imported without modification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import rfm
import numpy as np
import torch

def direction(p,seed,kind):
    if kind in ['break','preserve','generic']:
        return rfm.perturbation(p,seed,kind)
    e=rfm.perturbation(p,seed,'break')
    if kind=='diag':
        e[:p,p:]=0;e[p:,:p]=0
    elif kind=='cross':
        e[:p,:p]=0;e[p:,p:]=0
    else:raise ValueError(kind)
    return e/torch.linalg.norm(e)

def kernel(x,z,m,cfg):
    if cfg['kernel']=='quadratic':
        return 3*((x@m)@z.T)**2
    bw=cfg['bandwidth'] if cfg['formula']=='code' else (cfg['bandwidth']/2)**.5
    return rfm.kernel(x,z,m,bw)

def update(x,m,alpha,k,cfg):
    if cfg['kernel']=='quadratic':
        n,d=x.shape;c=alpha.shape[1]
        derivative=6*(x@m@x.T)
        weighted=(alpha.reshape(n,c,1)@(x@m).reshape(n,1,d)).reshape(n,c*d)
        g=(derivative.T@weighted).reshape(n,c,d)
        if cfg['centering']:g-=g.mean(0)
    else:
        bw=cfg['bandwidth'] if cfg['formula']=='code' else (cfg['bandwidth']/2)**.5
        g=rfm.gradients(x,m,alpha,k,bw,centering=cfg['centering'])
    agop=torch.zeros_like(m)
    for batch in torch.split(g,2):agop+=torch.sum(batch.transpose(1,2)@batch,dim=0)
    agop/=len(g)
    ev,q=torch.linalg.eigh(agop)
    negative=float(ev.min())
    cutoff=cfg['eig_cutoff']*ev.max()
    ev=torch.where(ev>cutoff,ev,0.)
    m=q@torch.diag(ev.sqrt())@q.T
    if cfg['project']:m=rfm.project(m)
    return m,negative

def prediction_metrics(pred,y):
    result=rfm.score(pred,y)
    centered=pred-pred.mean(1,keepdim=True)
    scale=centered.square().mean(1).sqrt()
    standard=centered/scale[:,None].clamp_min(1e-30)
    sm=rfm.score(standard,y)
    result['row_score_std_mean']=float(scale.mean())
    result['standardized_margin_mean']=sm['mean_margin']
    result['standardized_margin_min']=sm['min_margin']
    result['centered_prediction_norm']=float(torch.linalg.norm(centered))
    return result

def run(config,out):
    c={'p':29,'seed':0,'kind':'none','epsilon':0.,'formula':'code','kernel':'gaussian',
       'bandwidth':2.5,'centering':True,'project':False,'eig_cutoff':0.,'dtype':'float64',
       'device':'cuda','ridge':0.,'iterations':60,'split':'symmetric','timeout':900,**config}
    path=Path(out);path.mkdir(parents=True,exist_ok=False)
    (path/'config.json').write_text(json.dumps(c,indent=2))
    torch.set_default_dtype(getattr(torch,c['dtype']));torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    env=rfm.environment();env['v2_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (path/'environment.json').write_text(json.dumps(env,indent=2))
    started=time.monotonic();records=[];history=[];state='completed';err=None
    try:
        dt=getattr(torch,c['dtype'])
        x,y,xt,yt,_,_,data=rfm.dataset(c['p'],c['split'],c['seed'],dt,c['device'])
        m0=torch.eye(2*c['p'],dtype=torch.float64)
        if c['kind']!='none':m0+=c['epsilon']*torch.linalg.norm(m0)*direction(c['p'],c['seed'],c['kind'])
        assert torch.linalg.eigvalsh(m0).min()>0
        m=m0.to(dtype=dt,device=c['device'])
        np.savez_compressed(path/'data.npz',**data)
        for t in range(c['iterations']):
            if time.monotonic()-started>c['timeout']:raise TimeoutError('run budget exceeded')
            k=kernel(x,x,m,c)
            kr=k+c['ridge']*torch.eye(len(x),device=x.device)
            a=torch.linalg.solve(kr,y)
            pred=kernel(x,xt,m,c).T@a
            assert bool(torch.isfinite(pred).all())
            metrics=prediction_metrics(pred,yt)
            row={'iteration':t,**metrics,'relative_residual':float(torch.linalg.norm(kr@a-y)/torch.linalg.norm(y)),
                 'symmetry_defect':float(torch.linalg.norm(m-rfm.swapped(m))/torch.linalg.norm(m)),
                 'train_acc':rfm.score(k.T@a,y)['acc'],'elapsed_s':time.monotonic()-started}
            records.append(row);history.append(pred.cpu().numpy())
            with (path/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if t!=c['iterations']-1:m,_=update(x,m,a,k,c)
        np.savez_compressed(path/'final.npz',m=m.cpu().numpy(),m0=m0.numpy(),pred=pred.cpu().numpy(),y=yt.cpu().numpy(),history=np.stack(history))
    except Exception:
        state='failed';err=traceback.format_exc();(path/'error.txt').write_text(err)
    summary={'status':state,'config':c,'n_rows':len(records),'final':records[-1] if records else None,
             'elapsed_s':time.monotonic()-started,'error':err}
    (path/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({'run':path.name,'status':state,'acc':records[-1]['acc'] if records else None,
                      'margin':records[-1]['mean_margin'] if records else None,'elapsed_s':summary['elapsed_s']}),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();jobs=json.loads(Path(args.plan).read_text())
    failures=0
    for job in jobs:
        name=job.pop('name');out=Path(args.output)/name
        if (out/'summary.json').exists():continue
        result=run(job,out);failures+=result['status']!='completed'
    print(json.dumps({'finished':True,'failures':failures}),flush=True)
