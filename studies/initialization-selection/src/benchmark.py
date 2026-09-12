"""Frozen selection-only driver; never constructs final held-out inputs."""
from core import *
import argparse
import gc
import traceback
from datetime import datetime,timezone

def now(): return datetime.now(timezone.utc).isoformat()

def freeze():
    assert not (ROOT/'freeze.json').exists()
    assert json.loads((ROOT/'preflight/checks.json').read_text())['passed']
    paths=[ROOT/'protocol.md',ROOT/'plan.json',ROOT/'vendor/rfm.py',ROOT/'LICENSE']
    paths+=list((ROOT/'src').glob('*.py'))
    old=ROOT.parent/'arxiv-submission'
    originals={str(f.relative_to(ROOT.parent)):sha(f) for f in old.rglob('*') if f.is_file()}
    write_json(ROOT/'freeze.json',dict(utc=now(),hashes={str(f.relative_to(ROOT)):sha(f) for f in paths},
                                     old_manuscript_hashes=originals,environment=rfm.environment()))

def check_freeze():
    frozen=json.loads((ROOT/'freeze.json').read_text())
    for name,h in frozen['hashes'].items(): assert sha(ROOT/name)==h,name
    return frozen

def build_bank(p,validation):
    out=ROOT/'banks'/f'p{p}';out.mkdir(parents=True,exist_ok=False)
    data=selection_data(p,validation); bb=basis(p);d=len(bb)
    jobs=[('baseline',None,None,None)]
    jobs += [(f'single-{i}',bb[i],i,None) for i in range(d)]
    jobs += [(f'pair-{i}-{j}',(bb[i]+bb[j])/np.sqrt(2),i,j) for i in range(d) for j in range(i+1,d)]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    profiles={}; rows=[]
    for name,e,i,j in jobs:
        torch.cuda.synchronize();t=time.perf_counter()
        state=State(initializer(p,e,.002));hist,val=advance(state,60,data)
        profiles[name]=profile(hist[-1],validation,p)
        save_state(out/f'{name}.npz',state,validation_history=hist)
        torch.cuda.synchronize()
        rows.append(dict(name=name,i=i,j=j,seconds=time.perf_counter()-t,validation=val))
        print(json.dumps(dict(stage='bank',p=p,completed=len(rows),total=len(jobs))),flush=True)
    q=np.zeros((p,d,d));v0=profiles['baseline']
    for i in range(d):q[:,i,i]=(profiles[f'single-{i}']-v0)/.002**2
    for i in range(d):
        for j in range(i+1,d):
            q[:,i,j]=q[:,j,i]=(profiles[f'pair-{i}-{j}']-v0)/.002**2-.5*(q[:,i,i]+q[:,j,j])
    np.savez_compressed(out/'tensor.npz',q=q,baseline=v0)
    write_json(out/'trajectories.json',rows)
    torch.cuda.synchronize()
    summary=dict(p=p,trajectories=len(jobs),fit_count=len(jobs)*60,update_count=len(jobs)*59,
                 setup_seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                 tensor_sha256=sha(out/'tensor.npz'),status='complete',utc=now())
    write_json(out/'result.json',summary)

def run():
    check_freeze()
    assert not (ROOT/'evaluation').exists(),'Test outputs must not exist before global selection seal'
    assert not (ROOT/'runs').exists(),'Refusing implicit rerun/partial overwrite'
    (ROOT/'runs').mkdir()
    plan=json.loads((ROOT/'plan.json').read_text())
    write_json(ROOT/'run-environment.json',dict(started_utc=now(),pid=os.getpid(),environment=rfm.environment()))
    # Warmup is explicitly excluded from measured budgets and recorded.
    warm=State(initializer(7));advance(warm,3,selection_data(7,[0,1]));del warm
    for pp,conf in plan['moduli'].items():
        build_bank(int(pp),conf['validation']);gc.collect();torch.cuda.empty_cache()
    write_json(ROOT/'banks-seal.json',dict(utc=now(),files={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'banks').rglob('*') if f.is_file()}))
    for ix,job in enumerate(plan['jobs']):
        p=job['p'];data=selection_data(p,plan['moduli'][str(p)]['validation'])
        q=np.load(ROOT/'banks'/f'p{p}'/'tensor.npz')['q']
        gc.collect();torch.cuda.empty_cache()
        info,state,records=select(job['method'],p,job['seed'],data,q,ROOT/'runs'/job['name'])
        del state,records,data,q;gc.collect();torch.cuda.empty_cache()
        write_json(ROOT/'progress.json',dict(completed=ix+1,total=len(plan['jobs']),last=job,utc=now()))
        print(json.dumps(dict(stage='selection',completed=ix+1,total=len(plan['jobs']),**{k:info[k] for k in ['p','seed','method','chosen_code','fit_count','online_seconds']})),flush=True)
    assert not (ROOT/'evaluation').exists()
    check_freeze()
    files={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'runs').rglob('*') if f.is_file()}
    write_json(ROOT/'selection-seal.json',dict(utc=now(),count=len(plan['jobs']),freeze_sha256=sha(ROOT/'freeze.json'),files=files))
    print(json.dumps(dict(stage='all_choices_sealed',count=len(plan['jobs']))),flush=True)

if __name__=='__main__':
    setup();parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    try:
        freeze() if args.action=='freeze' else run()
    except Exception:
        write_json(ROOT/f'failure-{int(time.time())}.json',dict(utc=now(),traceback=traceback.format_exc()))
        raise
