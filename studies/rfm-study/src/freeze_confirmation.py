"""Freeze forecasts, interventions and boundary plan BEFORE held-out outcomes."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv,hashlib,json,itertools,datetime
from pathlib import Path
import numpy as np
from experiment import initial_direction
from components import circulant_projection,features,cross_matrix
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def score_profile(a,q):
    v=np.einsum('...i,cij,...j->...c',a,q,a)
    margin=v[...,0]-v[...,1:].max(-1)
    scale=np.sqrt(np.mean((v-v.mean(-1,keepdims=True))**2,axis=-1))
    return margin/scale,v
def main():
    frozen=ROOT/'frozen';frozen.mkdir(exist_ok=False)
    for name in ['quadratic_p17.npz','quadratic_p23.npz','calibration_candidate.json']:
        (frozen/name).write_bytes((ROOT/'models'/name).read_bytes())
    cal=json.loads((frozen/'calibration_candidate.json').read_text())
    init=ROOT/'initializations';init.mkdir(exist_ok=False)
    forecasts=[];profiles={};confirmation=[];interventions=[];geometry=[];boundaries=[]
    for p,end in [(17,150),(23,130)]:
        model=np.load(frozen/f'quadratic_p{p}.npz');q=model['q'];basis=model['basis'];n=len(basis)
        for seed in range(100,end):
            name=f'p{p}_s{seed}_cross';e=initial_direction(p,seed,'cross').numpy()
            a=np.einsum('kij,ij->k',basis,e);signal,v=score_profile(a,q)
            pred=float(np.interp(signal,cal['x'],cal['y']));feats=features(e[:p,p:])
            forecasts.append(dict(p=p,seed=seed,signal=float(signal),predicted_acc=pred,constant_acc=cal['constant'],**feats))
            profiles[name]=.01**2*v
            confirmation.append(dict(name=name,p=p,seed=seed,kind='cross'))
            if seed>= (120 if p==17 else 110):continue
            circ=np.einsum('k,kij->ij',a,basis);res=e-circ;energy=float(np.sum(a*a));unit=a/np.sqrt(energy)
            # Exhaustive sign search up to a global sign; known task class at a=0 is 0.
            signs=np.array([(1,)+tail for tail in itertools.product([-1,1],repeat=n-1)])
            candidates=signs*np.abs(a);candidates[:,0]*=np.sign(a[0])
            scores,_=score_profile(candidates,q)
            choices={'phase_best':candidates[np.argmax(scores)],'phase_worst':candidates[np.argmin(scores)]}
            variants={}
            for family,target_a in choices.items():
                target=np.einsum('k,kij->ij',target_a,basis)+res
                variants[family]=target
                cosine=float(np.clip(np.dot(a,target_a)/energy,-1,1))
                for rep in range(3):
                    rng=np.random.default_rng(p*10000000+seed*1000+(0 if family=='phase_best' else 100)+rep)
                    tangent=rng.normal(size=n);tangent-=np.dot(tangent,unit)*unit;tangent/=np.linalg.norm(tangent)
                    ca=np.sqrt(energy)*(cosine*unit+np.sqrt(max(0,1-cosine*cosine))*tangent)
                    variants[f'{family}_control{rep}']=np.einsum('k,kij->ij',ca,basis)+res
            variants['remove']=res/np.linalg.norm(res)
            cosine=float(np.sum(e*variants['remove']))
            for rep in range(3):
                rng=np.random.default_rng(p*10000000+seed*1000+200+rep)
                b=rng.normal(size=(p,p));v=cross_matrix((b-b.T)/2)
                v-=np.sum(v*e)*e;v/=np.linalg.norm(v)
                variants[f'remove_control{rep}']=cosine*e+np.sqrt(max(0,1-cosine*cosine))*v
            for variant,direction in variants.items():
                file=f'p{p}_s{seed}_{variant}.npz';np.savez_compressed(init/file,direction=direction)
                v_a=np.einsum('kij,ij->k',basis,direction);signal,_=score_profile(v_a,q) if np.linalg.norm(v_a)>1e-12 else (None,None)
                geometry.append(dict(p=p,seed=seed,variant=variant,file=file,sha256=sha(init/file),norm=float(np.linalg.norm(direction)),
                                     distance=float(np.linalg.norm(direction-e)),circulant_fraction=float(np.sum(v_a*v_a)),
                                     spectral_amplitude_max_error=float(np.max(abs(abs(v_a)-abs(a)))),predicted_signal=None if signal is None else float(signal)))
                interventions.append(dict(name=file[:-4],p=p,seed=seed,kind='custom',amount=file))
    for p in [17,23]:
        for seed in range(100,110):
            variants=[('paper',dict(formula='paper')),('uncentered',dict(centering=False)),('paper_uncentered',dict(formula='paper',centering=False)),
                      ('bw15',dict(bandwidth=1.5)),('bw20',dict(bandwidth=2.0)),('bw30',dict(bandwidth=3.0)),
                      ('eps001',dict(epsilon=.001)),('eps03',dict(epsilon=.03))]
            for name,settings in variants:
                boundaries.append(dict(name=f'p{p}_s{seed}_{name}',p=p,seed=seed,kind='cross',**settings))
    for name,jobs in [('confirmation',confirmation),('intervention',interventions),('boundary',boundaries)]:
        (frozen/f'{name}_plan.json').write_text(json.dumps(jobs,indent=2))
    (frozen/'forecasts.json').write_text(json.dumps(forecasts,indent=2))
    with (frozen/'forecasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(forecasts[0]));w.writeheader();w.writerows(forecasts)
    np.savez_compressed(frozen/'forecast_profiles.npz',**profiles)
    (frozen/'intervention_geometry.json').write_text(json.dumps(geometry,indent=2))
    files=list(frozen.iterdir())+list(init.iterdir())+[Path(__file__),ROOT/'src/experiment.py',ROOT/'src/components.py']
    manifest={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)}
    (frozen/'freeze_manifest.json').write_text(json.dumps(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=manifest,
        counts=dict(confirmation=len(confirmation),intervention=len(interventions),boundary=len(boundaries)),
        assertion='No reserved seed outcomes have been generated or read. Forecasts and all interventions are fixed before running these plans.'),indent=2))
    print(json.dumps(dict(confirmation=len(confirmation),intervention=len(interventions),boundary=len(boundaries),freeze_sha256=sha(frozen/'freeze_manifest.json'))))
if __name__=='__main__':main()
