import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut,cross_val_predict
from components import circulant_projection
ROOT=Path(__file__).resolve().parents[1]
def load():
    rows=[]
    for path in sorted((ROOT/'results/discovery').iterdir()):
        s=json.loads((path/'summary.json').read_text());c=s['config']
        rows.append(dict(name=path.name,p=c['p'],seed=c['seed'],kind=c['kind'],acc=s['final']['acc'],
                         **json.loads((path/'features.json').read_text())))
    return rows
def main():
    rows=load();out={};repeat=[]
    for p in [17,29]:
        for s in range(20,25):
            for k in ['cross','diag']:
                a=np.load(ROOT/f'results/discovery/p{p}_s{s}_{k}/final.npz')
                b=np.load(ROOT.parent/f'rfm-numerics/v2/results/followup/p{p}_s{s}_{k}/final.npz')
                repeat.append(dict(p=p,seed=s,kind=k,**{key+'_bitwise':bool(np.array_equal(a[key],b[key])) for key in ['m0','m','pred','history']},
                                   history_max_diff=float(abs(a['history']-b['history']).max())))
    out['repeat_check']=repeat
    new=[r for r in rows if r['p']==17 and 30<=r['seed']<70]
    feats=['circulant_fraction','frequency_concentration','spectral_fraction','effective_rank','centered_energy_fraction']
    y=np.array([r['acc'] for r in new]);out['new_accuracy']=[(r['seed'],r['acc']) for r in new]
    out['correlations']={f:dict(zip(['rho','p'],map(float,spearmanr([r[f] for r in new],y)))) for f in feats}
    out['loocv']={}
    for fs in [feats[:1],feats[:2],feats]:
        x=np.array([[r[f] for f in fs] for r in new])
        pr=cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=10)),x,y,cv=LeaveOneOut()).clip(0,1)
        out['loocv']['+'.join(fs)]=dict(mae=float(abs(y-pr).mean()),mse=float(((y-pr)**2).mean()))
    out['constant_loo_mae']=float(np.mean(abs(y-(y.sum()-y)/(len(y)-1))))
    dynamics=[]
    for p in [17,29]:
        for s in range(20,25):
            for k in ['cross','circ','residual','diag']:
                a=np.load(ROOT/f'results/discovery/p{p}_s{s}_{k}/states.npz')
                for t,m in zip(a['steps'],a['matrices']):
                    swapped=np.block([[m[p:,p:],m[p:,:p]],[m[:p,p:],m[:p,:p]]]);odd=(m-swapped)/2
                    diag=odd[:p,:p];b=odd[:p,p:];circ=circulant_projection(b)
                    dynamics.append(dict(p=p,seed=s,kind=k,t=int(t),diag_norm=float(np.linalg.norm(diag)),cross_norm=float(np.linalg.norm(b)),
                                         circ_norm=float(np.linalg.norm(circ)),residual_norm=float(np.linalg.norm(b-circ)),matrix_norm=float(np.linalg.norm(m))))
    out['dynamics']=dynamics
    (ROOT/'results/discovery_analysis.json').write_text(json.dumps(out,indent=2))
    with (ROOT/'results/discovery.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])+[f for f in feats if f not in rows[0]]);w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in out.items() if k!='dynamics'},indent=2))
    for p in [17,29]:
        for k in ['cross','circ','residual','diag']:
            d=[r for r in dynamics if r['p']==p and r['kind']==k and r['seed']==20]
            print(p,k,[(r['t'],*[round(r[f],8) for f in ['diag_norm','circ_norm','residual_norm']]) for r in d])
if __name__=='__main__':main()
