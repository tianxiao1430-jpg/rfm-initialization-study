"""Apply the frozen response tensor and isotonic rule without training a new RFM."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json
from pathlib import Path
import numpy as np
from experiment import initial_direction
ROOT=Path(__file__).resolve().parents[1]
def predict(p,seed):
    path=ROOT/f'frozen/quadratic_p{p}.npz'
    if not path.exists():raise ValueError(f'No frozen response tensor for p={p}; available p=17,23')
    model=np.load(path);e=initial_direction(p,seed,'cross').numpy();a=np.einsum('kij,ij->k',model['basis'],e)
    v=np.einsum('i,cij,j->c',a,model['q'],a)
    signal=float((v[0]-v[1:].max())/np.sqrt(np.mean((v-v.mean())**2)))
    cal=json.loads((ROOT/'frozen/calibration_candidate.json').read_text())
    return dict(p=p,seed=seed,signal=signal,predicted_acc=float(np.interp(signal,cal['x'],cal['y'])),
                circulant_fraction=float(np.sum(a*a)),profile=(.01**2*v).tolist(),
                scope='default epsilon=.01, Gaussian denominator=12.5, centered AGOP, 60 evaluations; finite-response forecast, not a guarantee')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--p',type=int,required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--output')
    args=a.parse_args();out=predict(args.p,args.seed);text=json.dumps(out,indent=2)
    if args.output:
        path=Path(args.output)
        if path.exists():raise FileExistsError(path)
        path.write_text(text)
    print(text)
