import json
from pathlib import Path
import numpy as np
from components import features,circulant_projection
ROOT=Path(__file__).resolve().parents[1]
old=ROOT.parent/'rfm-numerics/v2/results/followup'
rows=[]
for p in [17,29]:
    for seed in range(20,25):
        path=old/f'p{p}_s{seed}_cross'
        arr=np.load(path/'final.npz');b=arr['m0'][:p,p:]
        c=circulant_projection(b);r=b-c
        assert np.max(np.abs(c+c.T))<1e-15
        assert abs(np.sum(c*r))<1e-15
        assert np.max(np.abs(circulant_projection(c)-c))<1e-15
        assert all(abs(v-features(-b)[k])<1e-12 for k,v in features(b).items())
        config=json.loads((path/'summary.json').read_text())
        mf=arr['m'];u=np.block([[np.eye(p),np.eye(p)],[np.eye(p),-np.eye(p)]])/2**.5
        transformed=u.T@mf@u;coupling=transformed[:p,p:]
        sym=(coupling+coupling.T)/2;skew=(coupling-coupling.T)/2
        rows.append(dict(p=p,seed=seed,acc=config['final']['acc'],**features(b),
                         final_plus_minus_symmetric_coupling=float(np.linalg.norm(sym)),
                         final_plus_minus_skew_coupling=float(np.linalg.norm(skew))))
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/preflight.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
