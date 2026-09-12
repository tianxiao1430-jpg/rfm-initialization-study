import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=[]
for s in range(30,70):
    a=np.load(ROOT/f'results/discovery/p17_s{s}_cross/final.npz');h=a['history'];y=a['y'].argmax(1)
    out.append(dict(seed=s,acc=(h.argmax(2)==y).mean(1).tolist(),spatial_norm=np.linalg.norm(h-h.mean(1,keepdims=True),axis=(1,2)).tolist()))
for t in [0,1,2,3,5,10]:
    print(t,'accuracy_mae',float(np.mean([abs(r['acc'][t]-r['acc'][-1]) for r in out])), 'same_success',np.mean([(r['acc'][t]>=.9)==(r['acc'][-1]>=.9) for r in out]))
print([(r['seed'],[r['acc'][t] for t in [0,1,2,3,5,10,59]]) for r in out])
(ROOT/'results/early_forecast_discovery.json').write_text(json.dumps(out,indent=2))
# Fixed one additional mechanistic candidate: five updates of circulant-only
# initialization, PRESERVING its amplitude from the original cross direction.
# All 40 discovery seeds are used; none of the reserved confirmation seeds.
jobs=[]
for s in range(30,70):
    feat=json.loads((ROOT/f'results/discovery/p17_s{s}_cross/features.json').read_text())
    jobs.append(dict(name=f'p17_s{s}_circ_matched_probe',p=17,seed=s,kind='circ',epsilon=.01*np.sqrt(feat['circulant_fraction']),iterations=6))
for p in [17,29]:
    for s in range(20,25):
        feat=json.loads((ROOT/f'results/discovery/p{p}_s{s}_cross/features.json').read_text())
        jobs.append(dict(name=f'p{p}_s{s}_circ_matched_probe',p=p,seed=s,kind='circ',epsilon=.01*np.sqrt(feat['circulant_fraction']),iterations=6))
(ROOT/'probe_plan.json').write_text(json.dumps(jobs,indent=2))
