import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for s in range(30,70):
    a=np.load(ROOT/f'results/discovery/p17_s{s}_cross/final.npz');base=np.load(ROOT/'results/discovery/p17_none/final.npz')['history']
    delta=a['history']-base;y=a['y'].argmax(1);final=a['pred'].argmax(1)
    rows.append(dict(seed=s,final_acc=float((final==y).mean()),
                     delta_acc=(delta.argmax(2)==y).mean(1).tolist(),
                     centered_delta_acc=((delta-delta.mean(1,keepdims=True)).argmax(2)==y).mean(1).tolist(),
                     class_agreement=(delta.argmax(2)==final).mean(1).tolist()))
for t in [0,1,2,3,5,10]:
    print(t,{key:dict(mae=float(np.mean([abs(r[key][t]-r['final_acc']) for r in rows])),success_agreement=float(np.mean([(r[key][t]>=.9)==(r['final_acc']>=.9) for r in rows]))) for key in ['delta_acc','centered_delta_acc']},'class_agreement',np.mean([r['class_agreement'][t] for r in rows]))
(ROOT/'results/delta_discovery.json').write_text(json.dumps(rows,indent=2))
