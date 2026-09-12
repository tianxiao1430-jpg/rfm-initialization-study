import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
jobs=[dict(name=f'p17_s{s}_cross',p=17,seed=s,kind='cross') for s in range(30,70)]
for p in [17,29]:
    jobs.append(dict(name=f'p{p}_none',p=p,kind='none',epsilon=0.))
    for s in range(20,25):
        for k in ['cross','circ','residual','diag']:
            jobs.append(dict(name=f'p{p}_s{s}_{k}',p=p,seed=s,kind=k))
(ROOT/'discovery_plan.json').write_text(json.dumps(jobs,indent=2))
print(json.dumps(dict(discovery_runs=len(jobs))))
