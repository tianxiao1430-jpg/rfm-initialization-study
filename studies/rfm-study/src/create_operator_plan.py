"""Create a versioned engine and deterministic training-only response probes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'src/experiment.py').read_text()
source=source.replace("def initial_direction(p,seed,kind,amount=None):","def unused_original_direction(p,seed,kind,amount=None):",1)
source=source.replace('def run(config,path):','from directions_v2 import initial_direction\n\ndef run(config,path):',1)
source=source.replace("ROOT/'src/components.py',ROOT/'vendor/rfm.py'", "ROOT/'src/components.py',ROOT/'vendor/rfm.py',ROOT/'src/experiment.py',ROOT/'src/directions_v2.py'")
path=ROOT/'src/experiment_v2.py'
if path.exists():raise FileExistsError(path)
path.write_text(source)
jobs=[]
for p in [17,23]:
    jobs.append(dict(name=f'p{p}_none',p=p,kind='none',epsilon=0.))
    for k in range(1,(p+1)//2):
        jobs.append(dict(name=f'p{p}_mode{k}',p=p,seed=k,kind='mode',epsilon=.002))
        for j in range(k+1,(p+1)//2):
            jobs.append(dict(name=f'p{p}_pair{k}_{j}',p=p,seed=k,kind='mode_pair',amount=j,epsilon=.002))
# These checks are not used to fit the quadratic tensor.
for p in [17,23]:
    for k,j in [(1,2),(1,3),(2,3)]:
        jobs.append(dict(name=f'p{p}_difference{k}_{j}',p=p,seed=k,kind='mode_difference',amount=j,epsilon=.002))
    jobs.append(dict(name=f'p{p}_mode1_half',p=p,seed=1,kind='mode',epsilon=.001))
(ROOT/'operator_plan.json').write_text(json.dumps(jobs,indent=2))
print(json.dumps(dict(runs=len(jobs))))
