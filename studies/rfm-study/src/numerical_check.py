"""Reuse the frozen V2 fixed-M audit on the new confirmation/intervention."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
import numerical_v2
numerical_v2.ROOT=ROOT  # Explicit output-root adaptation; numerical algorithm unchanged.
cases=[]
for p,seeds in [(17,[100,101,102]),(23,[100])]:
    for seed in seeds:
        cases.append((f'confirmation/p{p}_s{seed}_cross',False))
        for kind in ['phase_best','phase_worst','remove']:
            cases.append((f'intervention/p{p}_s{seed}_{kind}',p==17 and seed==100 and kind!='remove'))
results=[numerical_v2.run(path,mp_check=check) for path,check in cases]
passed=all(r['stored']['correct']==r['refined']['correct'] and ('mp_readout' not in r or r['mp_readout']['correct']==r['stored']['correct']) for r in results)
out=dict(passed=passed,scope='final readout at stored symmetrized M only; NOT arbitrary precision training',
         source_sha256=hashlib.sha256((ROOT/'vendor/numerical_v2.py').read_bytes()).hexdigest(),results=results)
(ROOT/'results/numerical_check.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(passed=passed,cases=len(results))),flush=True)
