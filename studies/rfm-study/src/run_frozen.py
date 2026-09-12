"""Verify pre-outcome seals, then execute the three fixed plans sequentially."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    frozen=ROOT/'frozen';manifest=json.loads((frozen/'freeze_manifest.json').read_text())
    for file,digest in manifest['files'].items():assert sha(ROOT/file)==digest,file
    geometry=json.loads((frozen/'intervention_geometry.json').read_text());groups={}
    for r in geometry:
        assert abs(r['norm']-1)<1e-12
        groups.setdefault((r['p'],r['seed']),{})[r['variant']]=r
    for key,g in groups.items():
        for family in ['phase_best','phase_worst','remove']:
            for rep in range(3):
                control=g[f'{family}_control{rep}']
                assert abs(control['distance']-g[family]['distance'])<1e-12,(key,family,rep)
                if family!='remove':assert abs(control['circulant_fraction']-g[family]['circulant_fraction'])<1e-12
            if family!='remove':assert g[family]['spectral_amplitude_max_error']<1e-12
    paths=[ROOT/'confirmation_design.md',frozen/'freeze_manifest.json']+list((ROOT/'src').glob('*.py'))
    seal=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(f.relative_to(ROOT)):sha(f) for f in paths},geometry_checks='passed',status='before confirmation execution')
    sealpath=frozen/'execution_seal.json'
    if sealpath.exists():raise FileExistsError(sealpath)
    sealpath.write_text(json.dumps(seal,indent=2))
    for phase,engine in [('confirmation','experiment.py'),('intervention','experiment_v3.py'),('boundary','experiment.py')]:
        print(json.dumps(dict(phase=phase,status='starting')),flush=True)
        plan=frozen/f'{phase}_plan.json';output=ROOT/'results'/phase
        subprocess.run([sys.executable,'-u',str(ROOT/'src'/engine),'--plan',str(plan),'--output',str(output)],check=True)
        jobs=json.loads(plan.read_text())
        for job in jobs:
            summary=json.loads((output/job['name']/'summary.json').read_text());assert summary['status']=='completed',job['name']
        print(json.dumps(dict(phase=phase,status='verified_completed',runs=len(jobs))),flush=True)
    print(json.dumps(dict(all_frozen_phases='completed')),flush=True)
if __name__=='__main__':main()
