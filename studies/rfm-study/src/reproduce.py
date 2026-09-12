"""Rerun existing plans into a NEW output directory; never overwrite evidence."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLANS={'discovery':('discovery_plan.json','experiment.py'),'probe':('probe_plan.json','experiment.py'),
       'operator':('operator_plan.json','experiment_v2.py'),'confirmation':('frozen/confirmation_plan.json','experiment.py'),
       'intervention':('frozen/intervention_plan.json','experiment_v3.py'),'boundary':('frozen/boundary_plan.json','experiment.py')}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=list(PLANS)+['all','smoke'],required=True);p.add_argument('--output',required=True)
    args=p.parse_args();dest=Path(args.output).resolve()
    if dest.exists():raise FileExistsError('Choose a new directory: '+str(dest))
    dest.mkdir(parents=True);study=dest/'rfm-mechanism';study.mkdir()
    for folder in ['src','vendor','frozen','initializations','models','reference_v2']:
        shutil.copytree(ROOT/folder,study/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for f in ROOT.iterdir():
        if f.is_file() and (f.suffix in ['.md','.txt'] or f.name in ['LICENSE','discovery_plan.json','probe_plan.json','operator_plan.json']):shutil.copy2(f,study/f.name)
    shutil.copytree(ROOT/'reference_v2',dest/'rfm-numerics/v2/results/followup')
    if args.phase=='smoke':
        import numpy as np
        jobs=[('confirmation','p17_s100_cross','experiment.py'),('intervention','p17_s100_phase_best','experiment_v3.py')]
        for phase,name,engine in jobs:
            plan=json.loads((study/PLANS[phase][0]).read_text());chosen=[j for j in plan if j['name']==name]
            tmp=dest/f'{phase}_smoke.json';tmp.write_text(json.dumps(chosen,indent=2))
            subprocess.run([sys.executable,'-u',str(study/'src'/engine),'--plan',str(tmp),'--output',str(study/'results'/phase)],check=True)
            before=np.load(ROOT/f'results/{phase}/{name}/final.npz');after=np.load(study/f'results/{phase}/{name}/final.npz')
            for key in ['m0','m','history','pred']:assert np.array_equal(before[key],after[key]),(name,key)
        (dest/'smoke_check.json').write_text(json.dumps(dict(passed=True,runs=2,scope='clean copied package: ordinary and custom initialization, full 60-evaluation trajectory bitwise equality'),indent=2))
        print('Clean-package smoke reproduction passed.');sys.exit(0)
    for phase in (list(PLANS) if args.phase=='all' else [args.phase]):
        plan,engine=PLANS[phase]
        subprocess.run([sys.executable,'-u',str(study/'src'/engine),'--plan',str(study/plan),'--output',str(study/'results'/phase)],check=True)
    if args.phase=='all':
        for script in ['preflight.py','analyze_discovery.py','taylor.py','probe_modes.py','validate_mechanism.py','fit_operator.py','analyze_confirmation.py','analyze_final.py','numerical_check.py','make_figures.py','audit.py','build_report.py']:
            subprocess.run([sys.executable,'-u',str(study/'src'/script)],check=True)
    print('Reproduced outputs: '+str(study/'results'))
