"""Portable wrapper around the unchanged study scripts."""
from pathlib import Path
import argparse,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
dest=Path(a.output).resolve()
if dest.exists():raise FileExistsError('Choose a new output directory')
dest.mkdir(parents=True)
old=dest/'rfm-mechanism';old.mkdir()
for name in ['src','vendor','initializations','models','reference_v2','frozen']:
    shutil.copytree(ROOT/'studies/rfm-study'/name,old/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
# An empty historical-manuscript directory makes the new execution's optional
# old-manuscript preservation inventory explicit; the public paper is not edited.
(dest/'arxiv-submission').mkdir()
study=dest/'rfm-init-selection';study.mkdir()
src=ROOT/'studies/initialization-selection'
for name in ['src','vendor']:
    shutil.copytree(src/name,study/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['plan.json','protocol.md','LICENSE','NOTICE.md','requirements.txt']:
    shutil.copy2(src/name,study/name)
for command in [['preflight.py'],['benchmark.py','freeze'],['benchmark.py','run'],['evaluate.py'],['verify.py'],['analyze.py'],['figure_readability.py']]:
    subprocess.run([sys.executable,str(study/'src'/command[0]),*command[1:]],check=True)
    if a.check_only:
        print('Portable preflight passed:',study/'preflight/checks.json')
        sys.exit(0)
print('Complete:',study/'facts.json')
