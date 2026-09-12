"""Read-only publication checks using the Python standard library."""
import csv, json, math
from collections import Counter
from pathlib import Path
from statistics import mean
root = Path(__file__).resolve().parent
def rows(name):
    with (root / 'results' / name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))
def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)
runs,steps = rows('all_runs.csv'),rows('all_steps.csv')
confirm,inter = rows('confirmation.csv'),rows('intervention.csv')
assert (len(runs),len(steps),len(confirm),len(inter)) == (844,47940,80,30)
counts = Counter((r['phase'],r['run']) for r in steps)
last = {}
for r in steps:
    key = (r['phase'],r['run'])
    if key not in last or int(r['iteration']) > int(last[key]['iteration']): last[key]=r
assert len(last)==len(runs)
for r in runs:
    key=(r['phase'],r['name'])
    assert counts[key]==int(r['n_rows'])
    for field in ['acc','mse','mean_margin','min_margin']:
        close(r[field],last[key][field])
lookup={(r['phase'],r['name']):r for r in runs}
report={'runs':len(runs),'per_step_records':len(steps),'confirmation':{}}
for p,n,expected in [(17,50,4.22),(23,30,6.58)]:
    subset=[r for r in confirm if int(r['p'])==p]
    assert len(subset)==n
    for r in subset:
        close(r['absolute_error'],abs(float(r['actual_acc'])-float(r['predicted_acc'])))
        close(r['actual_acc'],lookup[('confirmation',f"p{p}_s{r['seed']}_cross")]['acc'])
    mae=100*mean(float(r['absolute_error']) for r in subset)
    assert abs(mae-expected)<0.005
    report['confirmation'][str(p)]={'directions':n,'mae_pp':mae,
        'mean_accuracy_percent':100*mean(float(r['actual_acc']) for r in subset)}
for r in inter:
    close(r['base_acc'],lookup[('confirmation',f"p{r['p']}_s{r['seed']}_cross")]['acc'])
    for label in ['phase_best','phase_worst','remove']:
        control=mean(float(r[f'{label}_control{i}']) for i in range(3))
        close(r[f'{label}_control_mean'],control)
        close(r[f'{label}_effect'],float(r[label])-control)
        close(r[label],lookup[('intervention',f"p{r['p']}_s{r['seed']}_{label}")]['acc'])
    close(r['best_minus_worst'],float(r['phase_best'])-float(r['phase_worst']))
report['intervention']={label:100*mean(float(r[label]) for r in inter)
    for label in ['phase_best','phase_worst','remove','best_minus_worst']}
assert abs(report['intervention']['phase_best']-97.6)<0.05
assert abs(report['intervention']['phase_worst']-4.0)<0.05
report['passed']=True
report['scope']='Released CSV consistency and key manuscript values; no new training or independent implementation.'
print(json.dumps(report,indent=2))
