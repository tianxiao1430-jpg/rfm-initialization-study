"""Verify the downloaded supplement snapshot using only Python's standard library."""
from pathlib import Path
import hashlib,json,csv
root=Path(__file__).resolve().parent
records=json.loads((root/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
for rec in records:
    path=root/rec['path']
    assert path.stat().st_size==rec['bytes'],rec['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256'],rec['path']
def count(name):
    with (root/name).open(encoding='utf-8',newline='') as f:return sum(1 for _ in csv.DictReader(f))
assert count('run_summary.csv')==844
assert count('events.csv')==56452
assert count('primary_all_steps_by_run.csv')==9600
assert count('matched_amplitude_comparison.csv')==20
assert count('example_traces.csv')==180
assert count('endpoint_point_history.csv')==1540
assert count('endpoint_stratification.csv')==9
assert count('saved_metric_coverage.csv')==80
for name in ['verification.json','timeline-verification.json']:
    assert json.loads((root/name).read_text(encoding='utf-8'))['passed']
print(json.dumps({'passed':True,'verified_files':len(records),'runs':844,'event_records':56452}))
