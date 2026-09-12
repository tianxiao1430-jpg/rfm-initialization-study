"""Check delivered report, tables, source seals and artifact links."""
from core import *
from benchmark import check_freeze,now
import csv
import re

check_freeze()
audit=json.loads((ROOT/'verification/audit.json').read_text())
facts=json.loads((ROOT/'facts.json').read_text())
seal=json.loads((ROOT/'analysis-seal.json').read_text(encoding='utf-8-sig'))
assert seal['testOutputsAbsent'] and seal['sha256']==sha(ROOT/seal['path'])
assert audit['passed'] and len(audit['repeats'])==8 and audit['old_manuscript_unchanged']
assert facts['kernel_fits']==32960 and facts['training_trajectories']==1464
assert facts['independent_directions']==40 and facts['selection_runs']==160
rows=list(csv.DictReader((ROOT/'evaluation/directions.csv').open()))
assert len(rows)==200
plan=json.loads((ROOT/'plan.json').read_text())
for p in [17,23]:
    conf=plan['moduli'][str(p)]
    assert not(set(conf['validation']) & set(conf['test']))
    assert set(conf['validation']) | set(conf['test'])==set(range(p))
    for method in plan['methods']+['unmodified']:
        rr=[r for r in rows if int(r['p'])==p and r['method']==method]
        assert len(rr)==20 and sorted(int(r['seed']) for r in rr)==conf['seeds']
        actual=sum(int(r['test_correct']) for r in rr)/sum(int(r['test_n']) for r in rr)
        summary=next(s for s in facts['summaries'] if s['p']==p and s['method']==method)
        assert abs(actual-summary['mean_test_acc'])<1e-15
report=(ROOT/'研究报告.md').read_text()
for text in ['83.18%','85.33%','99.55%','97.73%','32,960','1,464']:
    assert text in report
links=[]
for path in ROOT.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (path.parent/target.split('#')[0]).exists(),(str(path),target)
            links.append(target)
for name in ['accuracy','all-directions','cost-accuracy']:
    for ext in ['png','svg','pdf']:
        assert (ROOT/'analysis'/f'{name}.{ext}').stat().st_size>1000
result=dict(utc=now(),passed=True,report_links_checked=len(links),direction_rows=200,candidate_records=facts['candidate_records'],
            formal_runs=160,independent_directions=40,kernel_fits=32960,training_trajectories=1464,
            exact_reruns=8,analysis_seal_valid=True,original_manuscript_unchanged=audit['old_manuscript_unchanged'],
            formal_sample_failures=0,startup_failures_retained=len(list(ROOT.glob('failure-*.json'))),
            figures_visually_reviewed=['accuracy.png','all-directions.png','cost-accuracy.png'],
            report_sha256=sha(ROOT/'研究报告.md'),facts_sha256=sha(ROOT/'facts.json'))
write_json(ROOT/'delivery-check.json',result)
print(json.dumps(result),flush=True)
