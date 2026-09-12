"""Independent every-step recomputation and checks of the final research-note facts."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'rfm-mechanism'

def read(name):
    with (HERE/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def load(phase,name):
    with np.load(SOURCE/'results'/phase/name/'final.npz',allow_pickle=False) as z:
        return z['history']

def threshold(s,relative=1e-8):
    rms=np.std(s,axis=1)
    return np.maximum(relative*rms,64*np.finfo(np.float64).eps*np.max(np.abs(s),axis=1))

def aligned_mean(h,p):
    mean=np.zeros((len(h),p))
    for a in range(p):
        for d in range(p):
            mean[:,d]+=h[:,a,(2*a+d)%p]/p
    return mean

def demotions(h,p,relative=1e-8):
    total=0
    for a in range(p):
        s=h[:,a,:];label=2*a%p
        wrong=[c for c in range(p) if c!=label]
        margin=s[:,label]-s[:,wrong].max(axis=1)
        signs=np.sign(margin[np.abs(margin)>threshold(s,relative)])
        total+=int(np.sum((signs[:-1]>0)&(signs[1:]<0)))
    return total

def main():
    rows=read('primary_all_steps_by_run.csv')
    saved={(r['name'],r['view'],int(r['step'])):r for r in rows}
    assert len(saved)==len(rows)==80*2*60
    primary=[r for r in read('run_summary.csv') if r['phase']=='confirmation']
    counts=collections.defaultdict(collections.Counter)
    comparisons=0
    # Different grouping and reduction implementation from the report generator.
    for r in primary:
        p=int(r['p']);name=r['name'];h=load('confirmation',name)
        for view in ['point','profile']:
            vectors=[h[:,a,:] for a in range(p)] if view=='point' else [aligned_mean(h,p)]
            for s in vectors:
                tolerance=threshold(s)
                for k in range(60):
                    group=np.flatnonzero(s[k]>=np.max(s[k])-tolerance[k]).tolist()
                    outside=[c for c in range(p) if c not in group]
                    lost=False
                    if outside and k<59:
                        inside_max=np.maximum.reduce([s[:,c] for c in group])
                        outside_max=np.maximum.reduce([s[:,c] for c in outside])
                        lost=bool(np.any((outside_max-inside_max-tolerance)[k+1:]>0))
                    c=counts[name,view,k]
                    c['sequences']+=1;c['eligible']+=bool(outside)
                    c['unique']+=len(group)==1;c['displaced']+=lost
        for view in ['point','profile']:
            for k in range(60):
                for field,value in counts[name,view,k].items():
                    assert int(saved[name,view,k][field])==value,(name,view,k,field)
                    comparisons+=1
    summary=read('primary_all_steps_summary.csv')
    assert len(summary)==240
    for row in summary:
        rr=[r for r in rows if r['p']==row['p'] and r['view']==row['view'] and r['step']==row['step']]
        assert len(rr)==int(row['runs'])
        assert sum(int(r['displaced'])>0 for r in rr)==int(row['runs_with_displacement'])
        assert sum(int(r['displaced']) for r in rr)==int(row['displaced_sequences'])
        assert sum(int(r['eligible']) for r in rr)==int(row['eligible_sequences'])
    stable={}
    for view in ['point','profile']:
        changes=[int(r['step']) for r in rows if r['view']==view and int(r['displaced'])>0]
        stable[view]=max(changes)+1
        assert all(int(r['eligible'])==int(r['sequences']) for r in rows if r['view']==view)
    assert stable=={'point':12,'profile':10}
    # Match all twenty amplitude pairs, not just the cases selected for illustration.
    matched=read('matched_amplitude_comparison.csv');assert len(matched)==20
    for r in matched:
        p=int(r['p'])
        for phase,key,field in [('confirmation','default_name','default_demotions'),('boundary','larger_name','larger_demotions')]:
            h=load(phase,r[key])
            assert demotions(h,p)==int(r[field])
            assert demotions(h,p,1e-4)==int(r[field])
        a=json.loads((SOURCE/'results'/'confirmation'/r['default_name']/'config.json').read_text())
        b=json.loads((SOURCE/'results'/'boundary'/r['larger_name']/'config.json').read_text())
        assert a.pop('epsilon')==.01 and b.pop('epsilon')==.03 and a==b
    # Every plotted sample is checked against original scores, including RMS units.
    traces=read('example_traces.csv');assert len(traces)==180
    cache={}
    for r in traces:
        key=r['phase'],r['name']
        if key not in cache:cache[key]=load(*key)
        p=int(r['p']);a=int(r['test_a']);t=int(r['step']);truth=2*a%p
        s=cache[key][t,a,:];margin=s[truth]-max(s[c] for c in range(p) if c!=truth)
        assert np.isclose(margin,float(r['margin']),rtol=1e-12,atol=1e-15)
        assert np.isclose(margin/np.std(s),float(r['normalized_margin']),rtol=1e-12,atol=1e-15)
    events=read('events.csv')
    pc=[e for e in events if e['phase']=='confirmation' and e['view']=='point' and e['event']=='wrong_to_correct']
    facts=json.loads((HERE/'report_facts.json').read_text())
    assert facts['primary_correctness_flips']==len(pc)==1254
    assert len({(e['name'],e['test_a']) for e in pc})==1254
    assert len({e['name'] for e in pc})==74
    assert facts['flips_at_8_or_9']==sum(int(e['step']) in [8,9] for e in pc)==1168
    assert facts['after10_runs']==sum(int(r['displaced'])>0 for r in rows if r['view']=='point' and r['step']=='10')==5
    assert facts['after10_points']==sum(int(r['displaced']) for r in rows if r['view']=='point' and r['step']=='10')==7
    assert facts['matched_larger_amplitude_demotions']==sum(int(r['larger_demotions']) for r in matched)==27
    assert facts['matched_larger_amplitude_demoted_runs']==sum(int(r['larger_demotions'])>0 for r in matched)==3
    assert facts['boundary_latest_step']==max(int(e['step']) for e in events if e['view']=='point')==18
    sensitivity=read('sensitivity_anchors.csv')
    for r in sensitivity:
        if r['view']=='point' and int(r['anchor']) in [1,5]:
            assert r['runs_with_displacement']==r['runs']
        if r['view']=='profile' and int(r['anchor'])>=10:
            assert int(r['runs_with_displacement'])==0
    endpoint=read('endpoint_point_history.csv')
    assert len(endpoint)==1540
    assert all(r['initial_state']=='-1' for r in endpoint)
    correct=[r for r in endpoint if r['final_state']=='1']
    wrong=[r for r in endpoint if r['final_state']=='-1']
    assert len(correct)==1254 and len(wrong)==286
    assert all(r['correctness_flips']=='1' and r['wrong_to_correct']=='1' for r in correct)
    assert all(r['correctness_flips']=='0' and r['ever_correct']=='0' for r in wrong)
    assert {(r['name'],r['test_a']) for r in correct}=={(e['name'],e['test_a']) for e in pc}
    checkpoints=read('saved_metric_coverage.csv')
    assert len(checkpoints)==80 and all(r['saved_steps']=='0|1|2|5|10|59' for r in checkpoints)
    artifacts={}
    for name in ['研究报告.md','figures/01_leader_stability.png','figures/02_reversal_examples.png',
                 'primary_all_steps_by_run.csv','primary_all_steps_summary.csv','matched_amplitude_comparison.csv',
                 'example_traces.csv','report_facts.json','build_report.py','verify_timeline.py',
                 'review_addendum.py','endpoint_point_history.csv','endpoint_stratification.csv',
                 'saved_metric_coverage.csv','review_addendum_facts.json','review_input_manifest.json']:
        path=HERE/name;assert path.stat().st_size>0
        artifacts[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={'passed':True,'primary_runs_recomputed':80,'all_60_anchors_each_view':True,
            'integer_comparisons':comparisons,'earliest_all_remaining_anchors_stable':stable,
            'matched_amplitude_pairs_checked':20,'plotted_score_rows_checked':180,
            'facts_reconciled':True,'endpoint_stratification_reconciled':True,'artifact_sha256':artifacts,
            'limits':'Checks saved numerical outputs and reporting only; no new training or causal proof.'}
    (HERE/'timeline-verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'}))

if __name__=='__main__':main()
