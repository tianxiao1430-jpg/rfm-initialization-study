"""Post hoc, read-only analysis of all recorded original RFM score histories."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'rfm-mechanism'
ANCHORS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 55, 59]
RTOL = 1e-8


def write_csv(name, rows):
    path = OUT / name
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return {'path': path.relative_to(OUT.parent.parent).as_posix(), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def tolerances(s, rtol=RTOL):
    centered = s - s.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=-1))
    tol = np.maximum(rtol * rms, 64 * np.finfo(np.float64).eps * np.max(np.abs(s), axis=-1))
    return rms, tol


def sequences(history, p):
    labels = (2 * np.arange(p)) % p
    aligned = np.take_along_axis(history, ((labels[:, None] + np.arange(p)) % p)[None, :, :], axis=2)
    yield 'profile', -1, 0, aligned.mean(axis=1)
    for a in range(p):
        yield 'point', a, int(labels[a]), history[:, a, :]


def analyze_sequence(s, truth, base, rtol=RTOL, detailed=True):
    rms, tol = tolerances(s, rtol)
    top = s.max(axis=1)
    winners = top[:, None] - s <= tol[:, None]
    n, p = s.shape
    wrong = s.copy()
    wrong[:, truth] = -np.inf
    margin = s[:, truth] - wrong.max(axis=1)
    state = np.where(margin > tol, 1, np.where(margin < -tol, -1, 0))
    events = []
    definite = np.flatnonzero(state)
    for before, after in zip(definite[:-1], definite[1:]):
        if state[before] != state[after]:
            events.append(dict(base, event='correct_to_wrong' if state[before] == 1 else 'wrong_to_correct',
                previous_step=int(before), step=int(after), old_classes=str(truth),
                new_classes=str(truth) if state[after] == 1 else '|'.join(map(str, np.flatnonzero(winners[after]))),
                previous_margin=float(margin[before]), current_margin=float(margin[after]),
                previous_tolerance=float(tol[before]), current_tolerance=float(tol[after]),
                current_normalized_margin=float(margin[after] / rms[after]) if rms[after] else 0.))
    # Preserve a tied incumbent group. A new group must strictly beat every member.
    start = next((t for t in range(n) if winners[t].sum() < p), None)
    if start is not None:
        incumbent = winners[start].copy()
        previous = start
        for t in range(start + 1, n):
            group_margin = s[t, incumbent].max() - s[t, ~incumbent].max()
            if group_margin > tol[t]:
                previous = t
            elif group_margin < -tol[t]:
                previous_margin = s[previous, incumbent].max() - s[previous, ~incumbent].max()
                assert previous_margin > tol[previous]
                events.append(dict(base, event='leader_group_replaced', previous_step=previous, step=t,
                    old_classes='|'.join(map(str, np.flatnonzero(incumbent))),
                    new_classes='|'.join(map(str, np.flatnonzero(winners[t]))),
                    previous_margin=float(previous_margin), current_margin=float(group_margin),
                    previous_tolerance=float(tol[previous]), current_tolerance=float(tol[t]),
                    current_normalized_margin=float(group_margin / rms[t]) if rms[t] else 0.))
                incumbent = winners[t].copy()
                previous = t
    anchors = []
    for k in ANCHORS:
        if k >= n:
            continue
        w = winners[k]
        eligible = bool(w.sum() < p)
        first = None
        end_lost = False
        if eligible and k + 1 < n:
            future_gap = s[k+1:, ~w].max(axis=1) - s[k+1:, w].max(axis=1)
            lost = future_gap > tol[k+1:]
            ix = np.flatnonzero(lost)
            first = int(k + 1 + ix[0]) if len(ix) else None
            end_lost = bool(lost[-1])
        anchors.append(dict(base, anchor=k, horizon=n-1, eligible=eligible,
            unique_leader=bool(w.sum()==1), winner_count=int(w.sum()),
            anchor_correct_state=int(state[k]), final_correct_state=int(state[-1]),
            displaced=first is not None, first_displaced_step=first, displaced_at_final=end_lost))
    ce = [e for e in events if e['event'] != 'leader_group_replaced']
    le = [e for e in events if e['event'] == 'leader_group_replaced']
    row = dict(base, horizon=n-1, initial_informative_step=start,
        raw_argmax_changes=int(np.count_nonzero(np.diff(s.argmax(axis=1)))),
        leader_replacements=len(le), correctness_flips=len(ce),
        correct_to_wrong=sum(e['event']=='correct_to_wrong' for e in ce),
        wrong_to_correct=sum(e['event']=='wrong_to_correct' for e in ce),
        last_leader_replacement=max((e['step'] for e in le), default=None),
        last_correctness_flip=max((e['step'] for e in ce), default=None),
        final_correct_state=int(state[-1]), final_winner_count=int(winners[-1].sum()),
        final_normalized_correct_margin=float(margin[-1]/rms[-1]) if rms[-1] else 0.)
    return row, anchors, events


def run_aggregate(base, seqs):
    out = dict(base)
    for view in ['point', 'profile']:
        rr = [r for r in seqs if r['view'] == view]
        out[view+'_sequences'] = len(rr)
        for field in ['raw_argmax_changes', 'leader_replacements', 'correctness_flips', 'correct_to_wrong', 'wrong_to_correct']:
            out[view+'_'+field] = sum(r[field] for r in rr)
            out[view+'_any_'+field] = sum(r[field] > 0 for r in rr)
        for field in ['last_leader_replacement', 'last_correctness_flip']:
            out[view+'_'+field] = max((r[field] for r in rr if r[field] is not None), default=None)
    return out


def aggregate_anchors(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['phase'], r['p'], r['name'], r['variant'], r['view'], r['anchor'], r['horizon'], r['rtol']].append(r)
    byrun = []
    for (phase,p,name,variant,view,k,horizon,rtol), rr in sorted(groups.items()):
        byrun.append(dict(phase=phase,p=p,name=name,variant=variant,view=view,anchor=k,horizon=horizon,rtol=rtol,
            sequences=len(rr), eligible=sum(r['eligible'] for r in rr),
            unique_leaders=sum(r['unique_leader'] for r in rr), displaced=sum(r['displaced'] for r in rr),
            displaced_at_final=sum(r['displaced_at_final'] for r in rr),
            first_displaced_step=min((r['first_displaced_step'] for r in rr if r['displaced']),default=None),
            latest_first_displaced_step=max((r['first_displaced_step'] for r in rr if r['displaced']),default=None)))
    groups = collections.defaultdict(list)
    for r in byrun:
        groups[r['phase'],r['p'],r['view'],r['anchor'],r['horizon'],r['rtol']].append(r)
    summary = []
    for (phase,p,view,k,horizon,rtol), rr in sorted(groups.items()):
        summary.append(dict(phase=phase,p=p,view=view,anchor=k,horizon=horizon,rtol=rtol,
            runs=len(rr), runs_with_eligible_leader=sum(r['eligible']>0 for r in rr),
            runs_with_displacement=sum(r['displaced']>0 for r in rr),
            runs_displaced_at_final=sum(r['displaced_at_final']>0 for r in rr),
            sequences=sum(r['sequences'] for r in rr), eligible_sequences=sum(r['eligible'] for r in rr),
            displaced_sequences=sum(r['displaced'] for r in rr),
            mean_within_run_displaced_fraction=float(np.mean([r['displaced']/r['sequences'] for r in rr]))))
    return byrun, summary


def main():
    source_rows = list(csv.DictReader((ROOT/'results/all_runs.csv').open(encoding='utf-8')))
    assert len(source_rows) == 844
    assert len({(r['phase'],r['name']) for r in source_rows}) == 844
    plan = json.loads((ROOT/'frozen/confirmation_plan.json').read_text())
    assert {r['name'] for r in source_rows if r['phase']=='confirmation'} == {r['name'] for r in plan}
    assert all(int(r['n_rows']) == (6 if r['phase']=='probe' else 60) for r in source_rows)
    manifest = [digest(ROOT/'results/all_runs.csv'), digest(ROOT/'frozen/confirmation_plan.json')]
    sequences_out, anchors_out, events_out, runs_out = [], [], [], []
    sensitivity_seq, sensitivity_anchors = [], []
    coverage = {'listed_runs':len(source_rows), 'read_runs':0,'step_records':0,'max_metric_error':{},'checks':[]}
    for index, r in enumerate(source_rows):
        path = ROOT/'results'/r['phase']/r['name']
        for filename in ['config.json','summary.json','metrics.jsonl','final.npz','data.npz']:
            manifest.append(digest(path/filename))
        config = json.loads((path/'config.json').read_text())
        summary = json.loads((path/'summary.json').read_text())
        assert summary['status']=='completed'
        p, seed, n = int(r['p']), int(r['seed']), int(r['n_rows'])
        assert config['p']==p and config['seed']==seed and config['iterations']==n
        with np.load(path/'final.npz',allow_pickle=False) as z:
            h, y, pred = z['history'],z['y'],z['pred']
        assert h.shape==(n,p,p) and h.dtype==np.float64 and np.isfinite(h).all()
        assert np.array_equal(pred,h[-1])
        assert np.array_equal(y,np.eye(p)[2*np.arange(p)%p])
        with np.load(path/'data.npz',allow_pickle=False) as z:
            pairs, mask=z['pairs'],z['train_mask']
            assert np.array_equal(pairs[~mask],np.stack([np.arange(p),np.arange(p)],axis=1))
        logs = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        assert [v['iteration'] for v in logs]==list(range(n))
        truth=y.argmax(1)
        actual_acc=(h.argmax(2)==truth).mean(1)
        tmp=h.copy();tmp[:,np.arange(p),truth]=-np.inf
        margins=h[:,np.arange(p),truth]-tmp.max(2)
        calculated={'acc':actual_acc,'mse':np.mean((h-y)**2,axis=(1,2)),
                    'mean_margin':margins.mean(1),'min_margin':margins.min(1)}
        for field, vals in calculated.items():
            err=float(np.max(np.abs(vals-np.array([v[field] for v in logs]))))
            coverage['max_metric_error'][field]=max(coverage['max_metric_error'].get(field,0),err)
            assert np.allclose(vals,[v[field] for v in logs],rtol=1e-10,atol=1e-13), (r['name'],field,err)
        assert abs(float(r['acc'])-actual_acc[-1])<1e-14
        variant = r['name'].split(f'p{p}_s{seed}_',1)[-1] if r['phase'] in ['intervention','boundary'] else r['kind']
        base=dict(phase=r['phase'],name=r['name'],p=p,seed=seed,variant=variant)
        current=[]
        for view,a,label,scores in sequences(h,p):
            info=dict(base,view=view,test_a=a,truth=label,rtol=RTOL)
            seq,aa,ee=analyze_sequence(scores,label,info)
            current.append(seq);sequences_out.append(seq);anchors_out.extend(aa);events_out.extend(ee)
            if r['phase']=='confirmation':
                sensitivity_seq.append(seq);sensitivity_anchors.extend(aa)
                for tol in [1e-10,1e-6,1e-4]:
                    sq,an,_=analyze_sequence(scores,label,dict(info,rtol=tol),tol,False)
                    sensitivity_seq.append(sq);sensitivity_anchors.extend(an)
        runs_out.append(run_aggregate(dict(base,horizon=n-1,final_acc=float(actual_acc[-1])),current))
        coverage['read_runs']+=1;coverage['step_records']+=n
        if (index+1)%100==0: print(f'Analyzed {index+1}/844 histories',flush=True)
    byrun,aggregate=aggregate_anchors(anchors_out)
    _,sensitivity=aggregate_anchors(sensitivity_anchors)
    write_csv('sequence_summary.csv',sequences_out)
    write_csv('anchor_sequences.csv',anchors_out)
    write_csv('anchor_by_run.csv',byrun)
    write_csv('anchor_summary.csv',aggregate)
    write_csv('events.csv',events_out)
    write_csv('run_summary.csv',runs_out)
    write_csv('sensitivity_anchors.csv',sensitivity)
    sensitivity_groups=[]
    for p in [17,23]:
        for view in ['point','profile']:
            for rtol in [1e-10,1e-8,1e-6,1e-4]:
                rr=[r for r in sensitivity_seq if r['p']==p and r['view']==view and r['rtol']==rtol]
                sensitivity_groups.append(dict(p=p,view=view,rtol=rtol,sequences=len(rr),
                    leader_replacements=sum(r['leader_replacements'] for r in rr),
                    correctness_flips=sum(r['correctness_flips'] for r in rr),
                    correct_to_wrong=sum(r['correct_to_wrong'] for r in rr),
                    wrong_to_correct=sum(r['wrong_to_correct'] for r in rr)))
    write_csv('sensitivity_events.csv',sensitivity_groups)
    groups=[]
    for phase,p,variant in sorted({(r['phase'],r['p'],r['variant']) for r in runs_out}):
        rr=[r for r in runs_out if (r['phase'],r['p'],r['variant'])==(phase,p,variant)]
        item=dict(phase=phase,p=p,variant=variant,runs=len(rr),horizon=rr[0]['horizon'])
        for view in ['point','profile']:
            for field in ['leader_replacements','correctness_flips','correct_to_wrong','wrong_to_correct']:
                item[view+'_runs_with_'+field]=sum(r[view+'_'+field]>0 for r in rr)
                item[view+'_events_'+field]=sum(r[view+'_'+field] for r in rr)
            for field in ['last_leader_replacement','last_correctness_flip']:
                item[view+'_'+field]=max((r[view+'_'+field] for r in rr if r[view+'_'+field] is not None),default=None)
        groups.append(item)
    write_csv('group_summary.csv',groups)
    manifest += [digest(OUT/'protocol.md'),digest(Path(__file__))]
    dump('source_manifest.json',manifest)
    assert coverage['step_records']==47940
    coverage['phases']=dict(collections.Counter(r['phase'] for r in source_rows))
    coverage['complete_60_evaluation_runs']=794
    coverage['partial_6_evaluation_runs']=50
    coverage['checks']=['all listed histories present','confirmation plan exact coverage','all labels and test ordering correct',
                        'all histories finite FP64','every history endpoint equals stored pred',
                        'all 47940 metric rows recomputed','all 844 final accuracies match published table']
    dump('coverage.json',coverage)
    dump('summary.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'post hoc original manuscript histories',
        'primary': [g for g in groups if g['phase']=='confirmation'],
        'primary_anchors':[r for r in aggregate if r['phase']=='confirmation'],
        'events':len(events_out),'sequences':len(sequences_out),'runs':len(runs_out),'rtol':RTOL,
        'warnings':['not an independent confirmation','not causal evidence','stability only through observed endpoint',
                    'points and repeated/paired runs are not independent observations','tolerance is a sensitivity rule, not a rigorous solver error bound']})
    print(json.dumps({'coverage':coverage,'primary':[g for g in groups if g['phase']=='confirmation']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
