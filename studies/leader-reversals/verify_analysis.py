"""Independent checks of primary anchor results and direct witnesses for every event."""
from pathlib import Path
import collections
import csv
import hashlib
import importlib.util
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
SOURCE=HERE.parent/'rfm-mechanism'


def read_csv(name):
    return list(csv.DictReader((HERE/name).open(encoding='utf-8')))


def score_vectors(history,p):
    # Explicit alignment, independently of analyze.py's take_along_axis.
    mean=np.zeros((history.shape[0],p))
    for a in range(p):
        for offset in range(p):
            mean[:,offset]+=history[:,a,(2*a+offset)%p]/p
    yield 'profile',-1,0,mean
    for a in range(p):
        yield 'point',a,2*a%p,history[:,a,:]


def tol(s):
    return np.maximum(1e-8*np.sqrt(((s-s.mean(-1,keepdims=True))**2).mean(-1)),
                      64*np.finfo(float).eps*np.abs(s).max(-1))


def semantic_cases():
    spec=importlib.util.spec_from_file_location('tested_analyzer',HERE/'analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases={
        'tie_bridge':(np.array([[3,1,0],[2,2,0],[1,3,0]],float),0,1,1,True,True),
        'wrong_tied_labels':(np.array([[1,3,3],[1,3+1e-12,3],[1,3,3+1e-12]],float),0,0,0,False,False),
        'recovery_after_loss':(np.array([[3,1,0],[1,3,0],[3,1,0]],float),0,2,2,True,False),
        'unresolved_start':(np.array([[1,1,1],[3,2,1],[2,3,1]],float),0,1,1,False,False),
        'always_tied':(np.ones((3,3)),0,0,0,False,False),
    }
    for name,(s,label,nleader,ncorrect,lost,endlost) in cases.items():
        row,anchors,events=module.analyze_sequence(s,label,{})
        assert row['leader_replacements']==nleader,(name,row)
        assert row['correctness_flips']==ncorrect,(name,row)
        assert anchors[0]['displaced']==lost,(name,anchors)
        assert anchors[0]['displaced_at_final']==endlost,(name,anchors)
        if name=='tie_bridge':
            assert all(e['previous_step']==0 and e['step']==2 for e in events)
        if name=='unresolved_start':
            assert not anchors[0]['eligible'] and row['initial_informative_step']==1
        # Clear outcomes must survive harmless positive rescaling and offset.
        scaled=module.analyze_sequence(7*s+2,label,{})[0]
        assert scaled['leader_replacements']==nleader and scaled['correctness_flips']==ncorrect
    return list(cases)


def main():
    cases=semantic_cases()
    manifest=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
    for rec in manifest:
        path=PROJECT/rec['path']
        assert path.stat().st_size==rec['bytes'],path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256'],path
    run_rows=read_csv('run_summary.csv')
    assert len(run_rows)==844 and len({(r['phase'],r['name']) for r in run_rows})==844
    primary=[r for r in run_rows if r['phase']=='confirmation']
    assert collections.Counter(int(r['p']) for r in primary)=={17:50,23:30}
    anchors={(r['phase'],r['name'],r['view'],int(r['anchor'])):r for r in read_csv('anchor_by_run.csv')}
    sequences={(r['phase'],r['name'],r['view'],int(r['test_a'])):r for r in read_csv('sequence_summary.csv')}
    comparisons=0
    # All primary runs, every stated anchor, both individual answers and aligned mean.
    for r in primary:
        p=int(r['p']);phase=r['phase'];name=r['name']
        with np.load(SOURCE/'results'/phase/name/'final.npz',allow_pickle=False) as z:
            h=z['history']
        sums=collections.defaultdict(lambda:collections.Counter())
        for view,a,label,s in score_vectors(h,p):
            tolerance=tol(s)
            best=s.max(1)
            wrong=np.max(np.delete(s,label,axis=1),axis=1)
            m=s[:,label]-wrong
            definite=[(t,1 if m[t]>0 else -1) for t in range(60) if abs(m[t])>tolerance[t]]
            positive_to_negative=sum(x[1]==1 and y[1]==-1 for x,y in zip(definite[:-1],definite[1:]))
            negative_to_positive=sum(x[1]==-1 and y[1]==1 for x,y in zip(definite[:-1],definite[1:]))
            saved=sequences[phase,name,view,a]
            assert positive_to_negative==int(saved['correct_to_wrong']),(name,view,a)
            assert negative_to_positive==int(saved['wrong_to_correct']),(name,view,a)
            for k in [0,1,2,3,5,10,15,20,30,40,50,55,59]:
                group=np.flatnonzero(best[k]-s[k]<=tolerance[k]).tolist()
                outside=[c for c in range(p) if c not in group]
                lost_times=[]
                if outside:
                    for t in range(k+1,60):
                        if max(s[t,c] for c in outside)-max(s[t,c] for c in group)>tolerance[t]:
                            lost_times.append(t)
                counter=sums[view,k]
                counter['sequences']+=1
                counter['eligible']+=bool(outside)
                counter['unique_leaders']+=len(group)==1
                counter['displaced']+=bool(lost_times)
                counter['displaced_at_final']+=59 in lost_times
        for (view,k),counter in sums.items():
            saved=anchors[phase,name,view,k]
            for field,value in counter.items():
                assert int(saved[field])==value,(name,view,k,field,value,saved[field])
                comparisons+=1
    # Verify every published event against the raw score history, including controls.
    events=read_csv('events.csv')
    byrun=collections.defaultdict(list)
    for event in events:
        byrun[event['phase'],event['name']].append(event)
    witnesses=0
    for (phase,name),ee in byrun.items():
        p=int(ee[0]['p'])
        with np.load(SOURCE/'results'/phase/name/'final.npz',allow_pickle=False) as z:
            vectors={(view,a):(label,s) for view,a,label,s in score_vectors(z['history'],p)}
        for e in ee:
            label,s=vectors[e['view'],int(e['test_a'])]
            ts=tol(s);b=int(e['previous_step']);t=int(e['step'])
            assert 0<=b<t<len(s)
            if e['event']=='leader_group_replaced':
                inside=list(map(int,e['old_classes'].split('|')))
                outside=[c for c in range(p) if c not in inside]
                mb=max(s[b,inside])-max(s[b,outside]);mt=max(s[t,inside])-max(s[t,outside])
                assert mb>ts[b] and mt < -ts[t],e
            else:
                wrong=[c for c in range(p) if c!=label]
                mb=s[b,label]-max(s[b,wrong]);mt=s[t,label]-max(s[t,wrong])
                if e['event']=='correct_to_wrong':
                    assert mb>ts[b] and mt < -ts[t],e
                else:
                    assert mb < -ts[b] and mt>ts[t],e
            assert math.isclose(mt,float(e['current_margin']),rel_tol=1e-7,abs_tol=1e-13),e
            witnesses+=1
    # Reconcile every phase/p/anchor aggregate with its constituent runs.
    for row in read_csv('anchor_summary.csv'):
        rr=[r for key,r in anchors.items() if r['phase']==row['phase'] and r['p']==row['p']
            and r['view']==row['view'] and r['anchor']==row['anchor']]
        assert len(rr)==int(row['runs'])
        assert sum(int(r['displaced'])>0 for r in rr)==int(row['runs_with_displacement'])
        assert sum(int(r['displaced']) for r in rr)==int(row['displaced_sequences'])
    out={'passed':True,'source_hashes_verified':len(manifest),'all_manifest_inputs_unchanged':True,
         'semantic_edge_cases':cases,'independently_recomputed_primary_runs':80,
         'primary_anchor_integer_comparisons':comparisons,'all_event_witnesses_verified':witnesses,
         'aggregate_reconciliation':True,
         'limits':'Independent analysis formulas on saved outputs; not independent model training or a proof of floating-point error bounds.'}
    (HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out))


if __name__=='__main__':
    main()
